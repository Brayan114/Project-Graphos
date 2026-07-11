import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data import DataLoader
import math
import os
from PIL import Image

# Import renderer and policy network
from RFC_01_Differentiable_Canvas_Renderer import DifferentiableBezierRenderer
from policy_network import DifferentiableGraphosPolicy
from pretrain_policy import SyntheticStrokeDataset

class PPOTrainingLoop:
    def __init__(self, K=5, H=128, W=128, lr=3e-4, clip_eps=0.2, gamma=0.99, gae_lambda=0.95, value_coef=0.5, entropy_coef=0.01):
        """
        PPO Training Environment for Project Graphos.
        Trains the policy network to sequentially paint strokes to match target images.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.K = K # Total strokes in a trajectory (horizon T)
        self.H = H
        self.W = W
        
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Initialize Renderer & Policy
        self.renderer = DifferentiableBezierRenderer(canvas_height=H, canvas_width=W, tau=0.5).to(self.device)
        self.policy = DifferentiableGraphosPolicy(img_size=H, num_layers=4, num_heads=4, embed_dim=128).to(self.device)
        
        # Compile model to optimize self-attention and cross-attention blocks (unlocks FlashAttention SDPA)
        if torch.cuda.is_available():
            try:
                self.policy = torch.compile(self.policy, mode="reduce-overhead")
                print("PyTorch model compilation initialized successfully.")
            except Exception as e:
                print(f"Skipping torch.compile: {e}")
                
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.scaler = torch.amp.GradScaler('cuda') # GradScaler for stable FP16 AMP training

    def collect_rollouts(self, batch_targets):
        """
        Collects trajectories for a batch of target images under PyTorch inference mode
        to prevent autograd activation leaks (resolving the VRAM leak bottleneck).
        """
        B = batch_targets.shape[0]
        
        # Initial blank white canvas [B, 4, H, W]
        canvas = torch.zeros((B, 4, self.H, self.W), device=self.device)
        canvas[:, :3, :, :] = 1.0 # White background
        canvas[:, 3:4, :, :] = 1.0 # Opaque
        
        # History lists
        obs_canvas_rgb = []
        obs_canvas_alpha = []
        obs_step_t = []
        actions_history = []
        log_probs_history = []
        rewards_history = []
        values_history = []
        
        # Pre-compute target L1 loss for reward calculation
        current_l1 = F.l1_loss(canvas[:, :3, :, :], batch_targets, reduction='none').mean(dim=[1,2,3]) # [B]
        
        # Scale coordinates for rectangular support
        scale_cp = torch.tensor([self.W, self.H], device=self.device).view(1, 1, 2)
        
        # Disable gradient computation during rollouts
        with torch.no_grad():
            for t in range(self.K):
                step_t = torch.full((B, 1), t / self.K, device=self.device)
                canvas_rgb = canvas[:, :3, :, :]
                canvas_alpha = canvas[:, 3:4, :, :]
                
                # Store detached, cloned observations for history safety
                obs_canvas_rgb.append(canvas_rgb.clone().detach())
                obs_canvas_alpha.append(canvas_alpha.clone().detach())
                obs_step_t.append(step_t.clone().detach())
                
                # Forward pass under FP16 autocast
                with torch.amp.autocast('cuda'):
                    alpha, beta, value = self.policy(batch_targets, canvas_rgb, canvas_alpha, step_t)
                
                # CRITICAL: Detach and clone value function predictions from graph to prevent CUDA Graph overwrite
                values_history.append(value.squeeze(-1).clone().detach()) # [B]
                
                # Sample actions
                dist = Beta(alpha, beta)
                action = dist.sample() # [B, 13]
                log_prob = dist.log_prob(action).sum(dim=-1) # [B]
                
                # CRITICAL: Detach and clone actions and log probabilities
                actions_history.append(action.clone().detach())
                log_probs_history.append(log_prob.clone().detach())
                
                # Render strokes
                modes = torch.ones((B, 1), device=self.device)
                
                # Aspect ratio and width constraint checks
                cp = action[:, 0:6].view(B, 3, 2) * scale_cp
                # Limit stroke width to max 12% of height to prevent canvas-filling exploitation
                w = action[:, 6:9] * (self.H * 0.12)
                c = action[:, 9:12]
                opacities = action[:, 12:13]
                
                new_canvas = self.renderer(cp, w, c, opacities, modes, canvas)
                
                # Compute step reward (L1 delta)
                new_l1 = F.l1_loss(new_canvas[:, :3, :, :], batch_targets, reduction='none').mean(dim=[1,2,3])
                
                # Scheduled paint waste penalty (regularizer decays over time t)
                reg_weight = 0.05 * (1.0 - (t / self.K))
                reg = action[:, 6:9].mean(dim=-1) * action[:, 12]
                
                # CRITICAL: Detach and clone reward
                reward = (current_l1 - new_l1) - reg_weight * reg
                rewards_history.append(reward.clone().detach())
                
                # Update canvas and L1 trackers
                canvas = new_canvas.clone().detach()
                current_l1 = new_l1.clone().detach()
                
            # Value of the final state
            final_step = torch.full((B, 1), 1.0, device=self.device)
            with torch.amp.autocast('cuda'):
                _, _, final_value = self.policy(batch_targets, canvas[:, :3, :, :], canvas[:, 3:4, :, :], final_step)
            final_value = final_value.squeeze(-1).clone().detach()
            
        # Stack rollout variables
        obs_canvas_rgb = torch.stack(obs_canvas_rgb)     # [K, B, 3, H, W]
        obs_canvas_alpha = torch.stack(obs_canvas_alpha) # [K, B, 1, H, W]
        obs_step_t = torch.stack(obs_step_t)             # [K, B, 1]
        
        actions = torch.stack(actions_history)      # [K, B, 13]
        log_probs = torch.stack(log_probs_history)  # [K, B]
        rewards = torch.stack(rewards_history)      # [K, B]
        values = torch.stack(values_history)        # [K, B]
        
        # 5. Compute Returns and GAE Advantages
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(B, device=self.device)
        
        next_value = final_value
        for t in reversed(range(self.K)):
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            next_value = values[t]
            
        rollouts = (obs_canvas_rgb, obs_canvas_alpha, obs_step_t, actions, log_probs, returns.clone().detach(), advantages.clone().detach())
        return rollouts

    def update_policy(self, batch_targets, rollouts, ppo_epochs=4, mini_batch_size=16):
        """
        Updates the policy by flattening all trajectory transitions to break inner-loop
        parameter dependencies, parallelizing updates across mini-batches under FP16 AMP.
        """
        obs_rgb, obs_alpha, obs_step, actions, old_log_probs, returns, advantages = rollouts
        K, B = actions.shape[0], actions.shape[1]
        
        # Flatten all trajectory states to run parallel updates
        flat_obs_rgb = obs_rgb.view(K*B, 3, self.H, self.W)
        flat_obs_alpha = obs_alpha.view(K*B, 1, self.H, self.W)
        flat_obs_step = obs_step.view(K*B, 1)
        flat_actions = actions.view(K*B, 13)
        flat_old_log_probs = old_log_probs.view(K*B)
        flat_returns = returns.view(K*B)
        flat_advantages = advantages.view(K*B)
        
        # Tile targets to match flattened dimensions
        flat_targets = batch_targets.repeat(K, 1, 1, 1)
        
        # Standardize advantages globally
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
        
        dataset_size = K * B
        for _ in range(ppo_epochs):
            # Shuffle batch
            permutation = torch.randperm(dataset_size)
            for start_idx in range(0, dataset_size, mini_batch_size):
                batch_indices = permutation[start_idx : start_idx + mini_batch_size]
                
                # Slice mini-batches
                t_targets = flat_targets[batch_indices]
                c_rgb = flat_obs_rgb[batch_indices]
                c_alpha = flat_obs_alpha[batch_indices]
                s_t = flat_obs_step[batch_indices]
                act = flat_actions[batch_indices]
                old_log_p = flat_old_log_probs[batch_indices]
                ret = flat_returns[batch_indices]
                adv = flat_advantages[batch_indices]
                
                # Autocast policy forward pass in FP16
                with torch.amp.autocast('cuda'):
                    alpha, beta, value = self.policy(t_targets, c_rgb, c_alpha, s_t)
                    value = value.squeeze(-1) # [B]
                    
                    dist = Beta(alpha, beta)
                    new_log_p = dist.log_prob(act).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1)
                    
                    ratio = torch.exp(new_log_p - old_log_p)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    value_loss = F.mse_loss(value, ret)
                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
                    
                self.optimizer.zero_grad()
                
                # Backprop using AMP GradScaler
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()

def run_rl_sprint(epochs=5, batch_size=8, K=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Project Graphos PPO RL Sprint on: {device.type.upper()}")
    
    # Create dataset (128x128 resolution for RL training speed)
    dataset = SyntheticStrokeDataset(size=120, K=K, H=128, W=128, device=device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    trainer = PPOTrainingLoop(K=K, H=128, W=128)
    
    # Load pre-trained weights if available to warm start
    if os.path.exists("pretrained_graphos_policy.pth"):
        print("Warm-starting policy using 'pretrained_graphos_policy.pth'...")
        try:
            trainer.policy.load_state_dict(torch.load("pretrained_graphos_policy.pth", map_location=device), strict=False)
            print("Successfully loaded pre-trained weights.")
        except Exception as e:
            print(f"Failed to load weights: {e}")
            
    print("\nStarting PPO optimization loops...")
    
    for epoch in range(epochs):
        epoch_reward = 0.0
        batches_count = 0
        
        for target_img, _, _, _ in loader:
            target_img = target_img.to(device)
            
            # Rollout collection (Safe from VRAM leaks and CUDA Graph overwrites)
            rollouts = trainer.collect_rollouts(target_img)
            rewards = rollouts[3] # [K, B]
            
            # Flattened, parallel update
            trainer.update_policy(target_img, rollouts)
            
            epoch_reward += rewards.sum(dim=0).mean().item()
            batches_count += 1
            
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Avg Trajectory Reward: {epoch_reward/batches_count:.4f}")
        
        # Save checkpoint weights
        torch.save(trainer.policy.state_dict(), "graphos_policy_rl.pth")
        
    print("\n🎉 PPO Training sprint complete! Policy saved to 'graphos_policy_rl.pth'.")

if __name__ == "__main__":
    run_rl_sprint(epochs=5, batch_size=8, K=5)
