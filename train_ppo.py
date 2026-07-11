import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
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
        
        # 1. Initialize Renderer & Policy
        self.renderer = DifferentiableBezierRenderer(canvas_height=H, canvas_width=W, tau=0.5).to(self.device)
        self.policy = DifferentiableGraphosPolicy(img_size=H, num_layers=4, num_heads=4, embed_dim=128).to(self.device)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
    def collect_rollouts(self, batch_targets):
        """
        Collects trajectories for a batch of target images.
        Args:
            batch_targets: [B, 3, H, W]
        Returns:
            states: List of dict states
            actions: [K, B, 13]
            log_probs: [K, B]
            rewards: [K, B]
            values: [K, B]
            returns: [K, B]
            advantages: [K, B]
        """
        B = batch_targets.shape[0]
        
        # Initial blank white canvas [B, 4, H, W]
        canvas = torch.zeros((B, 4, self.H, self.W), device=self.device)
        canvas[:, :3, :, :] = 1.0 # White background
        canvas[:, 3:4, :, :] = 1.0 # Opaque
        
        # History lists
        states_history = [] # Will store (target, canvas_rgb, canvas_alpha, step_t)
        actions_history = []
        log_probs_history = []
        rewards_history = []
        values_history = []
        
        # Pre-compute target L1 loss for reward calculation
        current_l1 = F.l1_loss(canvas[:, :3, :, :], batch_targets, reduction='none').mean(dim=[1,2,3]) # [B]
        
        # Autoregressive drawing loop
        for t in range(self.K):
            step_t = torch.full((B, 1), t / self.K, device=self.device)
            canvas_rgb = canvas[:, :3, :, :]
            canvas_alpha = canvas[:, 3:4, :, :]
            
            # Store state tuple
            states_history.append((canvas_rgb.detach(), canvas_alpha.detach(), step_t.detach()))
            
            # 1. Run policy forward pass
            alpha, beta, value = self.policy(batch_targets, canvas_rgb, canvas_alpha, step_t)
            values_history.append(value.squeeze(-1)) # [B]
            
            # 2. Sample actions from Beta distribution
            dist = Beta(alpha, beta)
            action = dist.sample() # [B, 13]
            log_prob = dist.log_prob(action).sum(dim=-1) # [B]
            
            actions_history.append(action)
            log_probs_history.append(log_prob)
            
            # 3. Render strokes (using Draw mode: 1.0)
            modes = torch.ones((B, 1), device=self.device)
            
            # Scale actions to pixel coordinates
            cp = action[:, 0:6].view(B, 3, 2) * self.H
            w = action[:, 6:9] * self.H
            c = action[:, 9:12]
            opacities = action[:, 12:13]
            
            new_canvas = self.renderer(cp, w, c, opacities, modes, canvas)
            
            # 4. Compute Step-wise Delta Reward
            new_l1 = F.l1_loss(new_canvas[:, :3, :, :], batch_targets, reduction='none').mean(dim=[1,2,3]) # [B]
            
            # Paint-wasting regularization: penalize large widths and opacities
            # regularizer = width_mean * opacity
            reg = action[:, 6:9].mean(dim=-1) * action[:, 12] # [B]
            
            reward = (current_l1 - new_l1) - 0.05 * reg # [B]
            rewards_history.append(reward)
            
            # Update canvas and L1 tracker
            canvas = new_canvas.detach()
            current_l1 = new_l1.detach()
            
        # Stack histories along the time dimension K
        actions = torch.stack(actions_history)      # [K, B, 13]
        log_probs = torch.stack(log_probs_history)  # [K, B]
        rewards = torch.stack(rewards_history)      # [K, B]
        values = torch.stack(values_history)        # [K, B]
        
        # Calculate Value estimate of final state (needed for GAE edge case)
        with torch.no_grad():
            final_step = torch.full((B, 1), 1.0, device=self.device)
            _, _, final_value = self.policy(batch_targets, canvas[:, :3, :, :], canvas[:, 3:4, :, :], final_step)
            final_value = final_value.squeeze(-1) # [B]
            
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
            
        return states_history, actions, log_probs, rewards, values, returns, advantages

    def update_policy(self, batch_targets, states_hist, actions, old_log_probs, returns, advantages, ppo_epochs=4):
        """
        Runs PPO optimization updates.
        """
        B = batch_targets.shape[0]
        
        for _ in range(ppo_epochs):
            # We iterate sequentially over the time trajectory K
            for t in range(self.K):
                canvas_rgb, canvas_alpha, step_t = states_hist[t]
                
                # Fetch transitions at time t
                old_log_p = old_log_probs[t]  # [B]
                ret = returns[t]              # [B]
                adv = advantages[t]            # [B]
                act = actions[t]              # [B, 13]
                
                # Standardize advantages
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                
                # Forward pass
                alpha, beta, value = self.policy(batch_targets, canvas_rgb, canvas_alpha, step_t)
                value = value.squeeze(-1) # [B]
                
                dist = Beta(alpha, beta)
                new_log_p = dist.log_prob(act).sum(dim=-1) # [B]
                entropy = dist.entropy().sum(dim=-1) # [B]
                
                # PPO Ratio & Clipped loss
                ratio = torch.exp(new_log_p - old_log_p)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Critic value loss
                value_loss = F.mse_loss(value, ret)
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                # Clip gradients to prevent numeric explosion in transformers
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

def run_rl_sprint(epochs=5, batch_size=8, K=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Project Graphos PPO RL Sprint on: {device.type.upper()}")
    
    # 1. Create dataset (128x128 resolution for RL speed)
    dataset = SyntheticStrokeDataset(size=120, K=K, H=128, W=128, device=device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    trainer = PPOTrainingLoop(K=K, H=128, W=128)
    
    # Load pre-trained weights if available to start warm
    if os.path.exists("pretrained_graphos_policy.pth"):
        print("Warm-starting policy using 'pretrained_graphos_policy.pth'...")
        try:
            # Load state dict (ignoring size mismatch if any)
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
            
            # 1. Rollout collection
            states_hist, actions, log_probs, rewards, values, returns, advantages = trainer.collect_rollouts(target_img)
            
            # 2. Optimize policy
            trainer.update_policy(target_img, states_hist, actions, log_probs, returns, advantages)
            
            epoch_reward += rewards.sum(dim=0).mean().item()
            batches_count += 1
            
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Avg Trajectory Reward: {epoch_reward/batches_count:.4f}")
        
        # Save checkpoint weights
        torch.save(trainer.policy.state_dict(), "graphos_policy_rl.pth")
        
    print("\n🎉 PPO Training sprint complete! Policy saved to 'graphos_policy_rl.pth'.")

if __name__ == "__main__":
    run_rl_sprint(epochs=5, batch_size=8, K=5)
