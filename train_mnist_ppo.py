import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import math
import os
import time

# Import renderer and policy network
from RFC_01_Differentiable_Canvas_Renderer import DifferentiableBezierRenderer
from policy_network import DifferentiableGraphosPolicy

class PreprocessedMNISTVRAMDataset(Dataset):
    def __init__(self, size=2000, H=128, W=128, device="cuda"):
        """
        Downloads, pre-processes, and caches the MNIST dataset directly in GPU VRAM.
        This resolves all disk I/O, CPU worker thread, and CPU-to-GPU transfer bottlenecks.
        """
        self.device = device
        self.H = H
        self.W = W
        
        # 1. Download standard MNIST
        mnist_raw = datasets.MNIST(
            root="./mnist_data", 
            train=True, 
            download=True, 
            transform=transforms.ToTensor()
        )
        
        # Subsample to the requested training size
        subsample_indices = torch.randperm(len(mnist_raw))[:size]
        
        print(f"Caching {size} MNIST target images directly in {device.upper()} VRAM...")
        cached_images = []
        
        # Process and load onto GPU in chunks
        for idx in subsample_indices:
            img_tensor, label = mnist_raw[idx] # [1, 28, 28] in [0, 1]
            img_tensor = img_tensor.to(device)
            
            # Invert grayscale values: 1.0 - img (makes background white and digit black)
            inverted_img = 1.0 - img_tensor
            
            # Resize target to canvas dimension (128x128) using GPU-accelerated bicubic interpolation
            with torch.no_grad():
                # Add batch dim, interpolate, squeeze, clamp
                scaled_img = F.interpolate(
                    inverted_img.unsqueeze(0), 
                    size=(H, W), 
                    mode="bicubic", 
                    align_corners=True
                ).squeeze(0)
                scaled_img = torch.clamp(scaled_img, 0.0, 1.0)
                
            cached_images.append(scaled_img.detach())
            
        # Stack into a single tensor of shape [size, 1, H, W]
        self.data_tensor = torch.stack(cached_images).to(device)
        print("VRAM Caching Completed.")

    def __len__(self):
        return self.data_tensor.shape[0]

    def __getitem__(self, idx):
        # Fetch target grayscale image [1, H, W] directly from VRAM
        target_gray = self.data_tensor[idx]
        
        # Replicate grayscale to RGB RGB on-the-fly via view expansion
        target_rgb = target_gray.repeat(3, 1, 1) # [3, H, W]
        
        # Canvas Alpha: derived from the digit mask (original non-inverted MNIST digit)
        target_alpha = 1.0 - target_gray # [1, H, W]
        
        return target_rgb, target_alpha

class MNISTPPOTrainingLoop:
    def __init__(self, K=4, H=128, W=128, lr=1e-4, clip_eps=0.2, gamma=0.99, gae_lambda=0.95, value_coef=0.5, entropy_coef=0.01):
        """
        PPO Trainer specialized for MNIST handwriting reconstruction.
        Uses a 14-dimensional action space to support Draw and Erase modes.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.K = K # Drawing steps (horizon T)
        self.H = H
        self.W = W
        
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Initialize Renderer & Policy (using action_dim=14 for Draw/Erase support)
        self.renderer = DifferentiableBezierRenderer(canvas_height=H, canvas_width=W, tau=0.5).to(self.device)
        self.policy = DifferentiableGraphosPolicy(img_size=H, num_layers=4, num_heads=4, embed_dim=128, action_dim=14).to(self.device)
        
        # Compile model to optimize self-attention and cross-attention blocks (unlocks FlashAttention SDPA)
        if torch.cuda.is_available():
            try:
                self.policy = torch.compile(self.policy, mode="reduce-overhead")
                print("PyTorch MNIST Policy compilation initialized successfully.")
            except Exception as e:
                print(f"Skipping torch.compile: {e}")
                
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.scaler = torch.amp.GradScaler('cuda') # GradScaler for stable FP16 AMP training

    def collect_rollouts(self, batch_targets):
        """
        Collects trajectories for a batch of MNIST targets.
        Supports 14-dimensional action parameters to control Draw/Erase mode.
        """
        B = batch_targets.shape[0]
        
        # Initial blank white canvas [B, 4, H, W]
        canvas = torch.zeros((B, 4, self.H, self.W), device=self.device)
        canvas[:, :3, :, :] = 1.0 # White background
        canvas[:, 3:4, :, :] = 1.0 # Opaque
        
        obs_canvas_rgb = []
        obs_canvas_alpha = []
        obs_step_t = []
        actions_history = []
        log_probs_history = []
        rewards_history = []
        values_history = []
        
        # Pre-compute L1 target error
        current_l1 = F.l1_loss(canvas[:, :3, :, :], batch_targets, reduction='none').mean(dim=[1,2,3])
        
        # Scale coordinates for aspect-ratio support
        scale_cp = torch.tensor([self.W, self.H], device=self.device).view(1, 1, 2)
        
        with torch.no_grad():
            for t in range(self.K):
                step_t = torch.full((B, 1), t / self.K, device=self.device)
                canvas_rgb = canvas[:, :3, :, :]
                canvas_alpha = canvas[:, 3:4, :, :]
                
                # Store detached observations
                obs_canvas_rgb.append(canvas_rgb.clone().detach())
                obs_canvas_alpha.append(canvas_alpha.clone().detach())
                obs_step_t.append(step_t.clone().detach())
                
                # Forward pass under FP16 autocast
                with torch.amp.autocast('cuda'):
                    alpha, beta, value = self.policy(batch_targets, canvas_rgb, canvas_alpha, step_t)
                
                # Detach and clone values
                values_history.append(value.squeeze(-1).clone().detach())
                
                # Sample actions from the 14D Beta distribution
                dist = Beta(alpha, beta)
                action = dist.sample() # [B, 14]
                log_prob = dist.log_prob(action).sum(dim=-1) # [B]
                
                actions_history.append(action.clone().detach())
                log_probs_history.append(log_prob.clone().detach())
                
                # Draw / Erase decision (14th parameter mapped to mode: Draw if > 0.5 else Erase)
                # Map [0, 1] parameter to [-1, 1] mode range
                modes = (action[:, 13:14] > 0.5).float() * 2.0 - 1.0 # [B, 1]
                
                # Render strokes
                # Scale coordinates and apply a minimum stroke width of 2.0 pixels to prevent dissipation
                cp = action[:, 0:6].view(B, 3, 2) * scale_cp
                w = 2.0 + action[:, 6:9] * (self.H * 0.12 - 2.0)
                c = action[:, 9:12]
                opacities = action[:, 12:13]
                
                new_canvas = self.renderer(cp, w, c, opacities, modes, canvas)
                
                # Compute step reward (L1 delta)
                new_l1 = F.l1_loss(new_canvas[:, :3, :, :], batch_targets, reduction='none').mean(dim=[1,2,3])
                
                # Paint waste regularization (decays over step t, scaled down to prevent early collapse)
                reg_weight = 0.01 * (1.0 - (t / self.K))
                reg = action[:, 6:9].mean(dim=-1) * action[:, 12]
                
                reward = (current_l1 - new_l1) - reg_weight * reg
                rewards_history.append(reward.clone().detach())
                
                # Update canvas and L1 trackers
                canvas = new_canvas.clone().detach()
                current_l1 = new_l1.clone().detach()
                
            # Value of the final state
            final_step = torch.full((B, 1), 1.0, device=self.device)
            with torch.amp.autocast('cuda'):
                _, _, final_value = self.policy(batch_targets, canvas[:, :3, :, :], canvas[:, 3:4, :, :], final_step)
            final_value = final_value.squeeze(-1).detach()
            
        # Stack rollout variables
        obs_canvas_rgb = torch.stack(obs_canvas_rgb)     # [K, B, 3, H, W]
        obs_canvas_alpha = torch.stack(obs_canvas_alpha) # [K, B, 1, H, W]
        obs_step_t = torch.stack(obs_step_t)             # [K, B, 1]
        
        actions = torch.stack(actions_history)      # [K, B, 14]
        log_probs = torch.stack(log_probs_history)  # [K, B]
        rewards = torch.stack(rewards_history)      # [K, B]
        values = torch.stack(values_history)        # [K, B]
        
        # 5. Compute GAE Returns and Advantages
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
        Updates the policy using flattened trajectories to break temporal inner-loop
        parameter dependencies, parallelizing updates across mini-batches under FP16 AMP.
        """
        obs_rgb, obs_alpha, obs_step, actions, old_log_probs, returns, advantages = rollouts
        K, B = actions.shape[0], actions.shape[1]
        
        # Flatten all trajectory states to run parallel updates
        flat_obs_rgb = obs_rgb.view(K*B, 3, self.H, self.W)
        flat_obs_alpha = obs_alpha.view(K*B, 1, self.H, self.W)
        flat_obs_step = obs_step.view(K*B, 1)
        flat_actions = actions.view(K*B, 14) # 14D Actions
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

def run_mnist_rl_sprint(epochs=5, batch_size=32, K=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Project Graphos MNIST PPO RL Sprint on: {device.type.upper()}")
    
    # 1. Load VRAM cached dataset (Size 2000 images, res 128x128)
    dataset = PreprocessedMNISTVRAMDataset(size=2000, H=128, W=128, device=device.type)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    trainer = MNISTPPOTrainingLoop(K=K, H=128, W=128)
    
    print("\nStarting MNIST PPO optimization loops...")
    
    for epoch in range(epochs):
        epoch_reward = 0.0
        batches_count = 0
        
        start_time = time.time()
        for target_img, _ in loader:
            target_img = target_img.to(device)
            
            # Rollout collection (Safe from leaks)
            rollouts = trainer.collect_rollouts(target_img)
            rewards = rollouts[3] # [K, B]
            
            # Parallel update
            trainer.update_policy(target_img, rollouts)
            
            epoch_reward += rewards.sum(dim=0).mean().item()
            batches_count += 1
            
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Avg Trajectory Reward: {epoch_reward/batches_count:.5f} | Time: {elapsed:.2f}s")
        
        # Save checkpoint weights (unwrapping torch.compile wrapper if it exists)
        raw_policy = trainer.policy._orig_mod if hasattr(trainer.policy, "_orig_mod") else trainer.policy
        torch.save(raw_policy.state_dict(), "graphos_mnist_policy.pth")
        
    print("\n🎉 MNIST PPO training complete! Policy saved to 'graphos_mnist_policy.pth'.")

if __name__ == "__main__":
    run_mnist_rl_sprint(epochs=5, batch_size=32, K=4)
