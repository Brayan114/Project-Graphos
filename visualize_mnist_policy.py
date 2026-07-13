import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Import renderer and policy network
from RFC_01_Differentiable_Canvas_Renderer import DifferentiableBezierRenderer
from policy_network import DifferentiableGraphosPolicy

def generate_bezier_points(p0, p1, p2, num_points=100):
    """
    Evaluates a quadratic Bezier curve at num_points coordinates.
    """
    u = np.linspace(0, 1, num_points)[:, None]
    curve = (1 - u)**2 * p0 + 2 * u * (1 - u) * p1 + u**2 * p2
    return curve[:, 0], curve[:, 1]

@torch.no_grad()
def run_and_save_reconstruction(weights_path="graphos_mnist_policy.pth", save_path="mnist_reconstruction_grid.png", K=4, H=128, W=128):
    """
    Loads the trained PPO policy, pulls a target digit from the MNIST dataset,
    runs greedy inference to draw it step-by-step, and plots the progression and path overlays.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running MNIST Policy Visualization on: {device.type.upper()}")
    
    # 1. Download/Load a sample MNIST digit
    mnist_raw = datasets.MNIST(root="./mnist_data", train=False, download=True, transform=transforms.ToTensor())
    # Pull a clean sample (e.g. index 10 is a '0', index 2 is a '1', index 0 is a '7')
    # Let's pull a random digit index to make the plots interesting
    target_idx = np.random.randint(len(mnist_raw))
    img_tensor, label = mnist_raw[target_idx]
    
    # Pre-process image to match canvas style (invert: white background, black digit)
    inverted_img = 1.0 - img_tensor.to(device)
    target_img = F.interpolate(inverted_img.unsqueeze(0), size=(H, W), mode="bicubic", align_corners=True).squeeze(0)
    target_img = torch.clamp(target_img, 0.0, 1.0)
    target_rgb = target_img.repeat(3, 1, 1).unsqueeze(0) # [1, 3, H, W]
    
    # 2. Instantiate and Load Policy & Renderer (with action_dim=14)
    renderer = DifferentiableBezierRenderer(canvas_height=H, canvas_width=W, tau=0.5).to(device)
    policy = DifferentiableGraphosPolicy(img_size=H, num_layers=4, num_heads=4, embed_dim=128, action_dim=14).to(device)
    
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        # Strip '_orig_mod.' prefix added by torch.compile wrappers during saving
        clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        policy.load_state_dict(clean_state_dict)
        print(f"Successfully loaded policy weights from '{weights_path}'")
    else:
        print(f"Warning: '{weights_path}' not found! Running visualization with random weights.")
        
    policy.eval()
    
    # 3. Autoregressive drawing rollout (greedy inference: take mean of Beta distribution)
    canvas = torch.zeros((1, 4, H, W), device=device)
    canvas[:, :3, :, :] = 1.0 # White background
    canvas[:, 3:4, :, :] = 1.0 # Opaque
    
    canvas_history = [canvas[:, :3, :, :].cpu().squeeze(0).permute(1, 2, 0).numpy()]
    stroke_params = []
    
    scale_cp = torch.tensor([W, H], device=device).view(1, 1, 2)
    
    for t in range(K):
        step_t = torch.full((1, 1), t / K, device=device)
        canvas_rgb = canvas[:, :3, :, :]
        canvas_alpha = canvas[:, 3:4, :, :]
        
        # Forward pass (autocast for consistency with T4 training)
        with torch.amp.autocast('cuda') if torch.cuda.is_available() else torch.amp.autocast('cpu', enabled=False):
            alpha, beta, _ = policy(target_rgb, canvas_rgb, canvas_alpha, step_t)
            
        # Greedy action: expected value of Beta distribution (mean)
        action = alpha / (alpha + beta) # [1, 14]
        # Clamp action to prevent boundary NaNs
        action_clamped = torch.clamp(action, min=1e-5, max=1.0 - 1e-5)
        
        # Extract stroke geometry & mode (enforce minimum width bias to match training setup)
        cp = action_clamped[:, 0:6].view(1, 3, 2) * scale_cp
        w = 2.0 + action_clamped[:, 6:9] * (H * 0.12 - 2.0)
        c = action_clamped[:, 9:12]
        opacities = action_clamped[:, 12:13]
        modes = (action_clamped[:, 13:14] > 0.5).float() * 2.0 - 1.0 # Draw (+1) or Erase (-1)
        
        # Render
        canvas = renderer(cp, w, c, opacities, modes, canvas)
        
        # Record
        canvas_history.append(canvas[:, :3, :, :].cpu().squeeze(0).permute(1, 2, 0).numpy())
        stroke_params.append({
            'cp': cp.squeeze(0).cpu().numpy(),
            'w': w.squeeze(0).cpu().numpy(),
            'mode': modes.item()
        })
        
    # 4. Generate Plot Layout: 2 Rows, K + 2 Columns
    fig, axes = plt.subplots(2, K + 2, figsize=(2.5 * (K + 2), 5), dpi=120)
    
    # Target Image Plot
    target_np = target_rgb.squeeze(0).permute(1, 2, 0).cpu().numpy()
    target_np = np.clip(target_np, 0.0, 1.0)
    axes[0, 0].imshow(target_np)
    axes[0, 0].set_title("Target Image", fontsize=10, fontweight='bold', color='blue')
    axes[0, 0].axis('off')
    
    # Start Canvas Plot
    axes[0, 1].imshow(canvas_history[0])
    axes[0, 1].set_title("Start (Canvas)", fontsize=9)
    axes[0, 1].axis('off')
    
    # Sequentially plot drawing progression
    for t in range(K):
        col = t + 2
        step_canvas = np.clip(canvas_history[t + 1], 0.0, 1.0)
        axes[0, col].imshow(step_canvas)
        axes[0, col].set_title(f"Stroke {t+1}", fontsize=9)
        axes[0, col].axis('off')
        
    # 5. Row 2: Vector Path Overlays plotted on top of the Target Image
    for col in range(K + 2):
        axes[1, col].imshow(target_np)
        axes[1, col].axis('off')
    axes[1, 0].set_title("Path Overlays", fontsize=10, fontweight='bold')
    
    for t in range(K):
        col = t + 2
        ax = axes[1, col]
        
        # Plot all paths up to step t
        for prev_t in range(t + 1):
            params = stroke_params[prev_t]
            pts = params['cp'] # [3, 2] -> P0, P1, P2
            mode = params['mode']
            
            p0, p1, p2 = pts[0], pts[1], pts[2]
            
            # Generate curve coordinates
            cx, cy = generate_bezier_points(p0, p1, p2)
            
            # Color coding: Green/Blue for drawing, Red for erasing
            path_color = '#1f77b4' if mode > 0.0 else '#d62728'
            
            # Plot Bezier path
            ax.plot(cx, cy, color=path_color, linewidth=2.0, alpha=0.8)
            
            # Draw dashed control skeleton
            ax.plot([p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]], color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
            
            # Plot control points (P0=filled dot, P1=open square)
            ax.scatter(p0[0], p0[1], color='black', marker='o', s=12, zorder=5)
            ax.scatter(p1[0], p1[1], color='gray', marker='s', s=8, zorder=5)
            
            # Add Tangent Arrow at Endpoint P2
            arrow_dx = p2[0] - p1[0]
            arrow_dy = p2[1] - p1[1]
            arrow_len = np.sqrt(arrow_dx**2 + arrow_dy**2) + 1e-8
            dx_norm = (arrow_dx / arrow_len) * 4.0
            dy_norm = (arrow_dy / arrow_len) * 4.0
            
            ax.add_patch(patches.FancyArrow(
                p2[0] - dx_norm, p2[1] - dy_norm, dx_norm, dy_norm,
                width=0.6, color='black', head_width=2.5, zorder=6
            ))
            
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"🎉 Matplotlib grid saved successfully to: {save_path}")

if __name__ == "__main__":
    # Check if a model checkpoint exists
    ckpt = "graphos_mnist_policy.pth" if os.path.exists("graphos_mnist_policy.pth") else "pretrained_graphos_policy.pth"
    run_and_save_reconstruction(weights_path=ckpt, save_path="mnist_reconstruction_grid.png", K=4)
