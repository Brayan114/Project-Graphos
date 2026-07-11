import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from PIL import Image
import os

# Import our differentiable renderer
from RFC_01_Differentiable_Canvas_Renderer import DifferentiableBezierRenderer

def generate_mock_target(H=128, W=128):
    """
    Generates a simple target image (a black circle on a white background)
    for testing reconstruction when no target image is uploaded.
    """
    canvas = Image.new("RGB", (W, H), "white")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    # Draw a thick black ring
    draw.ellipse([20, 20, W-20, H-20], outline="black", width=12)
    # Convert to tensor [3, H, W] normalized to [0, 1]
    tensor = torch.tensor(list(canvas.getdata())).view(H, W, 3).permute(2, 0, 1).float() / 255.0
    return tensor

def optimize_target_image(target_tensor, num_steps_per_window=100, total_strokes=20, window_size=4, H=128, W=128):
    """
    Fits a set of Bezier strokes to match target_tensor using a sliding window optimization strategy.
    
    Args:
        target_tensor: [3, H, W] Target image tensor normalized to [0, 1]
        num_steps_per_window: Number of gradient descent steps per window step
        total_strokes: Total strokes (K) to draw
        window_size: Number of active strokes to optimize simultaneously (W_active)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Starting Project Graphos Direct Optimization on: {device.type.upper()}")
    
    target_img = target_tensor.to(device)
    
    # 1. Initialize stroke parameter tensors in normalized [0, 1] coordinate space
    # For a stroke s, params are: [P0_x, P0_y, P1_x, P1_y, P2_x, P2_y, w0, w1, w2, r, g, b, alpha] (13 parameters)
    # Coordinates are initialized randomly around the center
    all_params = torch.rand((total_strokes, 13), device=device)
    # Set widths: initialize to roughly 0.05 to 0.1 of canvas (representing 5-12 pixels)
    all_params[:, 6:9] = 0.05 + 0.05 * torch.rand((total_strokes, 3), device=device)
    # Set colors to random or dark grayscale by default
    all_params[:, 9:12] = 0.1 + 0.2 * torch.rand((total_strokes, 3), device=device)
    # Set opacity (alpha): initialize to 0.8
    all_params[:, 12] = 0.8
    
    # Static background canvas (white, fully opaque) [1, 4, H, W]
    bg_canvas = torch.zeros((1, 4, H, W), device=device)
    bg_canvas[:, :3, :, :] = 1.0 # White background
    bg_canvas[:, 3:4, :, :] = 1.0 # Fully opaque
    
    # We will accumulate frozen (completed) strokes into this canvas
    frozen_canvas = bg_canvas.clone()
    
    # Temperature schedule parameters
    tau_start = 8.0
    tau_end = 0.5
    
    # Output directory for saving drawing steps
    os.makedirs("drawing_steps", exist_ok=True)
    
    # Optimization runs window-by-window
    # Every step, we add 1 new stroke, keeping a sliding window of active strokes
    for i in range(total_strokes):
        active_start = max(0, i - window_size + 1)
        active_end = i + 1
        num_active = active_end - active_start
        
        print(f"\n--- Fitting Window {i+1}/{total_strokes} (Optimizing strokes {active_start+1} to {active_end}) ---")
        
        # Determine the background for this window step:
        # It's the canvas with all strokes *before* the active window rendered statically
        with torch.no_grad():
            win_bg = bg_canvas.clone()
            # Render frozen strokes (0 to active_start)
            # Create a local temp renderer with sharp resolution
            temp_renderer = DifferentiableBezierRenderer(canvas_height=H, canvas_width=W, tau=0.3).to(device)
            for s_idx in range(active_start):
                p = all_params[s_idx:s_idx+1]
                # Scale normalized coords [0, 1] to pixel space [0, H]
                cp = p[:, 0:6].view(1, 3, 2) * H
                w = p[:, 6:9] * H
                c = p[:, 9:12]
                alpha = p[:, 12:13]
                modes = torch.ones((1, 1), device=device) # Always Draw mode
                win_bg = temp_renderer(cp, w, c, alpha, modes, win_bg)
        
        # Gather active parameters as leaves for PyTorch Autograd
        active_params = all_params[active_start:active_end].clone().detach().requires_grad_(True)
        
        # Set up optimizer (optimize only the active window)
        optimizer = optim.Adam([active_params], lr=0.03)
        
        # Run local gradient descent iterations
        for step in range(num_steps_per_window):
            optimizer.zero_grad()
            
            # Cosine Temperature Annealing
            tau = tau_end + 0.5 * (tau_start - tau_end) * (1.0 + math.cos(math.pi * step / num_steps_per_window))
            renderer = DifferentiableBezierRenderer(canvas_height=H, canvas_width=W, tau=tau).to(device)
            
            # Autoregressive render of the active strokes on top of the window background
            canvas_state = win_bg.clone()
            for k in range(num_active):
                p = active_params[k:k+1]
                # Scale normalized coords to pixel space [0, H]
                cp = p[:, 0:6].view(1, 3, 2) * H
                w = p[:, 6:9] * H
                c = p[:, 9:12]
                alpha = p[:, 12:13]
                modes = torch.ones((1, 1), device=device)
                
                canvas_state = renderer(cp, w, c, alpha, modes, canvas_state)
            
            # Extract RGB layer for loss
            rendered_rgb = canvas_state[:, :3, :, :]
            
            # L1 + L2 Reconstruction Loss
            loss_recon = F.l1_loss(rendered_rgb, target_img.unsqueeze(0)) + 0.5 * F.mse_loss(rendered_rgb, target_img.unsqueeze(0))
            
            # Soft Boundary Penalty: penalize coordinates that drift outside the [0, 1] range
            coords_only = active_params[:, 0:6]
            boundary_loss = torch.sum(F.relu(-coords_only) ** 2) + torch.sum(F.relu(coords_only - 1.0) ** 2)
            
            # Color boundary clipping penalty
            color_loss = torch.sum(F.relu(-active_params[:, 9:12]) ** 2) + torch.sum(F.relu(active_params[:, 9:12] - 1.0) ** 2)
            
            total_loss = loss_recon + 1.0 * boundary_loss + 0.5 * color_loss
            
            total_loss.backward()
            optimizer.step()
            
            # Project constraints (widths and opacities must remain valid)
            with torch.no_grad():
                active_params[:, 6:9].clamp_(min=0.005, max=0.2) # Stroke width between 0.5% and 20% of canvas
                active_params[:, 12].clamp_(min=0.05, max=1.0)   # Opacity scale
                active_params[:, 9:12].clamp_(min=0.0, max=1.0)  # Colors in [0, 1]
                
            if (step + 1) % 25 == 0:
                print(f"  Step {step+1:03d}/{num_steps_per_window} | Loss: {loss_recon.item():.6f} (Boundary: {boundary_loss.item():.4f})")
        
        # Save optimized parameters back into global storage
        with torch.no_grad():
            all_params[active_start:active_end] = active_params.clone().detach()
            
        # Export progress image for this window
        with torch.no_grad():
            temp_renderer = DifferentiableBezierRenderer(canvas_height=H, canvas_width=W, tau=0.3).to(device)
            preview_canvas = bg_canvas.clone()
            for s_idx in range(active_end):
                p = all_params[s_idx:s_idx+1]
                cp = p[:, 0:6].view(1, 3, 2) * H
                w = p[:, 6:9] * H
                c = p[:, 9:12]
                alpha = p[:, 12:13]
                modes = torch.ones((1, 1), device=device)
                preview_canvas = temp_renderer(cp, w, c, alpha, modes, preview_canvas)
                
            # Save PNG preview
            img_tensor = preview_canvas[0, :3, :, :].detach().cpu()
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
            img = Image.fromarray(img_np, mode='RGB')
            img_path = f"drawing_steps/stroke_{i+1:02d}.png"
            img.save(img_path)
            print(f"✔️ Progress saved: {img_path}")
            
    print("\n🎉 Optimization complete! All strokes have been aligned.")
    return all_params

if __name__ == "__main__":
    # Create target (circle)
    target = generate_mock_target(128, 128)
    
    # Save the target image as a reference
    target_np = (target.permute(1, 2, 0).numpy() * 255).astype('uint8')
    Image.fromarray(target_np).save("target_reference.png")
    print("Saved target_reference.png for comparison.")
    
    # Run optimization (20 strokes, fitting the circle)
    optimized_params = optimize_target_image(target, num_steps_per_window=120, total_strokes=15, window_size=5)
