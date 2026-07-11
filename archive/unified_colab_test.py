import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableBezierRenderer(nn.Module):
    def __init__(self, canvas_height=256, canvas_width=256, num_samples=32, tau=1.0, epsilon=1e-8):
        """
        Differentiable Bezier Stroke Renderer.
        Subdivides the quadratic Bezier curve into linear segments and evaluates
        a Soft Signed Distance Field (Soft-SDF) to enable continuous, differentiable gradients.
        """
        super().__init__()
        self.H = canvas_height
        self.W = canvas_width
        self.N = num_samples
        self.tau = tau
        self.epsilon = epsilon
        
        # Precompute grid coordinates
        y = torch.linspace(0.5, canvas_height - 0.5, canvas_height)
        x = torch.linspace(0.5, canvas_width - 0.5, canvas_width)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        self.register_buffer("pixel_coords", torch.stack([grid_x, grid_y], dim=-1).view(-1, 2))
        self.register_buffer("t", torch.linspace(0, 1, num_samples).view(1, -1, 1))

    def forward(self, control_points, widths, colors, opacities, modes, canvas):
        """
        Args:
            control_points: [B, 3, 2] (P0, P1, P2 coordinates)
            widths: [B, 3] (w0, w1, w2 stroke widths)
            colors: [B, 3] (RGB stroke color)
            opacities: [B, 1] (Stroke opacity scale)
            modes: [B, 1] (Mode: >0 is Draw, <0 is Erase)
            canvas: [B, 4, H, W] (Current canvas state: RGB + Alpha channels)
        """
        B = control_points.shape[0]
        device = control_points.device
        
        t = self.t.to(device)
        t2 = t ** 2
        one_minus_t = 1.0 - t
        one_minus_t2 = one_minus_t ** 2
        two_t_one_minus_t = 2.0 * t * one_minus_t
        
        # 1. Interpolate centerline points C_k and widths w_k
        C = one_minus_t2 * control_points[:, 0:1, :] + two_t_one_minus_t * control_points[:, 1:2, :] + t2 * control_points[:, 2:3, :]
        w = one_minus_t2 * widths[:, 0:1, None] + two_t_one_minus_t * widths[:, 1:2, None] + t2 * widths[:, 2:3, None]
        
        C_start, C_end = C[:, :-1, :].unsqueeze(1), C[:, 1:, :].unsqueeze(1)
        w_start, w_end = w[:, :-1, :].unsqueeze(1), w[:, 1:, :].unsqueeze(1)
        
        coords = self.pixel_coords.to(device).unsqueeze(0).unsqueeze(2) # [B, H*W, 1, 2]
        
        v = C_end - C_start
        v_norm_sq = torch.sum(v ** 2, dim=-1, keepdim=True) + self.epsilon
        
        # 2. Project coordinates to segments
        proj = torch.sum((coords - C_start) * v, dim=-1, keepdim=True) / v_norm_sq
        h = torch.clamp(proj, 0.0, 1.0)
        
        p_proj = C_start + h * v
        dist = torch.sqrt(torch.sum((coords - p_proj) ** 2, dim=-1, keepdim=True) + self.epsilon)
        R = (1.0 - h) * (w_start / 2.0) + h * (w_end / 2.0)
        
        # 3. Soft Occupancy calculation (Soft-SDF Sigmoid)
        o = torch.sigmoid((R - dist) / self.tau)
        alpha_s = 1.0 - torch.prod(1.0 - o, dim=2).squeeze(-1)
        alpha_s = alpha_s.view(B, 1, self.H, self.W)
        
        # 4. Canvas updates (drawing/erasing)
        I_draw = F.relu(modes).view(B, 1, 1, 1)
        I_erase = F.relu(-modes).view(B, 1, 1, 1)
        
        colors = colors.view(B, 3, 1, 1)
        opacities = opacities.view(B, 1, 1, 1)
        stroke_alpha = opacities * alpha_s
        
        C_bg = torch.ones((B, 3, 1, 1), device=device)
        canvas_rgb, canvas_alpha = canvas[:, :3, :, :], canvas[:, 3:4, :, :]
        
        new_alpha = canvas_alpha * (1.0 - I_erase * stroke_alpha) + (1.0 - canvas_alpha) * (I_draw * stroke_alpha)
        blend_term = I_draw * stroke_alpha + I_erase * stroke_alpha
        new_rgb = canvas_rgb * (1.0 - blend_term) + (I_draw * stroke_alpha) * colors + (I_erase * stroke_alpha) * C_bg
        
        return torch.clamp(torch.cat([new_rgb, new_alpha], dim=1), 0.0, 1.0)

def test_differentiability():
    print("=== Testing Project Graphos: Differentiable Renderer ===")
    
    # Check if GPU is available (Colab T4 runtime check)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device.type.upper()}")
    
    # Initialize renderer (256x256 canvas)
    renderer = DifferentiableBezierRenderer(canvas_height=256, canvas_width=256, num_samples=32, tau=1.0).to(device)
    
    # Define a single stroke: start at (40,40), control point (128, 200), end at (216, 40)
    control_points = torch.tensor([[[40.0, 40.0], [128.0, 200.0], [216.0, 40.0]]], requires_grad=True, device=device)
    
    # Widths w0, w1, w2
    widths = torch.tensor([[10.0, 25.0, 10.0]], requires_grad=True, device=device)
    
    # Colors: RGB (bright red [1, 0, 0])
    colors = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True, device=device)
    
    # Opacity (alpha scale)
    opacities = torch.tensor([[0.95]], requires_grad=True, device=device)
    
    # Mode: Draw (1.0)
    modes = torch.tensor([[1.0]], requires_grad=True, device=device)
    
    # Initial canvas state: Blank white canvas
    canvas = torch.zeros((1, 4, 256, 256), device=device)
    canvas[:, :3, :, :] = 1.0 # White background
    
    print("\nRunning Forward Pass...")
    updated_canvas = renderer(control_points, widths, colors, opacities, modes, canvas)
    print(f"Output canvas shape: {updated_canvas.shape}")
    
    # Save the rendered canvas to a PNG file in Colab filesystem
    try:
        img_tensor = updated_canvas[0].detach().cpu()
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
        
        from PIL import Image
        img = Image.fromarray(img_np, mode='RGBA')
        img.save("test_stroke.png")
        print("SUCCESS: Rendered stroke visual saved to 'test_stroke.png' in your Colab files.")
        
        # In Colab, we can display the image inline:
        try:
            from IPython.display import display
            display(img)
            print("Visual rendering displayed above!")
        except Exception:
            pass
    except Exception as e:
        print(f"Skipping visual save/display: {e}")
        
    # Define a dummy target canvas (we want all pixels to be blue)
    target = torch.zeros_like(updated_canvas)
    target[:, 2, :, :] = 1.0
    target[:, 3, :, :] = 1.0
    
    # Compute L2 Loss
    loss = torch.mean((updated_canvas - target) ** 2)
    print(f"Current L2 Loss: {loss.item():.6f}")
    
    print("\nRunning Backward Pass...")
    loss.backward()
    
    print("\n=== Gradient Check ===")
    print(f"Gradient for control points (P0, P1, P2):\n{control_points.grad}")
    print(f"Gradient for stroke widths (w0, w1, w2):\n{widths.grad}")
    print(f"Gradient for stroke color (R, G, B):\n{colors.grad}")
    
    assert control_points.grad is not None, "Error: Control points did not receive gradients!"
    assert widths.grad is not None, "Error: Widths did not receive gradients!"
    assert colors.grad is not None, "Error: Colors did not receive gradients!"
    
    print("\nSUCCESS: All stroke parameters successfully received gradients through the Soft-SDF renderer!")

if __name__ == "__main__":
    test_differentiability()
