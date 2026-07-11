import torch
import torch.nn as nn
from RFC_01_Differentiable_Canvas_Renderer import DifferentiableBezierRenderer

def test_differentiability():
    print("=== Testing Project Graphos: Differentiable Renderer ===")
    
    # Initialize renderer (256x256 canvas for visual fidelity)
    renderer = DifferentiableBezierRenderer(canvas_height=256, canvas_width=256, num_samples=32, tau=1.0)
    
    # Define a single stroke: start at (40,40), control point (128, 200), end at (216, 40)
    # Control points shape: [B, 3, 2]
    control_points = torch.tensor([[[40.0, 40.0], [128.0, 200.0], [216.0, 40.0]]], requires_grad=True)
    
    # Widths w0, w1, w2: [B, 3] (starting thin, swelling at center, ending thin)
    widths = torch.tensor([[10.0, 25.0, 10.0]], requires_grad=True)
    
    # Colors: RGB (bright red [1, 0, 0])
    colors = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    
    # Opacity (alpha scale): [B, 1]
    opacities = torch.tensor([[0.95]], requires_grad=True)
    
    # Mode: Draw (1.0)
    modes = torch.tensor([[1.0]], requires_grad=True)
    
    # Initial canvas state: Blank canvas [B, 4, H, W] (fully white with 0 alpha)
    canvas = torch.zeros((1, 4, 256, 256))
    canvas[:, :3, :, :] = 1.0 # White background
    
    print("\nRunning Forward Pass...")
    updated_canvas = renderer(control_points, widths, colors, opacities, modes, canvas)
    print(f"Output canvas shape: {updated_canvas.shape}")
    
    # Save the rendered canvas to a PNG file
    try:
        # Convert tensor to image format: [H, W, C]
        # Drop alpha channel for basic RGB preview or keep it
        img_tensor = updated_canvas[0].detach().cpu()
        
        # Multiply by 255 and cast to uint8
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
        
        # Use PIL to save
        from PIL import Image
        img = Image.fromarray(img_np, mode='RGBA')
        img.save("test_stroke.png")
        print("SUCCESS: Rendered stroke visual saved to 'test_stroke.png'")
    except Exception as e:
        print(f"Skipping visual save: {e}")
        
    # Let's define a dummy target canvas (e.g. we want all pixels to be blue [0, 0, 1, 1])
    target = torch.zeros_like(updated_canvas)
    target[:, 2, :, :] = 1.0 # Blue channel = 1
    target[:, 3, :, :] = 1.0 # Alpha channel = 1
    
    # Compute L2 Loss
    loss = torch.mean((updated_canvas - target) ** 2)
    print(f"Current L2 Loss: {loss.item():.6f}")
    
    print("\nRunning Backward Pass...")
    loss.backward()
    
    # Verify that gradients were successfully computed for coordinates
    grad_cp = control_points.grad
    grad_w = widths.grad
    grad_c = colors.grad
    
    print("\n=== Gradient Check ===")
    print(f"Gradient for control points (P0, P1, P2):\n{grad_cp}")
    print(f"Gradient for stroke widths (w0, w1, w2):\n{grad_w}")
    print(f"Gradient for stroke color (R, G, B):\n{grad_c}")
    
    assert grad_cp is not None, "Error: Control points did not receive gradients!"
    assert grad_w is not None, "Error: Widths did not receive gradients!"
    assert grad_c is not None, "Error: Colors did not receive gradients!"
    
    print("\nSUCCESS: All stroke parameters successfully received gradients through the Soft-SDF renderer!")

if __name__ == "__main__":
    test_differentiability()
