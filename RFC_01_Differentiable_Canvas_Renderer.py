import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableBezierRenderer(nn.Module):
    def __init__(self, canvas_height=128, canvas_width=128, num_samples=16, tau=1.0, epsilon=1e-8):
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
        
        # Precompute pixel coordinates grids
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
        Returns:
            updated_canvas: [B, 4, H, W]
        """
        B = control_points.shape[0]
        device = control_points.device
        
        t = self.t.to(device)
        t2 = t ** 2
        one_minus_t = 1.0 - t
        one_minus_t2 = one_minus_t ** 2
        two_t_one_minus_t = 2.0 * t * one_minus_t
        
        # 1. Interpolate points C_k and widths w_k along the Bezier curve centerline
        C = one_minus_t2 * control_points[:, 0:1, :] + two_t_one_minus_t * control_points[:, 1:2, :] + t2 * control_points[:, 2:3, :]
        w = one_minus_t2 * widths[:, 0:1, None] + two_t_one_minus_t * widths[:, 1:2, None] + t2 * widths[:, 2:3, None]
        
        C_start, C_end = C[:, :-1, :].unsqueeze(1), C[:, 1:, :].unsqueeze(1)
        w_start, w_end = w[:, :-1, :].unsqueeze(1), w[:, 1:, :].unsqueeze(1)
        
        # Grid coordinates: [1, H*W, 1, 2]
        coords = self.pixel_coords.to(device).unsqueeze(0).unsqueeze(2)
        
        v = C_end - C_start # [B, 1, N-1, 2]
        v_norm_sq = torch.sum(v ** 2, dim=-1, keepdim=True) + self.epsilon
        
        # 2. Project grid coordinates to segment lines
        proj = torch.sum((coords - C_start) * v, dim=-1, keepdim=True) / v_norm_sq
        h = torch.clamp(proj, 0.0, 1.0)
        
        p_proj = C_start + h * v
        dist = torch.sqrt(torch.sum((coords - p_proj) ** 2, dim=-1, keepdim=True) + self.epsilon)
        R = (1.0 - h) * (w_start / 2.0) + h * (w_end / 2.0)
        
        # 3. Soft Occupancy calculation (Soft-SDF Sigmoid)
        o = torch.sigmoid((R - dist) / self.tau)
        alpha_s = 1.0 - torch.prod(1.0 - o, dim=2).squeeze(-1)
        alpha_s = alpha_s.view(B, 1, self.H, self.W)
        
        # 4. Draw & Erase updates
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
