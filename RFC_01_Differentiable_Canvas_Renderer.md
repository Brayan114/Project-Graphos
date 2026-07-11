# RFC-01: Differentiable Canvas Renderer & Optimization Core

**Status:** Proposed  
**Authors:** Graphics Architect, Lead Scientist, Systems Engineer  
**Project:** Project Graphos, Orbit Studios R&D Lab  

---

## 1. Objective
To design and implement a custom, differentiable stroke renderer in PyTorch. The renderer must map continuous stroke parameters (coordinates, thickness, color, and mode) to a 2D canvas, enabling direct backpropagation of pixel-level and structural loss gradients back to the stroke parameters.

---

## 2. The Core Mathematical Specifications

### A. The Parametric Brush Stroke
Each brush stroke is modeled as a quadratic Bezier curve with control points $\mathbf{P}_0, \mathbf{P}_1, \mathbf{P}_2 \in [0, 1]^2$:
$$\mathbf{B}(t) = (1-t)^2 \mathbf{P}_0 + 2t(1-t)\mathbf{P}_1 + t^2 \mathbf{P}_2, \quad t \in [0, 1]$$

The width profile $w(t)$ and opacity $\alpha(t)$ scale along the trajectory to support pressure dynamics:
$$w(t) = (1-t)^2 w_0 + 2t(1-t)w_1 + t^2 w_2$$
$$\alpha_s(t) = (1-t)\alpha_{\text{start}} + t\alpha_{\text{end}}$$

### B. Soft Signed Distance Field (Soft-SDF) Rasterization
To allow gradients to flow to coordinates outside the stroke boundary, the hard step-function is replaced by a sigmoidal soft occupancy field. 
To bypass complex cubic roots, the Bezier curve is linearized into $S$ segments. For a pixel $\mathbf{x} = (x, y)$, the minimum distance $d_k(\mathbf{x})$ and local radius $R_k(\mathbf{x})$ are computed for each segment $k \in \{0, \dots, S-1\}$:

$$o_k(\mathbf{x}) = \sigma\left(\frac{R_k(\mathbf{x}) - d_k(\mathbf{x})}{\tau}\right) = \frac{1}{1 + \exp\left(-\frac{R_k(\mathbf{x}) - d_k(\mathbf{x})}{\tau}\right)}$$

where $\tau$ is the temperature parameter. The global stroke occupancy $\alpha_s(\mathbf{x})$ is aggregated using the smooth maximum union:
$$\alpha_s(\mathbf{x}) = 1 - \prod_{k=0}^{S-1} (1 - o_k(\mathbf{x}))$$

### C. Unified Draw/Erase Canvas Blending
We introduce a continuous mode parameter $M \in [-1, 1]$, where:
*   Drawing intensity: $I_{\text{draw}} = \text{ReLU}(M) \in [0, 1]$
*   Erasing intensity: $I_{\text{erase}} = \text{ReLU}(-M) \in [0, 1]$

Let $\mathbf{C}^{(t)}(\mathbf{x})$ and $\mathbf{A}^{(t)}(\mathbf{x})$ represent the RGB and alpha layers of the canvas. The unified, fully differentiable update equations are:
$$\mathbf{A}^{(t+1)}(\mathbf{x}) = \mathbf{A}^{(t)}(\mathbf{x}) \cdot (1 - I_{\text{erase}}\alpha_s(\mathbf{x})) + (1 - \mathbf{A}^{(t)}(\mathbf{x})) \cdot I_{\text{draw}}\alpha_s(\mathbf{x})$$
$$\mathbf{C}^{(t+1)}(\mathbf{x}) = \mathbf{C}^{(t)}(\mathbf{x}) \cdot (1 - I_{\text{draw}}\alpha_s(\mathbf{x}) - I_{\text{erase}}\alpha_s(\mathbf{x})) + I_{\text{draw}}\alpha_s(\mathbf{x})\mathbf{C}_{\text{stroke}} + I_{\text{erase}}\alpha_s(\mathbf{x})\mathbf{C}_{\text{bg}}$$

---

## 3. Systems Optimizations for Tesla T4 GPU

To prevent VRAM Out-of-Memory (OOM) errors and maximize execution speed:
1.  **Segment Linearization ($S=8$):** Rather than evaluating analytical roots on a large mesh, we subdivide the Bezier curve into 8 linear segments and broadcast the calculations, utilizing the GPU's native tensor vectorization.
2.  **Local Bounding Box Cropping:** Evaluate the Soft-SDF calculations only inside a local bounding box of shape $h \times w$ centered on the stroke:
    $$x_{\text{min}} = \min(P_x) - 3w_{\text{max}}, \quad x_{\text{max}} = \max(P_x) + 3w_{\text{max}}$$
    This decreases pixel calculations by up to 95%.
3.  **Automatic Mixed Precision (AMP):** Execute all calculations in float16 to leverage the T4 Tensor Cores.

---

## 4. The Unified Loss Objective
To solve the "zero-spatial-overlap gradient" problem (strokes not receiving gradients because they don't overlap the target features), we propose a compound objective:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{MSE}} \mathcal{L}_{\text{MSE}} + \lambda_{\text{SSIM}} \mathcal{L}_{\text{SSIM}} + \lambda_{\text{LPIPS}} \mathcal{L}_{\text{LPIPS}} + \lambda_{\text{DT}} \mathcal{L}_{\text{DT}}$$

Where:
*   $\mathcal{L}_{\text{DT}}$ is the **Distance Transform edge potential field**, which mathematically pulls stroke coordinates toward the nearest target outlines, even when they do not overlap.
*   $\lambda$ parameters are dynamically scheduled from global visual alignment to pixel-level refinement.

---

## 5. Differentiable Bezier Renderer Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableBezierRenderer(nn.Module):
    def __init__(self, canvas_height=128, canvas_width=128, num_samples=16, tau=1.0, epsilon=1e-8):
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
            control_points: [B, 3, 2] (P0, P1, P2)
            widths: [B, 3] (w0, w1, w2)
            colors: [B, 3] (RGB)
            opacities: [B, 1] (Alpha scale)
            modes: [B, 1] (Mode: >0 is Draw, <0 is Erase)
            canvas: [B, 4, H, W] (Current canvas state: RGB + Alpha)
        """
        B = control_points.shape[0]
        device = control_points.device
        
        t = self.t.to(device)
        t2 = t ** 2
        one_minus_t = 1.0 - t
        one_minus_t2 = one_minus_t ** 2
        two_t_one_minus_t = 2.0 * t * one_minus_t
        
        # Bezier centerline interpolation
        C = one_minus_t2 * control_points[:, 0:1, :] + two_t_one_minus_t * control_points[:, 1:2, :] + t2 * control_points[:, 2:3, :]
        w = one_minus_t2 * widths[:, 0:1, None] + two_t_one_minus_t * widths[:, 1:2, None] + t2 * widths[:, 2:3, None]
        
        C_start, C_end = C[:, :-1, :].unsqueeze(1), C[:, 1:, :].unsqueeze(1)
        w_start, w_end = w[:, :-1, :].unsqueeze(1), w[:, 1:, :].unsqueeze(1)
        
        coords = self.pixel_coords.to(device).unsqueeze(0).unsqueeze(2) # [B, H*W, 1, 2]
        
        v = C_end - C_start
        v_norm_sq = torch.sum(v ** 2, dim=-1, keepdim=True) + self.epsilon
        
        proj = torch.sum((coords - C_start) * v, dim=-1, keepdim=True) / v_norm_sq
        h = torch.clamp(proj, 0.0, 1.0)
        
        p_proj = C_start + h * v
        dist = torch.sqrt(torch.sum((coords - p_proj) ** 2, dim=-1, keepdim=True) + self.epsilon)
        R = (1.0 - h) * (w_start / 2.0) + h * (w_end / 2.0)
        
        # Soft Sigmoidal Occupancy
        o = torch.sigmoid((R - dist) / self.tau)
        alpha_s = 1.0 - torch.prod(1.0 - o, dim=2).squeeze(-1)
        alpha_s = alpha_s.view(B, 1, self.H, self.W)
        
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
```
