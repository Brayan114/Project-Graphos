# Project Graphos: Differentiable Stroke-Based Visual Reconstruction
**Milestone 1 & 2 Research Report**

**Authors:** Orbit Studios R&D Lab  
**Date:** July 11, 2026  

---

## Abstract
This report details the architectural design and empirical validation of **Project Graphos**, a stroke-based visual agent that learns to draw images step-by-step. We introduce a fully differentiable quadratic Bezier stroke renderer utilizing a soft signed distance field (Soft-SDF) sigmoidal occupancy formulation. To resolve the vanishing gradient problem caused by stroke occlusion during direct backpropagation, we implement a Sliding-Window Optimization loop. We demonstrate that the system successfully reconstructs target shapes (a black ring outline) using 15 discrete strokes on a single Tesla T4 GPU in under two minutes, with stable coordinate gradient convergence.

---

## 1. Introduction & Differentiable Renderer Design
Standard vector graphics rasterization involves discrete step functions (pixel inside vs. outside a boundary), which have zero gradients almost everywhere. This makes them incompatible with gradient-based neural network optimization.

Graphos resolves this by implementing a **Soft Signed Distance Field (Soft-SDF)** rasterizer in PyTorch.

### 1.1 Mathematical Formulation
Let a stroke be a quadratic Bezier curve centerline $\mathbf{B}(t)$ with control points $\mathbf{P}_0, \mathbf{P}_1, \mathbf{P}_2$ and width profile $w(t)$ for $t \in [0, 1]$:
$$\mathbf{B}(t) = (1-t)^2 \mathbf{P}_0 + 2t(1-t)\mathbf{P}_1 + t^2 \mathbf{P}_2$$
$$w(t) = (1-t)^2 w_0 + 2t(1-t)w_1 + t^2 w_2$$

To bypass expensive cubic root solvers, the curve is linearized into $S=8$ segments. For a pixel $\mathbf{x} = (x, y)$, the minimum distance $d_k(\mathbf{x})$ and local radius $R_k(\mathbf{x})$ are evaluated for each segment $k$:
$$o_k(\mathbf{x}) = \sigma\left(\frac{R_k(\mathbf{x}) - d_k(\mathbf{x})}{\tau}\right) = \frac{1}{1 + \exp\left(-\frac{R_k(\mathbf{x}) - d_k(\mathbf{x})}{\tau}\right)}$$

where $\tau$ is a temperature parameter controlling the boundary smoothness. The global stroke occupancy $\alpha_s(\mathbf{x})$ is aggregated:
$$\alpha_s(\mathbf{x}) = 1 - \prod_{k=0}^{S-1} (1 - o_k(\mathbf{x}))$$

---

## 2. Sliding-Window Optimization Loop
When optimizing multiple strokes simultaneously, newer strokes that overlap older strokes block the gradient flow, causing the parameters of early layers to freeze (the **occlusion bottleneck**). 

We resolve this by implementing a **Sliding Window Optimization** loop:
*   At any step $i$, we only optimize the parameter set of the last $W_{\text{active}} = 5$ active strokes.
*   All strokes prior to the window ($< i - W_{\text{active}}$) are frozen and pre-rendered into a static background canvas.
*   This bounds the autograd computation graph depth, ensuring stable gradients and reducing VRAM usage to $O(1)$ scaling.

---

## 3. Experimental Validation & Results

We validated the system using a two-stage testing protocol on an NVIDIA Tesla T4 GPU in a Kaggle Notebook environment:

### 3.1 Milestone 1: Gradient Flow Validation
*   **Setup:** Render a single curved red Bezier stroke on a white canvas.
*   **Result:** The renderer successfully generated a smooth, variable-thickness stroke. Running a backward pass against an all-blue target canvas returned non-zero gradients across all control point coordinates $\mathbf{P}_i$, widths $w_i$, and RGB channels, verifying mathematical differentiability.

### 3.2 Milestone 2: Single-Image Reconstruction
*   **Setup:** Reconstruct a target black ring outline ($128 \times 128$ resolution) using 15 discrete drawing strokes.
*   **Optimization Hyperparameters:** Cosine temperature annealing ($\tau_{\text{start}} = 8.0 \to \tau_{\text{end}} = 0.5$), Adam optimizer ($lr=0.03$), Soft Boundary Penalty ($\lambda=1.0$), and $N=120$ steps per window.
*   **Result:** The agent successfully aligned the 15 strokes to form a clean, continuous black circle. The sliding-window algorithm successfully coordinated overlapping segments without triggering OOM errors or coordinate drift, completing the reconstruction in under 2 minutes.

---

## 4. Next Steps & Milestone 3 (Neural Policy Guidance)
Having established the differentiable environment and optimization loop, the next phase is to train a **Policy Network** to guide the drawing process:
1.  **State-Target Dual Encoder:** Implement a Vision Transformer (ViT) with cross-attention to analyze differences between the canvas and target image.
2.  **Action Prediction:** Output stroke parameter probability distributions (Beta distribution) to guide drawing order.
3.  **Policy-Guided Differentiable Refinement (PGDR):** Combine the policy network's quick initial drawing layout with the differentiable renderer's local gradient descent to achieve real-time, human-like painting.
