# RFC-02: Direct Stroke Optimization & Single-Image Fitting

**Status:** Proposed  
**Authors:** Systems Engineer, Lead Scientist, Graphics Architect  
**Project:** Project Graphos, Orbit Studios R&D Lab  

---

## 1. Objective
To design and implement a robust, GPU-optimized training loop that fits a sequence of $K$ stroke parameters to reconstruct a single target image. The algorithm must prevent local minima traps, mitigate stroke occlusion gradient decay, and run efficiently in under 5 minutes on a Tesla T4 GPU.

---

## 2. Optimization Loop Design: Sliding-Window Blending

Optimizing all $K$ strokes simultaneously results in the **occlusion bottleneck**, where early strokes are covered by later strokes, collapsing their gradients to zero. 

To solve this, we implement a **Sliding Window Optimization** loop:
*   We only optimize a window of $W_{\text{active}}$ strokes (e.g., $W_{\text{active}} = 5$) simultaneously.
*   All strokes older than the active window are **frozen** and rendered statically into a background canvas tensor.
*   This bounds the autograd computation graph depth to $W_{\text{active}}$, eliminating gradient vanishing and keeping memory consumption at $O(1)$ relative to total strokes.

---

## 3. Mathematical Constraints & Training Mechanics

### A. Parameter Normalization
All parameters (control point coordinates, stroke widths, and RGB colors) are initialized and optimized in normalized space $[0, 1]^{13}$. When rendering, they are scaled up to pixel coordinates:
$$\mathbf{P}_{\text{pixel}} = \mathbf{P}_{\text{normalized}} \cdot \text{CanvasSize}$$
This ensures that the learning rate $\eta$ is scaled equally across geometry, width, and color channels.

### B. Soft Boundary Barrier Loss
To keep coordinates inside the canvas dimensions without causing vanishing gradients (which occurs with sigmoidal coordinate mapping), we apply a soft quadratic penalty:
$$\mathcal{L}_{\text{boundary}} = \lambda \sum_{i=0}^2 \left( \text{ReLU}(-x_i)^2 + \text{ReLU}(x_i - 1)^2 + \text{ReLU}(-y_i)^2 + \text{ReLU}(y_i - 1)^2 \right)$$

### C. Temperature Annealing (Cosine Schedule)
We decrease the Soft-SDF temperature $\tau$ over the iterations $s \in [0, N]$ using a cosine scheduler:
$$\tau(s) = \tau_{\text{end}} + \frac{1}{2}(\tau_{\text{start}} - \tau_{\text{end}}) \left( 1 + \cos\left( \frac{\pi s}{N} \right) \right)$$
*   **$\tau_{\text{start}} = 8.0$ pixels:** Blurs the stroke, providing wide attraction fields for coordinates.
*   **$\tau_{\text{end}} = 0.5$ pixels:** Sharpens the stroke, focusing gradients on details and pixel precision.

---

## 4. Run Execution Instructions

We have created the Python execution script:
### **[`fit_image.py`](file:///C:/Users/braya/Documents/antigravity/zealous-mendeleev/fit_image.py)**
This script contains the full sliding-window optimization loop. By default, it generates a target black circle outline and optimizes 15 red strokes to fit it.

### To execute on Colab:
1.  Copy the contents of `RFC_01_Differentiable_Canvas_Renderer.py` and `fit_image.py` into a new cell.
2.  Run the code block.
3.  The script will print steps and loss values, and save sequential canvas frames inside a folder named `drawing_steps/` as `stroke_01.png`, `stroke_02.png`, etc.
4.  You can inspect the `drawing_steps/` folder to see how the model literally draws the image stroke-by-stroke.
