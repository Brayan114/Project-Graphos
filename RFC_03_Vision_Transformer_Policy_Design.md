# RFC-03: Vision Transformer Policy Design

**Status:** Proposed  
**Authors:** Lead Scientist, Graphics Architect, Systems Engineer  
**Project:** Project Graphos, Orbit Studios R&D Lab  

---

## 1. Objective
To design and implement a Vision Transformer (ViT)-based Policy Network with cross-attention and temporal conditioning. The policy network will act as the "brain" of the drawing agent, mapping visual inputs and drawing progress to optimal continuous stroke distributions, while running efficiently under the hardware limits of a Tesla T4 GPU.

---

## 2. Input Space & State Representation

To ensure the agent has full context of the target, canvas, and transparency layers (necessary to distinguish white canvas from white paint), the input to the policy is a concatenated **8-channel tensor** $\mathbf{X}^{(t)} \in \mathbb{R}^{8 \times H \times W}$ at $224 \times 224$ resolution:

$$\mathbf{X}^{(t)} = \text{Concat}\left(\mathbf{I}_{\text{target}}, \mathbf{I}_{\text{canvas}}^{(t)}, \mathbf{I}_{\text{target}} - \mathbf{I}_{\text{canvas}}^{(t)}, \mathbf{A}_{\text{canvas}}^{(t)}\right)$$

Where:
*   $\mathbf{I}_{\text{target}}$: The RGB target image (3 channels).
*   $\mathbf{I}_{\text{canvas}}^{(t)}$: The RGB current canvas state (3 channels).
*   $\mathbf{I}_{\text{target}} - \mathbf{I}_{\text{canvas}}^{(t)}$: The signed residual difference map, directing attention to uncompleted features (3 channels).
*   $\mathbf{A}_{\text{canvas}}^{(t)}$: The canvas alpha channel, identifying where paint exists to guide erasing (1 channel).

---

## 3. Network Architecture

```
[Target Path: 4 channels] ──> [Shared ViT Encoder] ──> [Target Tokens] ──┐
                                                                          ├─> [Cross-Attention] ──> [AdaLN Temporal Projection] ──> [Heads]
[Canvas Path: 4 channels] ──> [Shared ViT Encoder] ──> [Canvas Tokens] ──┘
```

### A. Shared-Weight Dual-Encoder
*   The target path and canvas path are processed through a shared-weight Vision Transformer (ViT-Base) with patch size $P=16$. This produces token sequences $T_{\text{target}}$ and $T_{\text{canvas}}$ of length $N = (224/16)^2 + 1 = 197$.

### B. Cross-Attention Fusion
*   A stack of cross-attention layers allows the target tokens to query the canvas tokens:
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
    Where $Q = W_Q T_{\text{target}}$, $K = W_K T_{\text{canvas}}$, and $V = W_V T_{\text{canvas}}$.

### C. Sinusoidal Temporal Conditioning
To guide drawing style (sketching first, painting later), the step index $\bar{t} = t/T$ is projected into a time embedding $\mathbf{e}_t \in \mathbb{R}^D$ and injected into the Transformer blocks via **Adaptive Layer Normalization (AdaLN)**:
$$\text{AdaLN}(\mathbf{h}, \mathbf{e}_t) = \gamma(\mathbf{e}_t) \odot \text{LN}(\mathbf{h}) + \beta(\mathbf{e}_t)$$

---

## 4. Policy Distribution & Optimization

### A. Beta Distribution Actor Head
To avoid boundary clipping issues and keep continuous outputs strictly in $[0, 1]$, we map the latent projections to shapes $\alpha_i, \beta_i > 1$ using a Softplus function:
$$\alpha_i = \text{Softplus}(z^\alpha_i) + 1.0 + \epsilon, \quad \beta_i = \text{Softplus}(z^\beta_i) + 1.0 + \epsilon$$
The action vector is sampled from:
$$a_i \sim \text{Beta}(\alpha_i, \beta_i)$$

### B. Reward Formulation & GAE
*   **Reward:** Improvement in visual similarity minus a paint-wasting penalty:
    $$R_t = \mathcal{L}(C_{t-1}, I_{\text{target}}) - \mathcal{L}(C_t, I_{\text{target}}) - \gamma_p (w_t \cdot \alpha_t)$$
*   **Generalized Advantage Estimation (GAE):**
    $$\delta_t^V = R_t + \gamma V_\phi(S_{t+1}) - V_\phi(S_t), \quad \hat{A}_t = \sum_{l=0}^{T-t-1} (\gamma \lambda)^l \delta_{t+l}^V$$

### C. Self-Supervised Pre-Training (Warm-up)
Before running RL, we warm up the ViT backbone by generating synthetic Bezier stroke drawings $\mathcal{A}_{\text{gt}} \to I_{\text{synthetic}}$, and training the model to regress the parameter values directly:
$$\mathcal{L}_{\text{pretrain}} = \frac{1}{K} \sum_{k=1}^K \| \mathcal{A}_{\text{gt}, k} - \hat{\mathcal{A}}_k \|_1$$
This injects spatial-stroke mapping priors, stabilizing early RL iterations.

---

## 5. Tesla T4 Hardware Acceleration Setup

To run training smoothly on Colab/Kaggle's 16 GB T4 GPU:
1.  **FlashAttention Integration:** Use `torch.nn.functional.scaled_dot_product_attention` to bypass $O(N^2)$ attention map caching, reducing VRAM scaling linearly.
2.  **Parameter Efficiency (LoRA):** Freeze the core pre-trained ViT layers and inject **Low-Rank Adaptation (LoRA)** adapters (rank=8, alpha=16) on Q/K/V projections, reducing trainable parameters and optimizer states by >98%.
3.  **Batch Layout:** Micro-batch size $B_{\text{micro}}=8$, Gradient Accumulation Steps $= 8$ (effective batch size = 64).
4.  **Automatic Mixed Precision (AMP):** Run in FP16 to utilize the Turing Tensor Cores.
