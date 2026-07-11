import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TemporalEmbedding(nn.Module):
    def __init__(self, embedding_dim=256):
        """
        Projects a scalar normalized step index t/T in [0, 1] into a high-dimensional 
        sinusoidal embedding space, followed by a multi-layer perceptron.
        """
        super().__init__()
        self.dim = embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, t):
        # t: [B, 1]
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t * emb.unsqueeze(0) # [B, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1) # [B, dim]
        return self.mlp(emb) # [B, dim]

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=4, patch_size=16, embed_dim=256, img_size=224):
        """
        Splits a 4-channel image into 16x16 patches and projects them to token embeddings.
        """
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, in_channels, H, W]
        x = self.proj(x) # [B, embed_dim, H_patch, W_patch]
        x = x.flatten(2).transpose(1, 2) # [B, num_patches, embed_dim]
        return x

class CrossAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v):
        # q: [B, N_q, D], k: [B, N_k, D], v: [B, N_v, D]
        B, N_q, D = q.shape
        _, N_k, _ = k.shape
        
        # Project and reshape: [B, heads, N, head_dim]
        Q = self.q_proj(q).view(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.k_proj(k).view(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.v_proj(v).view(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # GPU accelerated scaled dot-product attention (handles FlashAttention under the hood)
        out = F.scaled_dot_product_attention(Q, K, V) # [B, heads, N_q, head_dim]
        
        # Concatenate and project back
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N_q, D)
        return self.out_proj(out)

class TransformerBlockWithAdaLN(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, time_dim=256):
        """
        Transformer block integrating Self-Attention and Adaptive Layer Normalization (AdaLN) 
        conditioned on the temporal embedding.
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.self_attn = CrossAttention(embed_dim, num_heads) # self-attn is cross-attn with identical Q,K,V
        
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        # AdaLN parameter generators (scale & shift parameters for two normalizations)
        self.adaln_proj = nn.Linear(time_dim, embed_dim * 4)

    def forward(self, x, t_emb):
        # x: [B, N, D], t_emb: [B, time_dim]
        # Generate scale & shift variables
        adaln_params = self.adaln_proj(t_emb) # [B, D * 4]
        scale1, shift1, scale2, shift2 = torch.chunk(adaln_params, 4, dim=-1) # Each [B, D]
        
        # Normalize and scale
        norm_x1 = self.ln1(x)
        x_cond1 = norm_x1 * (1.0 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        attn_out = self.self_attn(x_cond1, x_cond1, x_cond1)
        x = x + attn_out
        
        norm_x2 = self.ln2(x)
        x_cond2 = norm_x2 * (1.0 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        mlp_out = self.mlp(x_cond2)
        x = x + mlp_out
        
        return x

class DifferentiableGraphosPolicy(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=256, num_heads=8, num_layers=6, time_dim=256, action_dim=13):
        """
        Project Graphos Policy Network.
        Accepts: Target image, current canvas, target-canvas difference, and canvas alpha.
        Outputs: Shape parameters (alpha, beta) for continuous stroke Beta distributions, 
                 and a Critic scalar value estimate.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        
        # 1. Encoders & Embedding blocks
        self.patch_embed = PatchEmbedding(in_channels=4, patch_size=patch_size, embed_dim=embed_dim, img_size=img_size)
        self.num_patches = self.patch_embed.num_patches
        
        # Positional encodings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Temporal Embedding MLP
        self.time_embed = TemporalEmbedding(time_dim)
        
        # 2. Transformer layers (Target & Canvas encoders share weights)
        self.blocks = nn.ModuleList([
            TransformerBlockWithAdaLN(embed_dim, num_heads, time_dim)
            for _ in range(num_layers)
        ])
        
        # 3. Cross-Attention Fusion layers (Merge target tokens into canvas tokens)
        self.cross_attn_blocks = nn.ModuleList([
            CrossAttention(embed_dim, num_heads)
            for _ in range(2)
        ])
        self.fusion_norm = nn.LayerNorm(embed_dim)
        
        # 4. Actor-Critic Heads
        # Actor Head: Outputs 2 shape parameters (alpha, beta) per continuous parameter (action_dim parameters total)
        self.actor_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, action_dim * 2) # [B, action_dim * 2]
        )
        
        # Critic Head: Outputs scalar Value function V(s)
        self.critic_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
        
        # Weight initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _encode_path(self, x, t_emb):
        # x: [B, 4, H, W]
        B = x.shape[0]
        
        # Patch embedding
        tokens = self.patch_embed(x) # [B, num_patches, embed_dim]
        tokens = tokens + self.pos_embed
        
        # Concatenate CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1) # [B, 1, embed_dim]
        tokens = torch.cat((cls_tokens, tokens), dim=1) # [B, num_patches+1, embed_dim]
        
        # Forward pass through AdaLN blocks
        for block in self.blocks:
            tokens = block(tokens, t_emb)
            
        return tokens

    def forward(self, target_img, canvas_img, canvas_alpha, step_index):
        """
        Args:
            target_img: [B, 3, H, W] Target image
            canvas_img: [B, 3, H, W] Current canvas
            canvas_alpha: [B, 1, H, W] Canvas opacity channel
            step_index: [B, 1] Normalized step time t/T in [0, 1]
        Returns:
            alpha: [B, 13] Beta distribution shape parameter alpha
            beta: [B, 13] Beta distribution shape parameter beta
            value: [B, 1] Critic value function estimation V(s)
        """
        B = target_img.shape[0]
        device = target_img.device
        
        # Compute time embedding
        t_emb = self.time_embed(step_index) # [B, time_dim]
        
        # 1. Structure the 4-channel input paths
        # Target Path: Concat target image + difference map
        diff_img = target_img - canvas_img
        target_path = torch.cat([target_img, diff_img.mean(dim=1, keepdim=True)], dim=1) # [B, 4, H, W]
        
        # Canvas Path: Concat canvas image + canvas alpha
        canvas_path = torch.cat([canvas_img, canvas_alpha], dim=1) # [B, 4, H, W]
        
        # 2. Encode both paths using the shared weight transformer backbones
        target_tokens = self._encode_path(target_path, t_emb) # [B, N+1, D]
        canvas_tokens = self._encode_path(canvas_path, t_emb) # [B, N+1, D]
        
        # 3. Fuse Target information into Canvas token representation via Cross-Attention
        fused = canvas_tokens
        for cross_block in self.cross_attn_blocks:
            fused = fused + cross_block(fused, target_tokens, target_tokens)
            
        fused = self.fusion_norm(fused)
        
        # Extract the representation of the CLS token (token 0)
        cls_rep = fused[:, 0] # [B, embed_dim]
        
        # 4. Project to Actor parameters
        actor_out = self.actor_fc(cls_rep) # [B, action_dim * 2]
        alpha_raw, beta_raw = torch.chunk(actor_out, 2, dim=-1) # Each [B, action_dim]
        
        # Map shape parameters to Beta distribution space: Softplus(x) + 1.0
        # Prevents alpha/beta from drifting <= 1.0 which creates numeric instability
        alpha = F.softplus(alpha_raw) + 1.0001
        beta = F.softplus(beta_raw) + 1.0001
        
        # Project to Critic estimate
        value = self.critic_fc(cls_rep) # [B, 1]
        
        return alpha, beta, value

if __name__ == "__main__":
    print("=== Testing Project Graphos Policy Network ===")
    
    # Initialize model
    model = DifferentiableGraphosPolicy(img_size=224, num_layers=4, num_heads=4, embed_dim=128)
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Generate dummy batch
    B = 2
    target_img = torch.randn(B, 3, 224, 224)
    canvas_img = torch.randn(B, 3, 224, 224)
    canvas_alpha = torch.sigmoid(torch.randn(B, 1, 224, 224))
    step_index = torch.rand(B, 1)
    
    print("\nRunning Forward Pass...")
    alpha, beta, value = model(target_img, canvas_img, canvas_alpha, step_index)
    
    print("\n=== Model Output Check ===")
    print(f"Actor Alpha shape: {alpha.shape}")
    print(f"Actor Beta shape: {beta.shape}")
    print(f"Critic Value shape: {value.shape}")
    
    # Verify bounds
    print(f"Min Alpha: {alpha.min().item():.4f} | Max Alpha: {alpha.max().item():.4f}")
    print(f"Min Beta: {beta.min().item():.4f} | Max Beta: {beta.max().item():.4f}")
    
    assert alpha.min() > 1.0, "Actor outputs invalid Beta distribution shape parameter (alpha <= 1.0)"
    assert beta.min() > 1.0, "Actor outputs invalid Beta distribution shape parameter (beta <= 1.0)"
    
    # Test sampling
    dist = torch.distributions.Beta(alpha, beta)
    action = dist.sample()
    log_prob = dist.log_prob(action).sum(dim=-1)
    print(f"\nSampled Action shape: {action.shape} (All actions in range [{action.min().item():.4f}, {action.max().item():.4f}])")
    print(f"Log probability shape: {log_prob.shape}")
    
    print("\nSUCCESS: Policy network correctly outputs valid Beta distribution parameters and computes log likelihoods!")
