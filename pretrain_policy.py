import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

# Import renderer and policy network
from RFC_01_Differentiable_Canvas_Renderer import DifferentiableBezierRenderer
from policy_network import DifferentiableGraphosPolicy

class SyntheticStrokeDataset(Dataset):
    def __init__(self, size=1000, K=5, H=224, W=224, device="cpu"):
        """
        Generates synthetic drawing data on-the-fly to save disk space.
        Each sample contains:
            - target_img: The rendered result of K random strokes on a white canvas.
            - canvas_img: The starting canvas (blank white).
            - canvas_alpha: The starting alpha channel (fully opaque/1.0 or fully empty/0.0).
            - target_actions: The ground-truth parameters [K, 13] that generated the drawing.
        """
        self.size = size
        self.K = K
        self.H = H
        self.W = W
        self.device = device
        
        # Instantiate a fast rendering client
        self.renderer = DifferentiableBezierRenderer(canvas_height=H, canvas_width=W, tau=0.3).to(device)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # We generate random strokes on-the-fly using PyTorch
        # Generating outside gradient tape
        with torch.no_grad():
            # 1. Sample K random strokes
            # Parameters: [P0_x, P0_y, P1_x, P1_y, P2_x, P2_y, w0, w1, w2, r, g, b, alpha]
            strokes = torch.rand((self.K, 13), device=self.device)
            # Clip stroke widths to be realistic (between 1.5% and 8% of canvas)
            strokes[:, 6:9] = 0.015 + 0.065 * strokes[:, 6:9]
            # Set mode parameter to Draw (1.0)
            modes = torch.ones((self.K, 1), device=self.device)
            
            # 2. Render sequentially on a blank white canvas
            canvas = torch.zeros((1, 4, self.H, self.W), device=self.device)
            canvas[:, :3, :, :] = 1.0 # White background
            canvas[:, 3:4, :, :] = 1.0 # Opaque
            
            initial_canvas = canvas.clone()
            
            current_canvas = canvas
            for k in range(self.K):
                p = strokes[k:k+1]
                # Scale coordinates and widths to pixel space
                cp = p[:, 0:6].view(1, 3, 2) * self.H
                w = p[:, 6:9] * self.H
                c = p[:, 9:12]
                alpha = p[:, 12:13]
                current_canvas = self.renderer(cp, w, c, alpha, modes[k:k+1], current_canvas)
                
            # Extract target RGB
            target_img = current_canvas[0, :3, :, :]
            canvas_img = initial_canvas[0, :3, :, :]
            canvas_alpha = initial_canvas[0, 3:4, :, :]
            
        return target_img.cpu(), canvas_img.cpu(), canvas_alpha.cpu(), strokes.cpu()

def train_pretraining_loop(epochs=5, batch_size=8, K=5, train_size=320, val_size=80):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Self-Supervised Pre-training Loop on: {device.type.upper()}")
    print(f"Targeting sequence length: {K} strokes. Res: 224x224")
    
    # 1. Create Datasets
    print("\nGenerating synthetic training datasets...")
    train_dataset = SyntheticStrokeDataset(size=train_size, K=K, device=device)
    val_dataset = SyntheticStrokeDataset(size=val_size, K=K, device=device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. Initialize Policy Network (using lightweight dimensions for T4 training speed)
    model = DifferentiableGraphosPolicy(img_size=224, num_layers=4, num_heads=4, embed_dim=128).to(device)
    
    # We add a supervised regression head to predict stroke parameters [B, K * 13]
    regression_head = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, K * 13)
    ).to(device)
    
    optimizer = optim.Adam(list(model.parameters()) + list(regression_head.parameters()), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"\nModel Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Regression Head Params: {sum(p.numel() for p in regression_head.parameters() if p.requires_grad):,}")
    print("Starting training runs...")
    
    for epoch in range(epochs):
        model.train()
        regression_head.train()
        train_loss = 0.0
        
        for step, (target_img, canvas_img, canvas_alpha, gt_strokes) in enumerate(train_loader):
            target_img = target_img.to(device)
            canvas_img = canvas_img.to(device)
            canvas_alpha = canvas_alpha.to(device)
            gt_strokes = gt_strokes.to(device) # [B, K, 13]
            
            optimizer.zero_grad()
            
            # Since this is pre-training on synthetic drawings starting from blank canvas,
            # the step index t/T is always 0.0
            step_index = torch.zeros((target_img.shape[0], 1), device=device)
            
            # Forward pass through dual-encoder
            # Extract CLS token representation
            t_emb = model.time_embed(step_index)
            diff_img = target_img - canvas_img
            target_path = torch.cat([target_img, diff_img.mean(dim=1, keepdim=True)], dim=1)
            canvas_path = torch.cat([canvas_img, canvas_alpha], dim=1)
            
            target_tokens = model._encode_path(target_path, t_emb)
            canvas_tokens = model._encode_path(canvas_path, t_emb)
            
            fused = canvas_tokens
            for cross_block in model.cross_attn_blocks:
                fused = fused + cross_block(fused, target_tokens, target_tokens)
            fused = model.fusion_norm(fused)
            cls_rep = fused[:, 0] # [B, embed_dim]
            
            # Predict stroke parameters
            pred_params = regression_head(cls_rep).view(-1, K, 13) # [B, K, 13]
            
            # Compute Parameter Regression Loss (L1)
            # Opacities, colors, widths, and coordinates are all in range [0, 1]
            loss = F.l1_loss(pred_params, gt_strokes)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        scheduler.step()
        
        # Validation pass
        model.eval()
        regression_head.eval()
        val_loss = 0.0
        with torch.no_grad():
            for target_img, canvas_img, canvas_alpha, gt_strokes in val_loader:
                target_img = target_img.to(device)
                canvas_img = canvas_img.to(device)
                canvas_alpha = canvas_alpha.to(device)
                gt_strokes = gt_strokes.to(device)
                
                step_index = torch.zeros((target_img.shape[0], 1), device=device)
                t_emb = model.time_embed(step_index)
                diff_img = target_img - canvas_img
                target_path = torch.cat([target_img, diff_img.mean(dim=1, keepdim=True)], dim=1)
                canvas_path = torch.cat([canvas_img, canvas_alpha], dim=1)
                
                target_tokens = model._encode_path(target_path, t_emb)
                canvas_tokens = model._encode_path(canvas_path, t_emb)
                
                fused = canvas_tokens
                for cross_block in model.cross_attn_blocks:
                    fused = fused + cross_block(fused, target_tokens, target_tokens)
                fused = model.fusion_norm(fused)
                cls_rep = fused[:, 0]
                
                pred_params = regression_head(cls_rep).view(-1, K, 13)
                loss = F.l1_loss(pred_params, gt_strokes)
                val_loss += loss.item()
                
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss/len(train_loader):.5f} | Val Loss: {val_loss/len(val_loader):.5f}")
        
    # Save the pre-trained weights
    torch.save(model.state_dict(), "pretrained_graphos_policy.pth")
    print("\n🎉 Pre-training complete! Weights saved to 'pretrained_graphos_policy.pth'.")

if __name__ == "__main__":
    # Run a short training test (5 epochs)
    train_pretraining_loop(epochs=5, batch_size=8, K=4, train_size=80, val_size=24)
