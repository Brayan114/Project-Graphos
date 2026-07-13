import torch
import os
import matplotlib.pyplot as plt
from PIL import Image

# Import PPO training loop and visualization functions
from train_mnist_ppo import run_mnist_rl_sprint
from visualize_mnist_policy import run_and_save_reconstruction

def execute_complete_pipeline(epochs=60, batch_size=32, K=4):
    """
    Executes the complete Project Graphos MNIST pipeline:
    1. Trains the 14D Draw/Erase policy network on MNIST using PPO for specified epochs.
    2. Runs greedy policy evaluation to reconstruct a random target digit.
    3. Saves and displays the stroke progression grid and vector path overlays.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*80)
    print("🎨 PROJECT GRAPHOS COMPLETE TRAINING & EVALUATION PIPELINE 🎨")
    print("="*80)
    print(f"Device: {device.type.upper()}")
    print(f"Hyperparameters: Epochs={epochs} | Batch Size={batch_size} | Horizon K={K} strokes")
    print("="*80 + "\n")
    
    # 1. Run PPO training sprint
    run_mnist_rl_sprint(epochs=epochs, batch_size=batch_size, K=K)
    
    print("\n" + "="*80)
    print("📊 PILOT EVALUATION & PROGRESSION PLOTTING")
    print("="*80)
    
    # 2. Run greedy policy visualization
    grid_path = "mnist_reconstruction_grid.png"
    weights_path = "graphos_mnist_policy_10d.pth"
    
    run_and_save_reconstruction(
        weights_path=weights_path, 
        save_path=grid_path, 
        K=K
    )
    
    # 3. Display the resulting grid inline in the notebook
    if os.path.exists(grid_path):
        print(f"\nDisplaying progression grid from '{grid_path}':")
        img = Image.open(grid_path)
        plt.figure(figsize=(14, 7), dpi=150)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Project Graphos: PPO Policy Reconstruction (Epochs {epochs})", fontsize=12, fontweight='bold', pad=10)
        plt.show()
    else:
        print(f"Error: Progression grid image not found at '{grid_path}'")
        
    print("\nPipeline execution complete!")

if __name__ == "__main__":
    # Execute with 60 training epochs by default
    execute_complete_pipeline(epochs=60, batch_size=32, K=4)
