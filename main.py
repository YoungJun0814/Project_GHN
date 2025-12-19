import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np

# Import custom modules from the src folder
from src.data_fetcher import fetch_ghn_data
from src.processor import GHNDataProcessor
from src.models import EntropyGatedGCN
from src.physics_loss import HydraulicLoss
from src.visualizer import GHNVisualizer  # Import the visualizer class

def main():
    # --- Step 1: Data Preparation ---
    raw_data_path = os.path.join('data', 'ghn_raw_data.csv')
    
    # Download data if it doesn't exist
    if not os.path.exists(raw_data_path):
        print(">>> Raw data not found. Starting download...")
        fetch_ghn_data()

    # Process data into 3D Tensors
    # Horizons: [1D(Next Day), 5D(Weekly), 21D(Monthly)]
    processor = GHNDataProcessor(window_size=20, horizons=[1, 5, 21])
    datasets = processor.load_and_process(raw_data_path)
    
    X_train, Y_train = datasets['train']
    X_val, Y_val = datasets['val']
    
    # Create PyTorch DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=32, shuffle=False)

    # --- Step 2: Model & Optimizer Initialization ---
    # [Device Setup] 
    # Current environment (PyTorch 2.9.1+cu128) supports RTX 5070 Laptop GPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using device: {device}")

    # Initialize V4 Model (Physics-Hardcoded Gate)
    model = EntropyGatedGCN(num_nodes=8, input_window=20, hidden_dim=64).to(device)
    criterion = HydraulicLoss(lambda_smooth=0.1, lambda_flow=0.05)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Step 3: Training Loop ---
    epochs = 50
    print(f">>> Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Forward Pass
            # output: [Batch, 8, 3], alpha: [Batch, 1]
            optimizer.zero_grad()
            output, alpha = model(batch_x)
            
            # Calculate Physics-Informed Loss
            loss, loss_details = criterion(output, batch_y, alpha)
            
            # Backward Pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        # Validation Step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_out, v_alpha = model(vx)
                v_loss, _ = criterion(v_out, vy, v_alpha)
                val_loss += v_loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Alpha: {alpha.mean().item():.3f}")

    # --- Step 4: Save Model ---
    if not os.path.exists('models'):
        os.makedirs('models')
    model_save_path = 'models/ghn_model_v1.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f">>> Training complete. Model saved to {model_save_path}")

    # --- Step 5: Visualization & Analysis ---
    print(">>> Starting Visualization Analysis...")
    
    # Initialize Visualizer
    viz = GHNVisualizer(save_dir='img')
    
    # Prepare Test Data for Visualization
    X_test, Y_test = datasets['test']
    X_test = X_test.to(device)
    node_names = datasets['feature_names']

    # Run Inference on Test Set
    model.eval()
    with torch.no_grad():
        test_output, test_alpha = model(X_test)
        
    # Convert tensors to numpy for plotting
    alpha_np = test_alpha.cpu().numpy().flatten()
    output_np = test_output.cpu().numpy()
    y_true_np = Y_test.cpu().numpy()
    
    # 1. Plot Alpha Regime (Entire Test Period)
    viz.plot_regime_alpha(range(len(alpha_np)), alpha_np)
    
    # 2. Plot Contagion Heatmap & 3D Topology
    max_crisis_idx = np.argmax(alpha_np)
    print(f">>> Generating Graphs for Crisis Sample Index: {max_crisis_idx} (Alpha: {alpha_np[max_crisis_idx]:.4f})")
    
    # 2D Heatmap
    viz.plot_contagion_heatmap(output_np[max_crisis_idx], node_names, f"Test_Sample_{max_crisis_idx}")

    # 3D Surface Plot (INTERACTIVE)
    viz.plot_3d_surface(output_np[max_crisis_idx], node_names, f"Test_Sample_{max_crisis_idx}")

    # 3. Plot Global Forecast Dashboard
    print(">>> Generating Global Forecast Dashboard...")
    viz.plot_global_forecast_dashboard(
        range(len(y_true_np)), 
        y_true_np, 
        output_np, 
        node_names, 
        horizon_idx=0
    )
    
    print(">>> All visualizations saved to 'img/' folder.")

if __name__ == "__main__":
    main()