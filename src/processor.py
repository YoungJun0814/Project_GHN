import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

class GHNDataProcessor:
    """
    Processes raw financial time-series into 3D tensors for the GHN model.
    
    This processor is designed to capture the 'Velocity of Contagion' by creating 
    specific prediction horizons that reflect modern market dynamics.
    
    Output Dimensions: [Batch, Nodes, Features]
    - Input Tensor (X): [Batch, 8, Window_Size]
    - Target Tensor (Y): [Batch, 8, Horizons]
    """
    
    def __init__(self, window_size=20, horizons=[1, 5, 21]):
        """
        Args:
            window_size (int): Look-back period (default 20 days = 1 month).
            horizons (list): Prediction targets.
                - 1D: Immediate shock (High-frequency reaction in 2025)
                - 5D: Short-term propagation (Weekly flow)
                - 21D: Medium-term equilibrium (Monthly trend)
        """
        self.window_size = window_size
        self.horizons = horizons
        self.scaler = StandardScaler()

    def load_and_process(self, file_path):
        """
        Loads CSV, scales data, and converts to tensors.
        """
        # 1. Load Data
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        print(f">>> [Processor] Loaded data with shape: {df.shape}")
        
        # 2. Split Data (Chronological Split)
        # Train (2005-2018), Val (2019-2021), Test (2022-Present)
        train_end = int(len(df) * 0.70)
        val_end = int(len(df) * 0.85)
        
        train_data = df.iloc[:train_end]
        val_data = df.iloc[train_end:val_end]
        test_data = df.iloc[val_end:]
        
        # 3. Fit Scaler (Only on Train data to prevent leakage)
        self.scaler.fit(train_data.values)
        
        # 4. Create Tensors
        X_train, Y_train = self._create_dataset(train_data)
        X_val, Y_val = self._create_dataset(val_data)
        X_test, Y_test = self._create_dataset(test_data)
        
        print(f">>> [Processor] Tensor Shapes:")
        print(f"    - Train X: {X_train.shape}, Y: {Y_train.shape}")
        print(f"    - Val   X: {X_val.shape},   Y: {Y_val.shape}")
        print(f"    - Test  X: {X_test.shape},  Y: {Y_test.shape}")
        
        return {
            'train': (X_train, Y_train),
            'val': (X_val, Y_val),
            'test': (X_test, Y_test),
            'scaler': self.scaler,
            'feature_names': df.columns.tolist()
        }

    def _create_dataset(self, df):
        """
        Internal function to slide window and create 3D tensors.
        """
        data = self.scaler.transform(df.values)
        X, Y = [], []
        
        num_rows = len(data)
        max_horizon = max(self.horizons)
        
        # Sliding Window
        for i in range(num_rows - self.window_size - max_horizon):
            # Input: Past [Window_Size] days for all nodes
            # Transpose to [Nodes, Window_Size] for GNN input standard
            window = data[i : i + self.window_size].T
            X.append(window)
            
            # Target: Future values at specific horizons [1, 5, 21]
            # Shape: [Nodes, 3]
            targets = []
            for h in self.horizons:
                targets.append(data[i + self.window_size + h - 1])
            
            # Stack horizons to get [Nodes, Horizons]
            Y.append(np.stack(targets, axis=1)) 
            
        return torch.tensor(np.array(X), dtype=torch.float32), \
               torch.tensor(np.array(Y), dtype=torch.float32)

if __name__ == "__main__":
    # Test execution
    import os
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'ghn_raw_data.csv')
    
    if os.path.exists(file_path):
        processor = GHNDataProcessor()
        datasets = processor.load_and_process(file_path)
    else:
        print(">>> [!] Data file not found. Please run data_fetcher.py first.")