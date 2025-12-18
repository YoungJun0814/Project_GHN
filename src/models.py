import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph_utils import get_static_adjacency_matrix

class EntropyGatedGCN(nn.Module):
    """
    GHN Model V4: Debug Mode & Lower Threshold
    
    Fix:
    - Threshold lowered from 0.5 to 0.1 (to match local window volatility).
    - Added print statement to debug actual volatility values during training.
    """
    def __init__(self, num_nodes=8, input_window=20, hidden_dim=64, output_horizon=3):
        super(EntropyGatedGCN, self).__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # 1. Temporal Encoder
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        
        # 2. Physics-based Entropy Gate Parameters
        # Threshold: Lowered to 0.1 because local volatility is usually small.
        self.sensitivity = nn.Parameter(torch.tensor(10.0)) # Stronger reaction
        self.threshold = nn.Parameter(torch.tensor(0.1))    # Lower barrier
        
        # 3. Spatial Diffusion
        self.gcn_weight = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.gcn_weight)
        
        self.register_buffer('adj_static', get_static_adjacency_matrix(num_nodes))
        self.register_buffer('adj_global', torch.ones(num_nodes, num_nodes))

        # 4. Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_horizon)
        )
        
        self.res_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        batch_size = x.size(0)
        
        # --- Step 1: Gate Calculation ---
        # Volatility: Standard deviation across the window
        volatility = x.std(dim=2).mean(dim=1, keepdim=True) # [Batch, 1]
        
        # [DEBUG PRINT]: Check what the model actually sees! (Prints 1 in 100 times)
        if self.training and torch.rand(1).item() < 0.01:
            curr_vol = volatility.mean().item()
            curr_thr = self.threshold.item()
            print(f"  [DEBUG] Volatility: {curr_vol:.4f} | Threshold: {curr_thr:.4f} | Diff: {curr_vol - curr_thr:.4f}")

        # Physics Formula: Alpha = Sigmoid( Sensitivity * (Vol - Threshold) )
        gate_input = self.sensitivity * (volatility - self.threshold)
        alpha = torch.sigmoid(gate_input)
        
        # --- Step 2: LSTM ---
        x_flat = x.view(-1, x.size(2), 1)
        lstm_out, _ = self.lstm(x_flat)
        h = lstm_out[:, -1, :].view(batch_size, self.num_nodes, self.hidden_dim)
        
        # --- Step 3: Dynamic Adjacency ---
        alpha_broad = alpha.view(batch_size, 1, 1)
        adj_dynamic = (1 - alpha_broad) * self.adj_static.unsqueeze(0) + \
                      alpha_broad * self.adj_global.unsqueeze(0)
        
        row_sum = adj_dynamic.sum(dim=2, keepdim=True) + 1e-6
        adj_norm = adj_dynamic / row_sum
        
        # --- Step 4: GCN ---
        h_trans = torch.matmul(h, self.gcn_weight)
        h_diffused = torch.matmul(adj_norm, h_trans)
        h_out = F.relu(h_diffused)
        
        # --- Step 5: Decoder ---
        out = self.decoder(h_out)
        last_price = x[:, :, -1].unsqueeze(2)
        final_out = out + (last_price * self.res_weight)
        
        return final_out, alpha