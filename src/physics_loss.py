import torch
import torch.nn as nn
import torch.nn.functional as F

class HydraulicLoss(nn.Module):
    """
    Physics-Informed Loss Function for GHN.
    
    Components:
    1. MSE Loss: Accuracy of prediction.
    2. Temporal Smoothness Loss: Penalizes jagged predictions across 1D->5D->21D.
    3. Flow Consistency Loss: Enforces correlation consistency between Source and Targets during crisis.
    """
    def __init__(self, lambda_smooth=0.1, lambda_flow=0.05):
        super(HydraulicLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.lambda_smooth = lambda_smooth
        self.lambda_flow = lambda_flow
        
        # Node Indices for Flow Loss
        self.source_idx = 0  # SP500
        self.target_indices = [4, 5, 6] # KR, JP, HK

    def forward(self, predictions, targets, alpha):
        """
        predictions: [Batch, Nodes, Horizons]
        targets: [Batch, Nodes, Horizons]
        alpha: [Batch, 1] (Entropy Valve value)
        """
        
        # 1. Data Fidelity Term (MSE)
        loss_mse = self.mse(predictions, targets)
        
        # 2. Temporal Smoothness Term (Physics: Inertia)
        # The curve [1D, 5D, 21D] should be smooth (2nd derivative ~ 0)
        # Diff 1: (5D - 1D), (21D - 5D)
        diff1 = predictions[:, :, 1:] - predictions[:, :, :-1]
        # Diff 2: Acceleration change
        diff2 = diff1[:, :, 1:] - diff1[:, :, :-1]
        loss_smooth = torch.mean(diff2 ** 2)
        
        # 3. Flow Consistency Term (Physics: Momentum Transfer)
        # If Alpha (Crisis) is high, the movement of Source (US) should strongly 
        # dictate the movement of Targets (Asia).
        # We penalize if Source and Targets move in OPPOSITE directions when alpha is high.
        
        # Get 1D movement (Immediate reaction)
        source_move = predictions[:, self.source_idx, 0] # [Batch]
        target_moves = predictions[:, self.target_indices, 0] # [Batch, 3]
        
        # Directional agreement: Source * Target > 0 implies same direction
        # We want to minimize negative correlation during crisis.
        # Loss = Alpha * ReLU( - (Source * Target_Mean) )
        # If Source and Target move opposite, product is negative, minus makes it positive -> Penalty.
        
        flow_alignment = source_move.unsqueeze(1) * target_moves # [Batch, 3]
        flow_penalty = F.relu(-flow_alignment).mean(dim=1) # [Batch]
        
        # Weighted by Alpha: Only enforce this physics during high entropy (Crisis)
        loss_flow = torch.mean(alpha.squeeze() * flow_penalty)
        
        # Total Loss
        total_loss = loss_mse + (self.lambda_smooth * loss_smooth) + (self.lambda_flow * loss_flow)
        
        return total_loss, {
            "mse": loss_mse.item(),
            "smooth": loss_smooth.item(),
            "flow": loss_flow.item()
        }