# Adjacency matrix and graph definitions
import torch

def get_static_adjacency_matrix(num_nodes=8):
    """
    Defines the baseline 'Hydraulic Piping' of the global market.
    
    Nodes Mapping:
    0: US_SP500 (Source)
    1: US_Bond10Y (Gravity)
    2: Dollar_Idx (Valve)
    3: Crude_Oil (Viscosity)
    4: KR_KOSPI (Target 1)
    5: JP_Nikkei (Target 2)
    6: HK_HangSeng (Target 3)
    7: DE_DAX (Bridge)
    
    Logic:
    - Directional flow from US (Source) to targets.
    - Interaction between Bond/Dollar and Indices.
    - Regional connection (Germany -> Asia).
    """
    adj = torch.zeros((num_nodes, num_nodes))
    
    # --- 1. Main Pressure Lines (US -> World) ---
    adj[0, 4] = 1 # US -> KR
    adj[0, 5] = 1 # US -> JP
    adj[0, 6] = 1 # US -> HK
    adj[0, 7] = 1 # US -> DE
    
    # --- 2. Valve & Gravity Effects ---
    adj[1, 0] = 1 # Bond -> SP500 (Yield pressure)
    adj[2, 4] = 1 # Dollar -> KR (Exchange rate impact)
    adj[2, 5] = 1 # Dollar -> JP
    
    # --- 3. Regional Bridges ---
    adj[7, 4] = 1 # DE -> KR (Europe open -> Asia open next day)
    adj[5, 4] = 1 # JP -> KR (Intra-Asia correlation)
    
    # --- 4. Viscosity (Commodity) ---
    adj[3, 0] = 1 # Oil -> US (Inflation)
    adj[3, 4] = 1 # Oil -> KR (Import cost)

    # Add Self-loops (Internal market momentum)
    adj = adj + torch.eye(num_nodes)
    
    return adj