import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D

# Set academic style for plots
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'serif' 

class GHNVisualizer:
    """
    Visualization toolkit for the Global Hydraulic Network (GHN).
    v4 Update: Enabled interactive mode for 3D plots (plt.show).
    """
    
    def __init__(self, save_dir='img'):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def plot_regime_alpha(self, dates, alpha_values, threshold=0.1):
        """
        Plots the Entropy Gate's Alpha value.
        """
        plt.figure(figsize=(12, 5))
        
        plt.plot(dates, alpha_values, color='#D62728', linewidth=1.5, label='Crisis Probability (Alpha)')
        plt.fill_between(dates, alpha_values, color='#D62728', alpha=0.1)
        plt.axhline(y=threshold, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Crisis Threshold')

        plt.title('GHN Regime Detection: Entropy Valve Opening (Alpha)', fontsize=14, fontweight='bold')
        plt.ylabel('Valve Open Rate (0~1)')
        plt.ylim(-0.05, 1.05)
        plt.legend(loc='upper left')
        
        save_path = os.path.join(self.save_dir, 'ghn_regime_alpha.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f">>> [Visualizer] Alpha plot saved to: {save_path}")
        plt.close()

    def plot_contagion_heatmap(self, prediction_matrix, node_names, date_str):
        """
        Visualizes the 2D Heatmap.
        """
        horizons = ['1D', '5D', '21D']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(prediction_matrix, annot=True, fmt=".2f", 
                    cmap='RdBu', center=0, 
                    xticklabels=horizons, yticklabels=node_names,
                    linewidths=.5, cbar_kws={'label': 'Price Change (Z-Score)'})
        
        plt.title(f'Global Contagion Heatmap ({date_str})', fontsize=14, fontweight='bold')
        plt.xlabel('Time Horizon')
        plt.ylabel('Market Nodes')
        
        filename = f"contagion_heatmap_{date_str}.png"
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f">>> [Visualizer] 2D Heatmap saved to: {save_path}")
        plt.close()

    def plot_3d_surface(self, prediction_matrix, node_names, date_str):
        """
        [INTERACTIVE] 3D Surface Plot.
        This function will OPEN a window. You can rotate the graph with your mouse.
        The script will pause until you close the window.
        """
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        num_nodes = len(node_names)
        num_horizons = 3
        
        x = np.arange(num_horizons)
        y = np.arange(num_nodes)
        X, Y = np.meshgrid(x, y)
        Z = prediction_matrix

        # 1. Plot Surface
        surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='k', linewidth=0.3, alpha=0.85, antialiased=True)

        # 2. Adjust Aspect Ratio (Stretch Y-axis)
        ax.set_box_aspect((1, 2, 0.8)) 

        # 3. Title Adjustment
        ax.set_title(f'3D Hydraulic Pressure Topology ({date_str})', fontsize=18, fontweight='bold', y=1.02)
        
        # 4. Labels
        ax.set_xlabel('Time Horizon', fontsize=12, labelpad=10)
        ax.set_ylabel('Market Nodes', fontsize=12, labelpad=20)
        ax.set_zlabel('Pressure (Z-Score)', fontsize=12, labelpad=10)
        
        # 5. Ticks
        ax.set_xticks(np.arange(num_horizons))
        ax.set_xticklabels(['1D', '5D', '21D'], fontsize=11, fontweight='bold')
        
        ax.set_yticks(np.arange(num_nodes))
        ax.set_yticklabels(node_names, rotation=-15, va='center', ha='right', fontsize=10)

        # 6. Colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.1)
        cbar.set_label('Market Pressure (Z-Score)')

        # 7. Initial View Angle
        ax.view_init(elev=30, azim=-120)

        # Save the image first
        filename = f"3d_surface_{date_str}.png"
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f">>> [Visualizer] 3D Surface Plot saved to: {save_path}")
        
        # [NEW] Show Interactive Window
        print(">>> [Interactive] 3D Plot opened. Please close the window to continue...")
        plt.show() # This blocks execution until closed

    def plot_global_forecast_dashboard(self, dates, y_true, y_pred, node_names, horizon_idx=0):
        """
        Plots a dashboard (4x2 grid).
        """
        horizon_labels = ['1D (Next Day)', '5D (Weekly)', '21D (Monthly)']
        h_label = horizon_labels[horizon_idx]
        
        fig, axes = plt.subplots(4, 2, figsize=(16, 18))
        axes = axes.flatten()
        
        for i, node_name in enumerate(node_names):
            ax = axes[i]
            true_data = y_true[:, i, horizon_idx]
            pred_data = y_pred[:, i, horizon_idx]
            
            ax.plot(dates, true_data, label='Actual', color='black', alpha=0.5, linewidth=1)
            ax.plot(dates, pred_data, label='Predicted (GHN)', color='#1F77B4', linestyle='--', linewidth=1.5)
            
            ax.set_title(f"{node_name}", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.legend(loc='upper left')

        plt.suptitle(f'Global Forecast Dashboard: {h_label} Horizon', fontsize=22, fontweight='bold')
        plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.15, bottom=0.05)
        
        save_path = os.path.join(self.save_dir, f'global_forecast_dashboard_{h_label[:2]}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f">>> [Visualizer] Dashboard saved to: {save_path}")
        plt.close()