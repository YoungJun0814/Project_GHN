# Project GHN (Physics-Informed GNN for Fluid Dynamics)

This project implements a Physics-Informed Graph Neural Network for fluid dynamics prediction.

## Project Structure
- `data/`: Collected CSV data
- `img/`: Result graphs and architecture diagrams
- `src/`: Core source code
    - `data_fetcher.py`: Data collection and time-lag correction
    - `processor.py`: 3D tensor generation (Windowing)
    - `graph_utils.py`: Adjacency matrix and graph definitions
    - `models.py`: Physics-Informed GNN models
    - `physics_loss.py`: Fluid dynamics-based loss functions
- `main.py`: Main execution script
- `.gitignore`: Git ignore file
- `README.md`: Project description
