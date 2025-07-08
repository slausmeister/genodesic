from typing import Tuple
import torch
import numpy as np
from scipy.spatial import cKDTree

def compute_path_pseudotime(
    path: torch.Tensor, data: np.ndarray, pseudotime: np.ndarray, k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes the average and std dev of pseudotime for points along a path."""
    print(f"Computing average pseudotime along path (k={k})...")
    path_np = path.cpu().numpy()
    tree = cKDTree(data)
    _, indices = tree.query(path_np, k=k)
    
    avg_times = pseudotime[indices].mean(axis=1)
    std_times = pseudotime[indices].std(axis=1)
    
    return torch.from_numpy(avg_times), torch.from_numpy(std_times)