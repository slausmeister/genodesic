import matplotlib.pyplot as plt
import numpy as np
import torch

def calculate_euclidean_segment_lengths(phi: torch.Tensor) -> np.ndarray:
    """Calculates the Euclidean distance for each segment of a path."""
    if len(phi) < 2:
        return np.array([])
    diffs = torch.diff(phi, dim=0)
    lengths = torch.norm(diffs, p=2, dim=1)
    return lengths.detach().cpu().numpy()


def calculate_path_density(phi: torch.Tensor, model, s: int) -> np.ndarray:
    """
    Calculates the log-density of points interpolated along a path.
    """
    if len(phi) < 2:
        return np.array([])

    all_interpolated_segments = []

    for i in range(len(phi) - 1):
        start_point, end_point = phi[i], phi[i+1]
        t = torch.linspace(0., 1., s + 2, device=phi.device)[1:-1]
        interpolated_points = start_point + t.unsqueeze(1) * (end_point - start_point)
        all_interpolated_segments.append(interpolated_points)

    midpoints_tensor_3d = torch.stack(all_interpolated_segments)

    num_segments, s_val, dims = midpoints_tensor_3d.shape
    midpoints_tensor_2d = midpoints_tensor_3d.reshape(num_segments * s_val, dims)

    log_densities = model.compute_log_density(midpoints_tensor_2d)
        
    return log_densities.detach().cpu().numpy()

def calculate_fermat_length(
    path: torch.Tensor, 
    model, 
    beta: float, 
    s: int
) -> float:
    """
    Calculates the path length with respect to the Fermat metric.

    This function performs a numerical integration along the path, where the
    length of each small segment is weighted by the inverse of the local
    data density raised to the power of beta[cite: 8, 65].

    Args:
        path (torch.Tensor): A tensor of shape (N, D) representing the path coordinates.
        model: The model used to estimate log-density.
        beta (float): The hyperparameter controlling density weighting[cite: 8].
        s (int): The number of integration segments between each point in the path[cite: 71].

    Returns:
        float: The total calculated Fermat length of the path.
    """
    if len(path) < 2:
        return 0.0

    all_midpoints = []
    all_euclidean_distances = []

    # Discretize the path into finer segments for numerical integration
    for i in range(len(path) - 1):
        start_node, end_node = path[i], path[i+1]
        # Create S+1 points for S segments
        points = torch.linspace(0, 1, s + 1, device=path.device).unsqueeze(1)
        interpolated_line = start_node + points * (end_node - start_node)

        # Calculate midpoints and segment lengths
        for j in range(s):
            y_start, y_end = interpolated_line[j], interpolated_line[j+1]
            midpoint = 0.5 * (y_start + y_end)
            distance = torch.norm(y_end - y_start)
            all_midpoints.append(midpoint)
            all_euclidean_distances.append(distance)

    # Get density estimates for all midpoints in a single batch
    midpoints_tensor = torch.stack(all_midpoints)
    log_densities = model.compute_log_density(midpoints_tensor)
    
    # Convert log-density to density: p(x) = exp(log(p(x)))
    densities = torch.exp(log_densities.cpu())
    
    # Calculate Fermat length for each segment: dist / p(x)^beta 
    distances_tensor = torch.tensor(all_euclidean_distances, dtype=torch.float32)
    fermat_lengths = distances_tensor / (densities**beta)

    # The total length is the sum of the lengths of all small segments
    return fermat_lengths.sum().item()