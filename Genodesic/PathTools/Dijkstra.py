import heapq
from typing import List, Optional
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def find_dijkstra_path(
    start_idx: int,
    end_idx: int,
    data: np.ndarray,
    model=None,
    k: int = 10,
    use_density_weighting: bool = False,
    density_lambda: float = 1.0,
    density_power: float = 1.0,
    exponent_scale: float = 1.0,
    device: Optional[torch.device] = torch.device("cuda"),
) -> List[int]:
    """
    Finds the shortest path from start_idx to end_idx using a lazy Dijkstra's algorithm
    on a k-NN graph, optionally using density-weighted edges.

    Args:
        start_idx (int): Index of the starting point.
        end_idx (int): Index of the ending point.
        data (np.ndarray): N x D array of data points.
        model: Model with `.compute_log_density()` method, if density weighting is used.
        k (int): Number of neighbors for k-NN graph.
        use_density_weighting (bool): Whether to weight edges by density.
        density_lambda (float): Scaling factor for the density penalty.
        density_power (float): Power to raise the penalty term.
        exponent_scale (float): Exponential scaling for Euclidean weights (when not using density).
        device (torch.device): Device for torch operations.

    Returns:
        List[int]: List of indices representing the shortest path from start_idx to end_idx.
    """
    N = data.shape[0]

    print(f"Building k-NN graph (k={k}) for Dijkstra...")
    data_t = torch.from_numpy(data).to(device)
    nbrs = NearestNeighbors(n_neighbors=k).fit(data)
    _, knn_idx = nbrs.kneighbors(data)  # Shape: (N, k)

    print(f"Running lazy Dijkstra's from {start_idx} to {end_idx}...")
    dist = np.full(N, np.inf, dtype=np.float64)
    prev = np.full(N, -1, dtype=np.int32)
    dist[start_idx] = 0.0
    heap = [(0.0, start_idx)]

    while heap:
        d_u, u = heapq.heappop(heap)

        if d_u > dist[u]:
            continue
        if u == end_idx:
            break

        neighbors = knn_idx[u]
        p_u = data_t[u]
        p_neighbors = data_t[neighbors]

        # Euclidean distances
        euclidean_weights = torch.norm(p_neighbors - p_u, dim=1)

        if use_density_weighting:
            if model is None:
                raise ValueError("Model must be provided if use_density_weighting=True")

            midpoints = 0.5 * (p_u + p_neighbors)
            log_densities = model.compute_log_density(midpoints)
            penalties = torch.clamp(-log_densities, min=0)
            total_weights_tensor = euclidean_weights + density_lambda * (penalties ** density_power)
            total_weights = total_weights_tensor.detach().cpu().numpy()
        else:
            total_weights = euclidean_weights.cpu().numpy()
            total_weights = np.exp(total_weights * exponent_scale)

        new_dist = d_u + total_weights

        for v_idx, v in enumerate(neighbors):
            if new_dist[v_idx] < dist[v]:
                dist[v] = new_dist[v_idx]
                prev[v] = u
                heapq.heappush(heap, (dist[v], v))

    # Path reconstruction
    path_indices = []
    curr = end_idx
    while curr != -1:
        path_indices.append(curr)
        curr = prev[curr]

    if not path_indices or path_indices[-1] != start_idx:
        print("Warning: Path not found between start and end nodes.")
        return np.empty((0, data.shape[1]), dtype=data.dtype)

    path_indices.reverse()
    return data[path_indices]
