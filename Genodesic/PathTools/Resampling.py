import numpy as np
import torch
from typing import Optional

def DensityBasedResampling(
    phi: torch.Tensor,
    num_points: int,
    model,                                 
    *,
    beta: float = 0.5,                     
    device: str | torch.device = "cuda",
    decluster: bool = False,               
    pruning_threshold: float = 1e-3,       
    samples_per_segment: int = 3,          
    eps: float = 1e-9                      
) -> torch.Tensor:
    """
    Smooth-resamples a path in latent space, shortening dense clusters
    and re-parameterising by a density-weighted Fermat metric.

    Parameters
    ----------
    phi : torch.Tensor           (L, D)  input polyline (path)
    num_points : int                    desired number of output samples
    model : ScoreSDEModel-like          must expose .compute_log_density(x)
    beta : float, default 0.5           density exponent (length / ρ^β)
    device : str | torch.device         device for torch ops
    decluster : bool, default False     enable iterative interior pruning
    pruning_threshold : float           merge interior nodes closer than this
    samples_per_segment : int, default 3  density probes per segment (1–3–1)
    eps : float, default 1e-9           floor to avoid zero densities

    Returns
    -------
    torch.Tensor  (num_points, D)       resampled path
    """
    if len(phi) < 3:
        return phi                     # nothing to prune / resample

    # -- 1. Optional iterative interior pruning --------------------
    phi_pruned = phi.clone()
    if decluster:
        while True:
            # distances between *adjacent interior* points (exclude ends)
            dists = torch.norm(torch.diff(phi_pruned[1:-1], dim=0), dim=1)
            if dists.numel() == 0:
                break
            min_dist, idx_local = torch.min(dists, dim=0)
            if min_dist.item() >= pruning_threshold:
                break
            # merge the closest pair into their midpoint
            idx_global = idx_local.item() + 1
            midpoint = 0.5 * (phi_pruned[idx_global] + phi_pruned[idx_global + 1])
            phi_pruned = torch.cat([phi_pruned[:idx_global],
                                    midpoint.unsqueeze(0),
                                    phi_pruned[idx_global + 2:]], dim=0)

    if len(phi_pruned) < 2:
        return phi_pruned.to(device)

    # -- 2. Euclidean segment lengths --------------------------------
    phi_np = phi_pruned.cpu().numpy()
    seg_vecs = np.diff(phi_np, axis=0)
    eucl_len = np.linalg.norm(seg_vecs, axis=1)      

    if eucl_len.sum() == 0.0:
        return phi_pruned.to(device)

    # -- 3. Density estimation (samples_per_segment probes) ----------
    # collect mid-segment probes at (1/(m+1), 2/(m+1), …, m/(m+1))
    probes = []
    p1 = phi_pruned[:-1]
    p2 = phi_pruned[1:]
    for j in range(1, samples_per_segment + 1):
        probes.append(p1 + (j / (samples_per_segment + 1)) * (p2 - p1))
    probes = torch.cat(probes, dim=0)          

    log_rho = model.compute_log_density(probes)
    rho = torch.exp(log_rho).view(samples_per_segment, -1).mean(dim=0) + eps
    rho_np = rho.detach().cpu().numpy()      

    # -- 4. Fermat metric segment lengths ---------------------------
    fermat_len = eucl_len / (rho_np ** beta) 

    # -- 5. Parametrisation / re-sampling ---------------------------
    #   t_eucl  : cumulative normalised Euclidean arclength
    #   t_fermat: cumulative normalised Fermat arclength
    cum_eucl = np.concatenate([[0], np.cumsum(eucl_len)])
    t_eucl = cum_eucl / cum_eucl[-1]

    cum_fm = np.concatenate([[0], np.cumsum(fermat_len)])
    t_fm = cum_fm / cum_fm[-1]

    t_new_fm = np.linspace(0.0, 1.0, num_points)
    t_new_eucl = np.interp(t_new_fm, t_fm, t_eucl)

    # dimension-wise interpolation
    interp_coords = [
        np.interp(t_new_eucl, t_eucl, phi_np[:, dim]) for dim in range(phi_np.shape[1])
    ]
    phi_interp = np.stack(interp_coords, axis=1)

    return torch.tensor(phi_interp, dtype=phi.dtype, device=device)
