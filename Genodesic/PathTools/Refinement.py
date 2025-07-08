from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import yaml, torch
from pathlib import Path
from tqdm import tqdm

from Genodesic.PathTools.Resampling import DensityBasedResampling
from Genodesic.Utils.config_loader import _deep_merge



def relaxation_step(
    phi: torch.Tensor, model, beta: float, gamma: float, alpha: float, noise_level: Optional[float] = None
) -> torch.Tensor:
    """Performs a single relaxation-smoothing step on the path."""

    v = 0.5 * (phi[2:] - phi[:-2])
    scores = model.compute_score(phi[1:-1], noise_level=noise_level)

    # Energy minimization step
    phi_prime = 0.5 * (phi[2:] + phi[:-2]) + \
                (beta / 2) * (scores * (v.norm(dim=1, keepdim=True)**2) -
                              (torch.sum(scores * v, dim=1, keepdim=True) * v))


    # Smoothing step
    phi_prime_smoothed = phi_prime.clone()
    phi_prime_smoothed[1:-1] = (1 - gamma) * phi_prime[1:-1] + \
                               (gamma / 2) * (phi_prime[:-2] + phi_prime[2:])


    # Update step
    phi[1:-1] = alpha * phi_prime_smoothed + (1 - alpha) * phi[1:-1]

    return phi




# ---------- helper: density-weighted linear interpolation ------------------
def linear_interpolate_by_density(
    phi: torch.Tensor,
    *,
    num_points: int,
    model,
    beta: float,
    device: torch.device,
    decluster: bool = False,
) -> torch.Tensor:
    """Wrapper around DensityBasedResampling for code readability."""
    return DensityBasedResampling(
        phi=phi,
        num_points=num_points,
        model=model,
        beta=beta,
        device=device,
        decluster=decluster,
    )


# ---------- main loop ------------------------------------------------------
def run_refinement_loop(
    phi_initial,                     # (L, D)  NumPy array OR torch.Tensor
    model,
    *,
    config_overrides: Dict[str, Any] | None = None,
    default_config_path: str = "Config/refinement.yaml",
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Runs the geodesic relaxation loop and returns the final path plus history.

    Returns
    -------
    final_path : torch.Tensor          (N, D)
    path_history : list[torch.Tensor]  snapshots after each relaxation step
    """
    # ----------------------------------------------------------------------
    # 1. Load + merge configuration
    # ----------------------------------------------------------------------
    if not Path(default_config_path).exists():
        raise FileNotFoundError(default_config_path)

    with open(default_config_path, "r") as fh:
        cfg = yaml.safe_load(fh)["refinement"]

    if config_overrides:
        cfg = _deep_merge(config_overrides, cfg)

    # friendly names
    n_iter      = int(cfg["RELAX_ITERATIONS"])
    no0, no1    = float(cfg["NOISE_START"]), float(cfg["NOISE_END"])
    res_every   = int(cfg.get("RESAMPLE_EVERY", 50))

    # ----------------------------------------------------------------------
    # 2. Device & data preparation
    # ----------------------------------------------------------------------
    device = torch.device(cfg.get("device", "cuda")
                          if torch.cuda.is_available() else "cpu")

    if isinstance(phi_initial, np.ndarray):
        phi_curr = torch.as_tensor(phi_initial, dtype=torch.float32, device=device)
    else:
        phi_curr = phi_initial.to(device)

    model.to(device).eval()

    # ----------------------------------------------------------------------
    # 3. Initial density-based interpolation
    # ----------------------------------------------------------------------
    phi_curr = linear_interpolate_by_density(
        phi_curr,
        num_points=int(cfg["INTERPOLATION_POINTS"]),
        model=model,
        beta=float(cfg["INTERPOLATION_BETA"]),
        device=device,
    )

    path_history: List[torch.Tensor] = [phi_curr.clone()]

    # ----------------------------------------------------------------------
    # 4. Relaxation loop
    # ----------------------------------------------------------------------
    for i in tqdm(range(n_iter), desc="Relaxation"):
        # linear decay of noise level
        progress = i / max(1, n_iter - 1)
        noise_lvl = no0 * (1 - progress) + no1 * progress

        # periodic resampling to prevent point crowding
        if res_every and i > 0 and i % res_every == 0:
            phi_curr = linear_interpolate_by_density(
                phi_curr,
                num_points=int(cfg["INTERPOLATION_POINTS"]),
                model=model,
                beta=float(cfg["INTERPOLATION_BETA"]),
                device=device,
                decluster=True,
            )

        # one relaxation-smoothing step
        phi_curr = relaxation_step(
            phi_curr,
            model,
            beta=float(cfg["RELAX_BETA"]),
            gamma=float(cfg["RELAX_GAMMA"]),
            alpha=float(cfg["RELAX_ALPHA"]),
            noise_level=noise_lvl,
        )

        path_history.append(phi_curr.clone())

    # ----------------------------------------------------------------------
    # 5. Final “reverse” interpolation (beta → -beta)
    # ----------------------------------------------------------------------
    phi_curr = linear_interpolate_by_density(
        phi_curr,
        num_points=int(cfg["INTERPOLATION_POINTS"]),
        model=model,
        beta=-float(cfg["INTERPOLATION_BETA"]),
        device=device,
    )

    return phi_curr