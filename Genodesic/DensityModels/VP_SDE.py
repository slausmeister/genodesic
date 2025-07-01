import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Callable
from tqdm import tqdm
from .base_class import BaseDensityModel


class TimeScoreNet(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=512, num_layers=8):
        super().__init__()
        self.input_layer = nn.Linear(input_dim + 1, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
        # Define hidden layers with skip connections
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        self.activations = nn.ModuleList([
            nn.SELU() for _ in range(num_layers)
        ])

    def forward(self, x, t):

        inp = torch.cat([x, t], dim=1)  # Combine x and t
        out = self.input_layer(inp)  # First layer
        
        # Hidden layers with skip connections
        for layer, activation in zip(self.hidden_layers, self.activations):
            residual = out  # Store previous output for the skip connection
            out = layer(out)
            out = activation(out)
            out = out + residual  # Add skip connection
        
        # Final output layer
        out = self.output_layer(out)
        return out


def rk4_step(self, x, t, dt, beta_fn, model, num_div_estimates):
    """
    One Runge–Kutta 4 step for x(t), plus a Hutchinson estimate of divergence.

    x: shape (B, D), requires_grad=True
    t: scalar float
    dt: scalar float
    returns:
      x_next  (B, D)
      div_est (B,)  [divergence at the 'combined' step, used for logdet integration]
    """

    # k1 = h(x, t)
    h1, div1 = h_and_div(self, x, t, beta_fn, model, num_div_estimates)
    # k2 = h(x + dt/2 * k1, t + dt/2)
    x2 = x + 0.5 * dt * h1
    h2, div2 = h_and_div(self, x2, t + 0.5 * dt, beta_fn, model, num_div_estimates)
    # k3 = h(x + dt/2 * k2, t + dt/2)
    x3 = x + 0.5 * dt * h2
    h3, div3 = h_and_div(self, x3, t + 0.5 * dt, beta_fn, model, num_div_estimates)
    # k4 = h(x + dt * k3,   t + dt)
    x4 = x + dt * h3
    h4, div4 = h_and_div(self, x4, t + dt, beta_fn, model, num_div_estimates)

    # Weighted sum for the next x
    x_next = x + dt * (h1 + 2.0*h2 + 2.0*h3 + h4) / 6.0

    # Average divergence for this step
    div_est = (div1 + 2.0*div2 + 2.0*div3 + div4) / 6.0

    # Cleanup
    del h1, h2, h3, h4, div1, div2, div3, div4, x2, x3, x4
    torch.cuda.empty_cache()

    return x_next, div_est

def h_and_div(self, x, t, beta_fn, model, num_div_estimates):
    """
    Compute h_theta(x,t) and its divergence via Hutchinson.

    x: (B, D), requires_grad=True
    t: scalar float
    returns:
      h_val:   (B, D)
      div_est: (B,)
    """

    B, D = x.shape
    # Expand time to shape (B, 1)
    t_batch = x.new_full((B, 1), t)

    # Beta(t), shape (B,1)
    b_t = beta_fn(t_batch)

    # Score s_theta(x,t), shape (B, D)
    score = model.time_score_model(x, t_batch)

    # h(x,t) = -1/2 b(t) x - 1/2 b(t) score
    h_val = -0.5 * b_t * x - 0.5 * b_t * score

    # div(h) = sum_j d/dx_j h_j
    div_accum = torch.zeros(B, device=x.device)

    # We need grad wrt x
    for _ in range(num_div_estimates):
        eps = torch.randn_like(x)  # shape (B, D)
        # ∇_x h_val, dotted with eps
        #   grad_outputs=eps => effectively sum_j eps_j * dh_j/dx  => (B, D)
        grad_out = torch.autograd.grad(
            outputs=h_val,     # (B, D)
            inputs=x,
            grad_outputs=eps,
            retain_graph=True,  # We repeat this for num_div_estimates
            create_graph=False
        )[0]  # also (B, D)

        if grad_out is not None:
            # sum over dimension -> shape (B,)
            div_accum += (eps * grad_out).sum(dim=1)

        # Cleanup per iteration
        del eps, grad_out
        torch.cuda.empty_cache()

    div_est = div_accum / num_div_estimates

    # Cleanup
    del b_t, score, div_accum
    torch.cuda.empty_cache()

    return h_val, div_est


class ScoreSDEModel(BaseDensityModel):
    def __init__(
        self, 
        time_score_model: nn.Module, 
        dim: int = 16, 
        steps: int = 250, 
        device: Optional[torch.device] = None, 
        beta_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        beta_min: Optional[float] = None,
        beta_max: Optional[float] = None
    ):
        self.dim = dim
        self.steps = steps
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.time_score_model = time_score_model.to(self.device)
        self.time_score_model.eval()
        self.beta_fn = beta_fn if beta_fn is not None else self.default_beta_t
        self.beta_min = beta_min if beta_min is not None else 1
        self.beta_max = beta_max if beta_max is not None else 20

    def default_beta_t(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    @torch.no_grad()
    def generate_samples(self, num_samples: int, base_samples: Optional[torch.Tensor] = None,
                         num_corrector_steps: int = 1, snr: float = 0.1):
        """
        Generates samples using a Predictor-Corrector sampler.

        Args:
            num_samples: The number of samples to generate.
            base_samples: Optional tensor of starting points from the base distribution.
            num_corrector_steps: The number of corrector steps to perform at each time step.
            snr: Signal-to-noise ratio for the corrector (Langevin) step size.
        """
        self.time_score_model.eval()

        if base_samples is not None:
            x = base_samples.to(self.device)
        else:
            x = torch.randn(num_samples, self.dim, device=self.device)

        t_vals = torch.linspace(1, 0, self.steps + 1, device=self.device)
        dt = t_vals[0] - t_vals[1]

        for i in tqdm(range(self.steps), desc="PC Sampling"):
            t_i = t_vals[i]
            t_batch = t_i.view(1, 1).expand(x.size(0), 1)
            b_i = self.beta_fn(t_batch)

            # --- PREDICTOR STEP (Reverse SDE) ---
            score = self.time_score_model(x, t_batch)
            drift = -0.5 * b_i * x - b_i * score
            diffusion = torch.sqrt(b_i)

            x_mean = x - drift * dt
            x = x_mean + diffusion * torch.sqrt(dt) * torch.randn_like(x)

            # --- CORRECTOR STEPS (Langevin MCMC) ---
            for _ in range(num_corrector_steps):
                # The corrector does not change the time t_i
                score = self.time_score_model(x, t_batch)

                # Calculate Langevin step size
                grad_norm = torch.norm(score.reshape(score.shape[0], -1), dim=-1).mean()
                noise_norm = np.sqrt(self.dim)
                langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2

                # Take a Langevin step
                x = x + langevin_step_size * score + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)

        return x.detach().cpu().numpy()


    @torch.no_grad()
    def compute_score(self, batch: torch.Tensor, noise_level = None):

        batch = batch.to(self.device)
        if noise_level is None:
            t_zero = torch.zeros(batch.shape[0], 1, device=self.device)
        else:
            t_zero = torch.full((batch.shape[0], 1), noise_level, device=self.device)
        return self.time_score_model(batch, t_zero)
    

    def compute_log_density(
        self,
        batch: torch.Tensor,
        steps: int = 250,
        num_div_estimates: int = 25
    ) -> torch.Tensor:
        """
        Compute per-sample log density under the trained ScoreSDEModel
        using the probability-flow ODE for a variance-preserving SDE.

        We'll do an RK4 integration forward in time from t=0->1,
        then compute log p(1) under a standard normal plus the accumulated
        integral of -div(h).

        Args:
            batch: (N, D) data points
            steps: number of discrete time steps for RK4
            num_div_estimates: how many Hutchinson draws for divergence at each sub-step

        Returns:
            log_density: (N,) torch.Tensor
        """
        self.time_score_model.eval()
        device = self.device
        batch = batch.to(device)  # shape (N, D)

        N, D = batch.shape

        # For storing final log densities
        log_density = torch.zeros(N, device=device)

        # Step size dt
        dt = 1.0 / steps

        # Current state
        x_t = batch.requires_grad_(True)  # (N, D)
        logdet = torch.zeros(N, device=device)  # track \int -div h dt

        # Integrate from t=0 to t=1 in steps
        t_cur = 0.0
        for _ in range(steps):
            # 1-step of RK4
            x_next, div_est = rk4_step(self, x_t, t_cur, dt, self.beta_fn, self, num_div_estimates)

            # logdet <- logdet + \int -div(h) dt
            # here we just approximate integral by div_est * dt
            logdet = logdet - div_est * dt

            # Move forward in time
            t_cur += dt

            # x_next for next iteration
            # Detach to avoid building massive computational graph
            x_t = x_next.detach().requires_grad_(True)

        # At t=1, x_t ~ Normal(0,I) approximately, so
        #   log p_1(x(1)) = -0.5 ||x(1)||^2 - 0.5 * D * log(2*pi)
        # Compute that
        norm_sq = (x_t**2).sum(dim=1)  # (N,)
        log_p1 = -0.5 * norm_sq - 0.5 * D * torch.log(torch.tensor(2.0 * torch.pi, device=device))

        # log p_0(x(0)) = log p_1(x(1)) + logdet
        log_density = log_p1 + logdet

        # Cleanup
        del x_t, logdet, norm_sq, log_p1
        torch.cuda.empty_cache()

        return log_density
    

    @torch.no_grad()
    def reverse(self, batch: torch.Tensor, steps: int = 250):
        """
        Map data points from data space (t=0) to the base distribution (t=1)
        using the probability-flow ODE for a variance-preserving SDE.
        This function returns the entire trajectory for convenience,
        mirroring the style of OptimalCellFlowModel.reverse.

        Args:
            batch: (N, D) data points at time t=0
            steps: number of time steps for the RK4 integrator

        Returns:
            traj: (steps+1, N, D) torch.Tensor
                The trajectory from t=0 to t=1 in 'steps' increments.
                traj[-1] is the final state, i.e. mapped to base distribution.
        """

        self.time_score_model.eval()
        device = self.device
        x_t = batch.to(device).detach().clone()

        # We'll store the entire trajectory in a list
        traj = [x_t.cpu()]  # to match your style, store on CPU to save GPU memory

        # Define the ODE drift for the probability-flow ODE
        # For a VP-SDE with drift f(x,t) = -0.5*beta(t)*x and G(x,t)=sqrt(beta(t)) * I,
        #   h_theta(x,t) = f(x,t) - 0.5 G G^T s_theta(x,t)
        #                = -0.5 beta(t) x - 0.5 beta(t) score
        def h_theta(self, x, t):
            # x: (N, D), t: float
            # shape expansions
            B, D = x.shape
            t_batch = x.new_full((B, 1), t)
            b_t = self.beta_fn(t_batch)      # shape (B, 1)
            score = self.time_score_model(x, t_batch)  # (B, D)
            return - 1/2 * b_t * x - 1/2 * b_t * score   # (B, D)

        # Standard fixed-step RK4
        dt = 1.0 / steps
        t_cur = 0.0

        # Each iteration: x_{n+1} = x_n + RK4_update(...)
        for _ in range(steps):
            x_t = x_t.requires_grad_(False)  # no divergence needed, so we don't need grad

            k1 = h_theta(self, x_t, t_cur)
            k2 = h_theta(self, x_t + 0.5 * dt * k1, t_cur + 0.5 * dt)
            k3 = h_theta(self, x_t + 0.5 * dt * k2, t_cur + 0.5 * dt)
            k4 = h_theta(self, x_t + dt * k3,      t_cur + dt)

            x_next = x_t + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            t_cur += dt

            # Detach for the next iteration (avoid building a massive graph)
            x_t = x_next.detach()

            # Save in trajectory list
            traj.append(x_t.cpu())

            # Cleanup step-wise
            del k1, k2, k3, k4, x_next
            torch.cuda.empty_cache()

        # Stack trajectory into shape (steps+1, N, D)
        traj = torch.stack(traj, dim=0)
        return traj