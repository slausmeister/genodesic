import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Callable
from tqdm import tqdm
from .base_class import BaseDensityModel, _get_activation


class TimeScoreNet(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=512, num_layers=8, activation="selu"):
        super().__init__()
        # --- Corrected nn.Module classes ---
        self.input_layer = nn.Linear(input_dim + 1, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        self.activations = nn.ModuleList([
            _get_activation(activation) for _ in range(num_layers)
        ])

    def forward(self, x, t):
        inp = torch.cat([x, t], dim=1)
        out = self.input_layer(inp)
        
        for layer, activation in zip(self.hidden_layers, self.activations):
            residual = out
            out = layer(out)
            out = activation(out)
            out = out + residual
            
        out = self.output_layer(out)
        return out


# --- Corrected Class Name ---
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
        
        super().__init__()

        self.dim = dim
        self.steps = steps
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.time_score_model = time_score_model.to(self.device)
        self.time_score_model.eval()
        self.beta_fn = beta_fn if beta_fn is not None else self.default_beta_t
        self.beta_min = beta_min if beta_min is not None else 1e-4
        self.beta_max = beta_max if beta_max is not None else 20

    def default_beta_t(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def _h_theta(self, x, t):
        """The drift of the probability flow ODE: h(x,t) = -1/2 * b(t) * (x + score)."""
        t_batch = t.view(1, 1).expand(x.size(0), -1) if t.ndim == 0 else t
        b_t = self.beta_fn(t_batch)
        score = self.time_score_model(x, t_batch)
        return -0.5 * b_t * (x + score)

    def _h_and_div(self, x, t, num_div_estimates):
        """Compute h_theta(x,t) and its divergence via Hutchinson's estimator."""
        h_val = self._h_theta(x, t)
        
        div_accum = torch.zeros(x.size(0), device=x.device)
        for _ in range(num_div_estimates):
            eps = torch.randn_like(x)
            grad_out = torch.autograd.grad(h_val, x, grad_outputs=eps, retain_graph=True)[0]
            div_accum += (eps * grad_out).sum(dim=1)
            
        return h_val, div_accum / num_div_estimates

    def _rk4_step(self, x, t, dt, num_div_estimates):
        """One Runge-Kutta 4 step for log-density calculation."""
        h1, div1 = self._h_and_div(x, t, num_div_estimates)
        h2, div2 = self._h_and_div(x + 0.5 * dt * h1, t + 0.5 * dt, num_div_estimates)
        h3, div3 = self._h_and_div(x + 0.5 * dt * h2, t + 0.5 * dt, num_div_estimates)
        h4, div4 = self._h_and_div(x + dt * h3, t + dt, num_div_estimates)

        x_next = x + dt * (h1 + 2 * h2 + 2 * h3 + h4) / 6.0
        div_est = (div1 + 2 * div2 + 2 * div3 + div4) / 6.0
        return x_next, div_est
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cuda"):
        """Loads a ScoreSDEModel from a checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if checkpoint.get('model_type') != 'vpsde':
            raise ValueError("Checkpoint is not for a 'vpsde' model.")

        params = checkpoint['hyperparameters']
        network = TimeScoreNet(
            input_dim=params['dim'],
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            activation=params.get('activation', 'selu')
        )
        network.load_state_dict(checkpoint['model_state_dict'])
        
        final_model = cls(
            time_score_model=network,
            dim=params['dim'],
            beta_min=params['beta_min'],
            beta_max=params['beta_max'],
            device=device
        )
        final_model.eval()
        return final_model

    @torch.no_grad()
    def generate_samples(self, num_samples: int, base_samples: Optional[torch.Tensor] = None,
                         num_corrector_steps: int = 1, snr: float = 0.1):
        """Generates samples using a Predictor-Corrector sampler."""
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
                score = self.time_score_model(x, t_batch)
                grad_norm = torch.norm(score.reshape(score.shape[0], -1), dim=-1).mean()
                noise_norm = np.sqrt(self.dim)
                langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
                x = x + langevin_step_size * score + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)

        return x.detach().cpu().numpy()


    @torch.no_grad()
    def compute_score(self, batch: torch.Tensor, noise_level: Optional[float] = None):
        batch = batch.to(self.device)
        if noise_level is None:
            t_zero = torch.zeros(batch.shape[0], 1, device=self.device)
        else:
            t_zero = torch.full((batch.shape[0], 1), noise_level, device=self.device)
        return self.time_score_model(batch, t_zero)
    

    def compute_log_density(self, batch: torch.Tensor, steps: Optional[int] = None, num_div_estimates: int = 10, **kwargs):
        """Computes per-sample log density using the probability-flow ODE."""
        self.time_score_model.eval()
        num_steps = steps if steps is not None else self.steps

        x_t = batch.to(self.device).requires_grad_(True)
        logdet = torch.zeros(x_t.size(0), device=self.device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.tensor(i * dt, device=self.device)
            x_next, div_est = self._rk4_step(x_t, t, dt, num_div_estimates)
            logdet -= div_est * dt
            x_t = x_next.detach().requires_grad_(True)

        log_p1 = -0.5 * (x_t**2).sum(dim=1) - 0.5 * self.dim * np.log(2.0 * np.pi)
        return log_p1 + logdet
    

    @torch.no_grad()
    def reverse(self, batch: torch.Tensor, steps: Optional[int] = None, **kwargs):
        """Maps data points from data space (t=0) to the base distribution (t=1)."""
        self.time_score_model.eval()
        num_steps = steps if steps is not None else self.steps
        x_t = batch.to(self.device).detach().clone()
        dt = 1.0 / num_steps
        
        trajectory = [x_t.cpu()]
        
        for i in range(num_steps):
            t = torch.tensor(i * dt, device=self.device)
            k1 = self._h_theta(x_t, t)
            k2 = self._h_theta(x_t + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = self._h_theta(x_t + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = self._h_theta(x_t + dt * k3, t + dt)
            x_t = x_t + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            trajectory.append(x_t.cpu())
    
        return torch.stack(trajectory, dim=0)