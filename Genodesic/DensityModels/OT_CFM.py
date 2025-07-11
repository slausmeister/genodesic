import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from .base_class import BaseDensityModel, _get_activation
from torchdyn.core import NeuralODE
from torchcfm.utils import torch_wrapper


class MLP(nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False, activation="selu", num_layers=4):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        
        activation_fn = _get_activation(activation)
        
        layers = []
        # Input layer
        layers.extend([nn.Linear(dim + (1 if time_varying else 0), w), activation_fn])
        
        # Hidden layers
        for _ in range(num_layers):
            layers.extend([nn.Linear(w, w), activation_fn])
            
        # Output layer
        layers.append(nn.Linear(w, out_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class OptimalFlowModel(BaseDensityModel):
    def __init__(self, model: nn.Module, dim: int, device: Optional[str] = None):
        super().__init__()
        self.dim = dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.model.eval()
        
        self.node = NeuralODE(
            torch_wrapper(self.model),
            solver="dopri5",
            sensitivity="adjoint",
            atol=1e-4,
            rtol=1e-4
        )


    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cuda"):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if checkpoint.get('model_type') != 'otcfm':
            raise ValueError("Checkpoint is not for an 'otcfm' model.")
            
        params = checkpoint['hyperparameters']
        
        network = MLP(
            dim=params['dim'],
            w=params['hidden_dim'],
            time_varying=True,
            activation=params.get('activation', 'selu'),
            num_layers=params.get('num_layers', 4) 
        )
        network.load_state_dict(checkpoint['model_state_dict'])
        
        final_model = cls(model=network, dim=params['dim'], device=device)
        return final_model
    
    @torch.no_grad()
    def generate_samples(self, num_samples: Optional[int] = None, base_samples: Optional[torch.Tensor] = None, steps: int = 100):
        if base_samples is not None:
            samples = base_samples.to(self.device)
        elif num_samples is not None:
            samples = torch.randn(num_samples, self.dim).to(self.device)
        else:
            raise ValueError("Either 'num_samples' or 'base_samples' must be provided.")

        t_span = torch.linspace(0, 1, steps, device=self.device)
        traj = self.node.trajectory(samples, t_span=t_span)
        return traj[-1].cpu().numpy()
    
    @torch.no_grad()
    def reverse(self, batch: torch.Tensor, steps: int = 200) -> torch.Tensor:
        """
        Map data points from data space back to the base distribution.
        """
        batch = batch.to(self.device)
        t_span = torch.linspace(1, 0, steps, device=self.device)
        traj = self.node.trajectory(batch, t_span=t_span)
        return traj[-1]
    
    def compute_log_density(self, batch: torch.Tensor, num_div_estimates: int = 10, steps: int = 100) -> torch.Tensor:
        batch = batch.to(self.device).requires_grad_(True)
        t_span = torch.linspace(0, 1, steps, device=self.device)
        traj = self.node.trajectory(batch, t_span=t_span)

        base_samples = traj[-1].detach()
        log_prob_base = (-0.5 * (base_samples**2).sum(dim=1)) - (
            0.5 * base_samples.size(1) * torch.log(torch.tensor(2 * torch.pi, device=self.device))
        )
        del base_samples
        torch.cuda.empty_cache()

        log_det_jacobian = torch.zeros(batch.size(0), device=self.device)

        for t_idx in range(traj.shape[0] - 1):
            dt = t_span[t_idx + 1] - t_span[t_idx]
            t_input = t_span[t_idx].expand(batch.size(0), 1).to(self.device)
            traj_t = traj[t_idx].requires_grad_(True)
            input_with_time = torch.cat((traj_t, t_input), dim=1)

            divergence_estimate = 0
            for _ in range(num_div_estimates):
                noise = torch.randn_like(traj_t, device=self.device)
                with torch.autocast(device_type=self.device):
                    vec_field = self.model(input_with_time)
                
                grad_output = torch.autograd.grad(
                    outputs=vec_field, inputs=traj_t, grad_outputs=noise,
                    retain_graph=True, create_graph=False # FIX: create_graph is False
                )[0]

                if grad_output is not None:
                    divergence_estimate += (noise * grad_output).sum(dim=1)
                
                # FIX: Aggressive cleanup from prototype
                del noise, vec_field, grad_output
                torch.cuda.empty_cache()

            log_det_jacobian -= (divergence_estimate / num_div_estimates) * dt # Note: it's -= for fwd ODE

            del traj_t, t_input, input_with_time, divergence_estimate
            torch.cuda.empty_cache()
            
        log_density = log_prob_base + log_det_jacobian
        del traj, log_prob_base, log_det_jacobian, t_span
        torch.cuda.empty_cache()
        return log_density



    def compute_score(self, batch: torch.Tensor, num_div_estimates: int = 10, steps: int = 100) -> torch.Tensor:
        """
        Compute the score (gradient of log-density w.r.t data).
        This method is memory-intensive as it must build a graph for backpropagation.
        """
        batch = batch.to(self.device).requires_grad_(True)
        t_span = torch.linspace(0, 1, steps, device=self.device)
        
        traj = self.node.trajectory(batch, t_span=t_span)
        
        base_samples = traj[-1]
        log_prob_base = (-0.5 * (base_samples**2).sum(dim=1)) - (
            0.5 * base_samples.size(1) * torch.log(torch.tensor(2 * torch.pi, device=self.device))
        )
    
        log_det_jacobian = torch.zeros(batch.size(0), device=self.device)
    
        for t_idx in range(traj.shape[0] - 1):
            dt = t_span[t_idx + 1] - t_span[t_idx]
            t_input = t_span[t_idx].expand(batch.size(0), 1).to(self.device)
            traj_t = traj[t_idx].requires_grad_(True)
            input_with_time = torch.cat((traj_t, t_input), dim=1)
    
            divergence_estimate = 0
            for _ in range(num_div_estimates):
                noise = torch.randn_like(traj_t, device=self.device)
                with torch.autocast(device_type=self.device):
                    vec_field = self.model(input_with_time)
    
                grad_output = torch.autograd.grad(
                    outputs=vec_field, inputs=traj_t, grad_outputs=noise,
                    retain_graph=True, create_graph=False, allow_unused=True
                )[0]
    
                if grad_output is not None:
                    divergence_estimate += (noise * grad_output).sum(dim=1)
            
            log_det_jacobian -= (divergence_estimate / num_div_estimates) * dt

            del dt, t_input, traj_t, input_with_time, divergence_estimate, noise, vec_field, grad_output
            torch.cuda.empty_cache()

        log_density = log_prob_base + log_det_jacobian
        
        score = torch.autograd.grad(
            outputs=log_density.sum(), inputs=batch, create_graph=False
        )[0]
    
        del traj, base_samples, log_prob_base, log_det_jacobian, log_density
        torch.cuda.empty_cache()
    
        return score