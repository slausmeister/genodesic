import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from .base_class import BaseDensityModel 
from torchdyn.core import NeuralODE
from torchcfm.utils import torch_wrapper
from torch.cuda.amp import autocast


class MLP(nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False, d=None):
        super().__init__()
        print(f"Creating MLP with dim={dim}, out_dim={out_dim}, w={w}, time_varying={time_varying}")
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class OptimalFlowModel(BaseDensityModel):
    def __init__(self, dim=16, w=512, time_varying=True, device=None):
        super().__init__()
        self.dim = dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MLP(dim=dim, w=w, time_varying=time_varying).to(self.device)
        
        self.node = NeuralODE(
            torch_wrapper(self.model),
            solver="dopri5",
            sensitivity="adjoint",
            atol=1e-4,
            rtol=1e-4
        )

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cuda"):
        """Loads an OptimalFlowModel from a checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if checkpoint.get('model_type') != 'otcfm':
            raise ValueError("Checkpoint is not for an 'otcfm' model.")

        params = checkpoint['hyperparameters']
        
        # Instantiate the model shell
        final_model = cls(
            dim=params['dim'],
            w=params['hidden_dim'], 
            device=device
        )
        
        # Load the state dict into the model
        final_model.model.load_state_dict(checkpoint['model_state_dict'])
        final_model.model.eval()
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
        """
        Calculate per-sample log-density for a given batch of data with optimized memory management.
        """
        batch = batch.to(self.device).requires_grad_(True)
        t_span = torch.linspace(0, 1, steps, device=self.device)
        traj = self.node.trajectory(batch, t_span=t_span)

        # Compute log probability in the base space
        base_samples = traj[-1].detach()
        log_prob_base = (-0.5 * (base_samples**2).sum(dim=1)) - (
            0.5 * base_samples.size(1) * torch.log(torch.tensor(2 * torch.pi, device=self.device))
        )
        del base_samples  # Remove base_samples to clear memory
        torch.cuda.empty_cache()

        # Initialize log-determinant Jacobian
        log_det_jacobian = torch.zeros(batch.size(0), device=self.device)

        # Hutchinson's Trace Estimator for divergence
        for t_idx in range(traj.shape[0] - 1):
            dt = t_span[t_idx + 1] - t_span[t_idx]
            t_input = t_span[t_idx].expand(batch.size(0), 1).to(self.device)

            traj_t = traj[t_idx].requires_grad_(True)
            input_with_time = torch.cat((traj_t, t_input), dim=1)

            divergence_estimate = 0
            for _ in range(num_div_estimates):
                noise = torch.randn_like(traj_t, device=self.device)
                with autocast():
                    vec_field = self.model(input_with_time)

                grad_output = torch.autograd.grad(
                    outputs=vec_field,
                    inputs=traj_t,
                    grad_outputs=noise,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True
                )[0]

                if grad_output is not None:
                    divergence_estimate += (noise * grad_output).sum(dim=1)

                # Clear per-iteration memory
                del noise, vec_field, grad_output
                torch.cuda.empty_cache()

            log_det_jacobian += (divergence_estimate / num_div_estimates) * dt

            # Clear per-iteration memory
            del traj_t, t_input, input_with_time, divergence_estimate
            torch.cuda.empty_cache()

        # Final log density computation
        log_density = log_prob_base + log_det_jacobian

        # Cleanup large tensors
        del traj, log_prob_base, log_det_jacobian, t_span
        torch.cuda.empty_cache()

        return log_density

    def compute_score(self, batch: torch.Tensor, num_div_estimates: int = 20, steps: int = 100) -> torch.Tensor:
        """
        Compute the score (gradient of log-density w.r.t data and time).
        """
        batch = batch.to(self.device).requires_grad_(True)
        
        # We need to compute log_density with graph tracking enabled for this batch
        # so we cannot use the decorated `compute_log_density` directly. 
        # The logic is duplicated here with `requires_grad` flow.
        
        t_span = torch.linspace(0, 1, steps, device=self.device)
        # Note: torch.autograd.grad will need the graph from the trajectory computation
        traj = self.node.trajectory(batch, t_span=t_span)
        
        # Compute log probability in the base space
        log_prob_base = (-0.5 * (traj[-1]**2).sum(dim=1)) - (
            0.5 * traj[-1].size(1) * torch.log(torch.tensor(2 * torch.pi, device=self.device))
        )
    
        # Initialize log-determinant Jacobian
        log_det_jacobian = torch.zeros(batch.size(0), device=self.device)
    
        # Hutchinson's Trace Estimator for divergence
        for t_idx in range(traj.shape[0] - 1):
            dt = t_span[t_idx + 1] - t_span[t_idx]
            t_input = t_span[t_idx].expand(batch.size(0), 1).to(self.device)
    
            traj_t = traj[t_idx]
            # We need to re-enable grad for this intermediate tensor
            traj_t.requires_grad_(True)
            input_with_time = torch.cat((traj_t, t_input), dim=1)
    
            divergence_estimate = 0
            for _ in range(num_div_estimates):
                noise = torch.randn_like(traj_t, device=self.device)

                with torch.autocast():
                    vec_field = self.model(input_with_time)
    
                grad_output = torch.autograd.grad(
                    outputs=vec_field,
                    inputs=traj_t,
                    grad_outputs=noise,
                    retain_graph=True,
                    create_graph=True, # Must be true to backprop through the divergence
                    allow_unused=True
                )[0]
    
                if grad_output is not None:
                    divergence_estimate += (noise * grad_output).sum(dim=1)
            
            log_det_jacobian = log_det_jacobian + (divergence_estimate / num_div_estimates) * dt
    
        # Combine log probability and Jacobian
        log_density = log_prob_base + log_det_jacobian
    
        # Final gradient to get the score
        score = torch.autograd.grad(
            outputs=log_density.sum(), # Summing to get a scalar output for autograd
            inputs=batch,
            create_graph=False
        )[0]
    
        # Final cleanup
        del traj, log_prob_base, log_det_jacobian, log_density
        torch.cuda.empty_cache()
    
        return score
