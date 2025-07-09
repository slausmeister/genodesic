import torch
import torch.nn as nn
from .base_class import BaseDensityModel, _get_activation
import FrEIA.framework as Ff
from FrEIA.modules import RationalQuadraticSpline
from typing import Optional


def subnet_fc(dims_in: int, dims_out: int, width: int = 512, activation: str = "selu"):
    activation_fn = _get_activation(activation)
    return nn.Sequential(
        nn.Linear(dims_in, width),
        activation_fn,
        nn.Linear(width, width),
        activation_fn,
        nn.Linear(width, width),
        activation_fn,
        nn.Linear(width, dims_out)
    )

def build_rq_nsf_model(dim: int, n_blocks: int, bins: int, subnet_width: int, subnet_activation: str):
    
    # Create a lambda to pass parameters to the subnet constructor
    subnet_constructor = lambda dims_in, dims_out: subnet_fc(
        dims_in, dims_out, width=subnet_width, activation=subnet_activation
    )
    
    flow = Ff.SequenceINN(dim)
    for _ in range(n_blocks):
        flow.append(
            RationalQuadraticSpline,
            subnet_constructor=subnet_constructor,
            bins=bins
        )
    return flow

class RQNSFModel(BaseDensityModel, nn.Module):
    def __init__(self, model: nn.Module, dim: int, device: Optional[str] = None, **kwargs):
        super().__init__()
        self.dim = dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

    def to(self, device):
        """
        Move the model to the specified device.
        """
        self.device = device
        self.model.to(device)

    def parameters(self):
        """
        Expose parameters of the underlying model for optimizer.
        """
        return self.model.parameters()

    @torch.no_grad()
    def generate_samples(self, num_samples: int = None, base_samples: torch.Tensor = None):
        """
        Generate samples from the learned distribution.
        """
        if base_samples is not None:
            z = base_samples.to(self.device)
        elif num_samples is not None:
            z = torch.randn(num_samples, self.dim, device=self.device)
        else:
            raise ValueError("Either 'num_samples' or 'base_samples' must be provided.")

        # Flow inverse to go from z -> x
        x, _ = self.model(z, rev=True)
        return x.cpu().numpy()

    def compute_log_density(self, batch: torch.Tensor):
        """
        Compute log p(x) using the normalizing flow.
        """
        x = batch.to(self.device)
        z, log_jac = self.model(x)
        # Compute standard normal log-prob
        log_p_z = -0.5 * (z**2).sum(dim=1) - 0.5 * self.dim * torch.log(torch.tensor(2.0 * torch.pi, device=self.device))
        return log_p_z + log_jac
    
    def reverse(self, batch: torch.Tensor):
        """
        Compute log p(x) using the normalizing flow.
        """
        x = batch.to(self.device)
        z, _= self.model(x)
        # Compute standard normal log-prob
        log_p_z = -0.5 * (z**2).sum(dim=1)
        return log_p_z

    def compute_score(self, batch: torch.Tensor):
        """
        Compute the score (gradient of log-density w.r.t. input).
        """
        x = batch.clone().detach().to(self.device).requires_grad_(True)
        log_density = self.compute_log_density(x)
        score = torch.autograd.grad(log_density.sum(), x, create_graph=False)[0]
        return score
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cuda"):
        """Loads an RQNSFModel from a checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if checkpoint.get('model_type') != 'rqnsf':
            raise ValueError("Checkpoint is not for an 'rqnsf' model.")

        params = checkpoint['hyperparameters']
        
        network = build_rq_nsf_model(
            dim=params['dim'],
            n_blocks=params['n_blocks'],
            bins=params['bins'],
            subnet_width=params['subnet_width'],
            subnet_activation=params['subnet_activation']
        )
        network.load_state_dict(checkpoint['model_state_dict'])
        
        final_model = cls(model=network, dim=params['dim'], device=device)
        final_model.eval()
        return final_model