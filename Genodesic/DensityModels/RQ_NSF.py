import torch
import torch.nn as nn
from .base_class import BaseDensityModel
import FrEIA.framework as Ff
from FrEIA.modules import RationalQuadraticSpline
from typing import Optional


def subnet_fc(dims_in, dims_out):
    return nn.Sequential(
        nn.Linear(dims_in, 512),
        nn.SELU(),
        nn.Linear(512, 512),
        nn.SELU(),
        nn.Linear(512, 512),
        nn.SELU(),
        nn.Linear(512, dims_out)
    )

def build_rq_nsf_model(dim=13, n_blocks=3, bins=15):
    flow = Ff.SequenceINN(dim)
    for _ in range(n_blocks):
        flow.append(
            RationalQuadraticSpline,
            subnet_constructor=subnet_fc,
            bins=bins
        )

    return flow


# --------------------------------------------
# Drop-in RQ-NSF model to replace OptimalFlowModel
# --------------------------------------------
class RQNSFModel(BaseDensityModel, nn.Module):
    def __init__(self, model: Optional[nn.Module] = None, dim=13, n_blocks=6, bins=8, device=None):
        super().__init__()
        self.dim = dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model is not None:
            # Use the provided model
            self.model = model.to(self.device)
        else:
            # Build a new flow model if not provided
            self.model = build_rq_nsf_model(dim=dim, n_blocks=n_blocks, bins=bins).to(self.device)
        
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
        
        # Check if the checkpoint is for the correct model type (optional but good practice)
        if checkpoint.get('model_type') != 'rqnsf':
            raise ValueError(f"Checkpoint is for model type {checkpoint.get('model_type')}, not 'rqnsf'.")

        params = checkpoint['hyperparameters']
        
        # Rebuild the underlying FrEIA model
        network = build_rq_nsf_model(dim=params['dim'])
        network.load_state_dict(checkpoint['model_state_dict'])
        
        # Create an instance of the final wrapper class
        # `cls` here refers to RQNSFModel
        final_model = cls(model=network, dim=params['dim'], device=device)
        final_model.eval()
        return final_model