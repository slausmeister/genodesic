from abc import ABC, abstractmethod
import torch
from torch import Tensor

class BaseDensityModel(ABC, torch.nn.Module):
    """
    Common interface for density-estimation / score-matching models.
    Every implementing class must work on (B, D) tensors on the same device
    as `self`.
    """

    @abstractmethod
    def generate_samples(self,
                         num_samples: int | None = None,
                         base_samples: Tensor | None = None,
                         **kwargs) -> Tensor:
        """Draw samples x ~ p_theta."""

    @abstractmethod
    def compute_log_density(self, batch: Tensor, **kwargs) -> Tensor:
        """Return log p_theta(x) for each row in `batch`."""

    @abstractmethod
    def compute_score(self, batch: Tensor, **kwargs) -> Tensor:
        """Return ∇_x log p_theta(x) for each row in `batch`."""

    def reverse(self, batch: Tensor, **kwargs) -> Tensor | None:
        """Map data → base space if the model supports an inverse transform."""
        raise NotImplementedError
