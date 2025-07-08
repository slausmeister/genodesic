from Genodesic.DensityModels import OptimalFlowModel, ScoreSDEModel, RQNSFModel
import torch

def load_density_model_from_checkpoint(checkpoint_path: str, device: str):
    """
    Factory function to load the correct density model from a checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_type = checkpoint.get("model_type")

    print(f"Loading model of type '{model_type}' from {checkpoint_path}...")

    if model_type == "vpsde":
        return ScoreSDEModel.from_checkpoint(checkpoint_path, device=device)
    elif model_type == "rqnsf":
        return RQNSFModel.from_checkpoint(checkpoint_path, device=device)
    elif model_type == "otcfm":
        return OptimalFlowModel.from_checkpoint(checkpoint_path, device=device)
    else:
        raise ValueError(f"Unknown model type in checkpoint: {model_type}")