import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple

from ..LatentCell.model import LatentCell

def load_autoencoder_and_metadata(
    ae_checkpoint_path: str, 
    hvg_tensor_path: str, 
    device: str
) -> Tuple[LatentCell, List[str]]:
    """
    Loads a trained LatentCell autoencoder and its corresponding gene names.

    This function improves robustness by loading the highly variable gene (HVG) names
    directly from the tensor file that was used to train the autoencoder, ensuring
    perfect alignment between the model's output and the gene labels.

    Args:
        ae_checkpoint_path (str): Path to the autoencoder .pt checkpoint file.
        hvg_tensor_path (str): Path to the HVG tensor .pt file, which contains the gene names.
        device (str): The device to load the model onto ('cuda' or 'cpu').

    Returns:
        A tuple containing:
        - The loaded and evaluated LatentCell model.
        - A list of HVG names corresponding to the model's output dimension.
    """
    print("--- Loading AutoEncoder and Gene Metadata ---")
    # Load the autoencoder checkpoint
    ckpt = torch.load(ae_checkpoint_path, map_location=device)
    
    # Load the data file to get the exact gene list
    hvg_data = torch.load(hvg_tensor_path, map_location="cpu", weights_only=False)
    hvg_names = hvg_data['gene_names'] #
    
    # Verify consistency
    input_dim_from_ckpt = ckpt.get("input_dim")
    if input_dim_from_ckpt != len(hvg_names):
        print(f"WARNING: AE checkpoint input_dim ({input_dim_from_ckpt}) differs from metadata HVG count ({len(hvg_names)}).")

    # Build the model using parameters from the checkpoint
    model = LatentCell(
        data_dim=ckpt["input_dim"], 
        latent_dims=ckpt["latent_dims"], 
        variational=ckpt["variational"]
    ).to(device) #
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Successfully loaded AutoEncoder on {device} with {len(hvg_names)} genes.")
    
    return model, hvg_names


def decode_latent_trajectory(
    autoencoder_model: LatentCell, 
    latent_path: torch.Tensor, 
    device: str
) -> np.ndarray:
    """
    Decodes a latent space trajectory back to gene expression space.

    Args:
        autoencoder_model (LatentCell): The trained autoencoder model.
        latent_path (torch.Tensor): A tensor of shape (num_points, latent_dim) 
                                    representing the trajectory.
        device (str): The device the model is on.

    Returns:
        np.ndarray: The decoded gene expression curve of shape (num_points, num_genes).
    """
    autoencoder_model.eval()
    with torch.no_grad():
        z = latent_path.to(device)
        # The decoder part of the model does not include the final UMI scaling
        # which is what we want for visualizing relative expression.
        recon = autoencoder_model.decoder(z) 
    return recon.cpu().numpy()


def find_dynamic_genes(
    expression_curve: np.ndarray, 
    gene_names: List[str], 
    num_genes: int = 20
) -> Tuple[np.ndarray, List[str]]:
    """
    Identifies and orders the most dynamic genes along a trajectory.

    This function finds the genes with the highest variance in expression along the
    path and then orders them by their time of peak expression.

    Args:
        expression_curve (np.ndarray): Decoded gene expression trajectory.
        gene_names (List[str]): List of gene names corresponding to the expression data.
        num_genes (int): The number of top dynamic genes to select.

    Returns:
        A tuple containing:
        - The normalized and ordered expression data for the top genes.
        - The corresponding ordered list of gene names.
    """
    if expression_curve.size == 0 or not gene_names:
        raise ValueError("Expression data or gene names are empty.")
    
    num_total_genes = expression_curve.shape[1]
    
    # Find top N most variable genes
    variances = expression_curve.var(axis=0)
    genes_to_plot = min(num_genes, num_total_genes)
    top_hvg_indices = np.argsort(variances)[-genes_to_plot:]

    # Normalize the subset for visualization
    sub_expr_data = expression_curve[:, top_hvg_indices]
    mins = sub_expr_data.min(axis=0, keepdims=True)
    maxs = sub_expr_data.max(axis=0, keepdims=True)
    sub_expr_normalized = (sub_expr_data - mins) / (maxs - mins + 1e-8)

    # Order genes by their peak expression time
    peaks_in_pseudotime = np.argmax(sub_expr_normalized, axis=0)
    order_by_peak = np.argsort(peaks_in_pseudotime)

    # Apply ordering to both data and gene names
    final_ordered_data = sub_expr_normalized[:, order_by_peak]
    final_ordered_names = [gene_names[i] for i in top_hvg_indices[order_by_peak]]
    
    return final_ordered_data, final_ordered_names


def plot_dynamic_genes_heatmap(
    expression_data: np.ndarray, 
    gene_names: List[str], 
    figsize=(8, 20), 
    cmap="viridis"
):
    """
    Plots a heatmap of gene expression along a trajectory.

    Args:
        expression_data (np.ndarray): Normalized and ordered expression data from find_dynamic_genes.
        gene_names (List[str]): Ordered gene names corresponding to the data.
        figsize (tuple): Figure size for the plot.
        cmap (str): Colormap for the heatmap.
    """
    plt.figure(figsize=figsize)
    sns.heatmap(
        expression_data.T, 
        cmap=cmap,
        yticklabels=gene_names,
        xticklabels=False,
        cbar_kws={"label": "Normalized Expression Along Trajectory"}
    )
    plt.xlabel("Trajectory Progression")
    plt.ylabel("Gene")
    plt.title(f"Top {len(gene_names)} Most Dynamic Genes Along Geodesic Path")
    plt.tight_layout()
    plt.show()