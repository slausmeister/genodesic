import torch
import numpy as np
import k3d
from matplotlib import cm
from typing import Optional, List, Union
import matplotlib.pyplot as plt

# Attempt to import cuML UMAP, with a fallback to the standard UMAP
try:
    from cuml.manifold import UMAP as cumlUMAP
    HAS_CUML = True
    print("cuML found. Using GPU for UMAP acceleration.")
except ImportError:
    import umap
    HAS_CUML = False
    print("cuML not found. Falling back to CPU-based umap-learn for UMAP.")


def UMAP3D(
    latent_reps: np.ndarray,
    paths: Optional[List[Union[torch.Tensor, np.ndarray]]] = None,
    color_by_timepoints: Optional[np.ndarray] = None,
    color_by_labels: Optional[np.ndarray] = None,
    title: str = "Latent Space UMAP",
    point_size: float = 0.03,
    **umap_kwargs
):
    """
    Computes a 3D UMAP and visualizes it with k3d, optionally plotting paths.
    Prioritizes GPU-accelerated UMAP via cuML if available.
    """
    print("--- Starting 3D Visualization ---")

    # --- 1. UMAP Fitting ---
    print("Fitting UMAP model...")
    if 'n_components' not in umap_kwargs:
        umap_kwargs['n_components'] = 3
    if 'random_state' not in umap_kwargs:
        umap_kwargs['random_state'] = 42

    if HAS_CUML:
        umap_model = cumlUMAP(**umap_kwargs)
    else:
        umap_model = umap.UMAP(**umap_kwargs)

    embedding = umap_model.fit_transform(latent_reps)

    # --- 2. Plotting Setup ---
    print("Creating k3d plot...")
    plot = k3d.plot(name=title, grid_visible=False, camera_auto_fit=True)

    # --- 3. Color & Attribute Handling (unchanged) ---
    k3d_point_kwargs = {}

    if color_by_labels is not None:
        print("Coloring by discrete labels and creating legend.")
        unique_labels = np.unique(color_by_labels)
        num_labels = len(unique_labels)
        label_map = {label: i for i, label in enumerate(unique_labels)}
        int_labels = np.array([label_map[l] for l in color_by_labels])
        hues = np.linspace(0, 0.9, num_labels, endpoint=True)
        colors_rgb_list = (cm.hsv(hues)[:, :3] * 255).astype(np.uint8)
        k3d_colors = np.array([(0xFF << 24) | (r << 16) | (g << 8) | b for r, g, b in colors_rgb_list], dtype=np.uint32)
        k3d_point_kwargs['colors'] = k3d_colors[int_labels]
        for i, label in enumerate(unique_labels):
            color_int = k3d_colors[i]
            plot += k3d.text2d(str(label), position=[0.02, 0.9 - i * 0.05], color=int(color_int), size=1, label_box=False)

    elif color_by_timepoints is not None:
        print("Coloring by continuous timepoints and adding a color bar.")
        cmap = cm.get_cmap('viridis')
        cmap_samples = cmap(np.linspace(0, 1, 256))[:, :3]
        k3d_viridis = []
        for i, color in enumerate(cmap_samples):
            k3d_viridis.extend([i / 255.0, color[0], color[1], color[2]])
        k3d_point_kwargs['attribute'] = color_by_timepoints.astype(np.float32)
        k3d_point_kwargs['color_map'] = np.array(k3d_viridis, dtype=np.float32)
        k3d_point_kwargs['color_range'] = [color_by_timepoints.min(), color_by_timepoints.max()]
        plot.colorbar_enabled = True

    else:
        k3d_point_kwargs['color'] = 0x808080

    # --- 4. Plotting Geometries ---
    plot += k3d.points(
        positions=embedding.astype(np.float32),
        point_size=point_size,
        name='Background Data',
        **k3d_point_kwargs
    )
    if paths:
        print(f"Transforming and plotting {len(paths)} paths...")
        path_colors = [0xFF0000, 0x00FF00, 0x0000FF, 0xFFFF00, 0xFF00FF, 0x00FFFF]

        for i, path in enumerate(paths):
            def unpack_and_convert(item):
                if isinstance(item, torch.Tensor):
                    # Base case 1: The item is a tensor, convert it
                    return item.detach().cpu().numpy()
                elif isinstance(item, (list, tuple)):
                    # Recursive step: The item is a container, so process its contents
                    # and concatenate the results into a single array.
                    return np.concatenate([unpack_and_convert(sub_item) for sub_item in item], axis=0)
                else:
                    # Base case 2: Assume it's already a numpy array or something convertible
                    return np.asarray(item)

            # Process the path, whatever its structure
            path_np = unpack_and_convert(path).astype(np.float32)

            # Ensure the final array has the correct data type
            path_np = path_np.astype(np.float32)
            # --- safety checks (unchanged) ---
            if path_np.ndim == 1:
                path_np = path_np.reshape(1, -1)
            if path_np.ndim != 2:
                raise ValueError(f"Path {i} has shape {path_np.shape}; expected (L, D).")
            if path_np.shape[1] != latent_reps.shape[1]:
                raise ValueError(
                    f"Dim mismatch: path {i} has D={path_np.shape[1]}, "
                    f"latent_reps has D={latent_reps.shape[1]}"
                )

            # Project & draw
            path_emb = umap_model.transform(path_np).astype(np.float32)
            color = path_colors[i % len(path_colors)]

            plot += k3d.line(path_emb, color=color, width=0.015, name=f"Path {i+1}")
            plot += k3d.points(path_emb, color=color, point_size=point_size * 5, name=f"Path {i+1} Nodes")

    # --- 5. Display Plot ---
    print("--- Visualization Complete ---")
    return plot



def UMAP2D(
    latent_reps: np.ndarray,
    paths: Optional[List[Union[torch.Tensor, np.ndarray]]] = None,
    color_by_timepoints: Optional[np.ndarray] = None,
    color_by_labels: Optional[np.ndarray] = None,
    title: str = "Latent Space UMAP",
    point_size: float = 5.0,
    **umap_kwargs
):
    """
    Computes and visualizes a 2D UMAP with Matplotlib.
    Returns a single Figure object, acting as a drop-in replacement for UMAP3D.
    """
    print("--- Starting 2D Visualization ---")

    # --- 1. UMAP Fitting ---
    print("Fitting UMAP model...")
    umap_kwargs['n_components'] = 2
    if 'random_state' not in umap_kwargs:
        umap_kwargs['random_state'] = 42
    if 'ax' in umap_kwargs:
        umap_kwargs.pop('ax')
    if HAS_CUML:
        umap_model = cumlUMAP(**umap_kwargs)
    else:
        umap_model = umap.UMAP(**umap_kwargs)
    embedding = umap_model.fit_transform(latent_reps)

    # --- 2. Plotting Setup ---
    print("Creating matplotlib plot...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # --- 3. Color & Attribute Handling ---
    scatter_kwargs = {'s': point_size, 'alpha': 0.7, 'edgecolor': 'none'}
    if color_by_labels is not None:
        unique_labels = np.unique(color_by_labels)
        cmap = cm.get_cmap('hsv', len(unique_labels))
        for i, label in enumerate(unique_labels):
            idx = (color_by_labels == label)
            ax.scatter(embedding[idx, 0], embedding[idx, 1], color=cmap(i), label=str(label), **scatter_kwargs)
        ax.legend(title="Labels")
    elif color_by_timepoints is not None:
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=color_by_timepoints, cmap='viridis', **scatter_kwargs)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Timepoints')
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], c='gray', **scatter_kwargs)

    # --- 4. Path Plotting (Restored Logic) ---
    if paths:
        print(f"Transforming and plotting {len(paths)} paths...")
        path_colors = plt.get_cmap('tab10').colors

        def unpack_and_convert(item):
            if isinstance(item, torch.Tensor): return item.detach().cpu().numpy()
            if isinstance(item, (list, tuple)): return np.concatenate([unpack_and_convert(sub_item) for sub_item in item], axis=0)
            return np.asarray(item)

        for i, path in enumerate(paths):
            path_np = unpack_and_convert(path).astype(np.float32)
            if path_np.shape[1] != latent_reps.shape[1]:
                raise ValueError(f"Path {i} has D={path_np.shape[1]}, but latent space has D={latent_reps.shape[1]}")
            
            path_emb = umap_model.transform(path_np)
            color = path_colors[i % len(path_colors)]
            
            ax.plot(path_emb[:, 0], path_emb[:, 1], color=color, linewidth=2.5, label=f'Path {i+1}', zorder=10)
            ax.scatter(path_emb[:, 0], path_emb[:, 1], color=color, s=point_size*10, edgecolor='black', zorder=11)
        
        ax.legend()

    # --- 5. Final Adjustments ---
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.close(fig)

    print("--- Visualization Complete ---")
    return fig
