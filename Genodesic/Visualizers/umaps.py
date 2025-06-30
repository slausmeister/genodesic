import torch
import numpy as np
import k3d
from matplotlib import cm
from typing import Optional, List, Union

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

    # --- 3. Color & Attribute Handling ---
    k3d_point_kwargs = {}

    if color_by_labels is not None:
        # This section for discrete labels remains correct
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
        
        # --- MANUALLY CREATE THE COLORMAP ARRAY ---
        # 1. Get the colormap from matplotlib
        cmap = cm.get_cmap('viridis')
        # 2. Sample the colormap at 256 points
        cmap_samples = cmap(np.linspace(0, 1, 256))[:, :3] # Get RGB, drop Alpha
        # 3. Create the K3D format: a flat array of [pos, R, G, B, pos, R, G, B, ...]
        k3d_viridis = []
        for i, color in enumerate(cmap_samples):
            k3d_viridis.extend([i / 255.0, color[0], color[1], color[2]])
        # --- END OF MANUAL CREATION ---

        k3d_point_kwargs['attribute'] = color_by_timepoints.astype(np.float32)
        k3d_point_kwargs['color_map'] = np.array(k3d_viridis, dtype=np.float32) # Assign the array
        k3d_point_kwargs['color_range'] = [color_by_timepoints.min(), color_by_timepoints.max()]

        # Enable the colorbar on the plot
        plot.colorbar_enabled = True

    else:
        k3d_point_kwargs['color'] = 0x808080 # Grey

    # --- 4. Plotting Geometries ---
    plot += k3d.points(
        positions=embedding.astype(np.float32),
        point_size=point_size,
        name='Background Data',
        **k3d_point_kwargs
    )

    # Plot paths if provided
    if paths:
        # (Path plotting code remains the same)
        print(f"Transforming and plotting {len(paths)} paths...")
        path_colors = [0xFF0000, 0x00FF00, 0x0000FF, 0xFFFF00, 0xFF00FF, 0x00FFFF]
        for i, path in enumerate(paths):
            path_np = path.cpu().numpy() if isinstance(path, torch.Tensor) else path
            path_embedding = umap_model.transform(path_np).astype(np.float32)
            color = path_colors[i % len(path_colors)]
            plot += k3d.line(path_embedding, color=color, width=0.015, name=f'Path {i+1}')
            plot += k3d.points(path_embedding, color=color, point_size=point_size * 5, name=f'Path {i+1} Nodes')

    # --- 5. Display Plot ---
    print("--- Visualization Complete ---")
    return plot