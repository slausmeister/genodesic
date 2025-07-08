#!/usr/bin/env python
import scanpy as sc
import pandas as pd
import numpy as np
import os
import re
import argparse
from tqdm import tqdm
import torch
import warnings
import matplotlib.pyplot as plt
from typing import Dict, Any
from Genodesic.Utils.config_loader import load_config

# ===================================================================== #
# 1. Command-Line Interface Definition
# ===================================================================== #
def setup_arg_parser():
    """
    Sets up the command-line argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Load, combine, and process Schiebinger dataset from 10x H5 files."
    )
    # --- Required inputs ---
    parser.add_argument("--data_dir", required=True, help="Directory containing the raw .h5 files.")
    
    # --- Optional I/O (will override YAML if provided) ---
    parser.add_argument("--output_dir", default=None, help="Directory where the final .pt file will be saved. Overrides YAML.")
    parser.add_argument("--output_file", default=None, help="Name for the output .pt file. Overrides YAML.")
    
    # --- Optional Parameters (will override YAML if provided) ---
    parser.add_argument("--trunk", type=str, default=None, choices=["serum", "2i", "both"], help="Specify the developmental trunk. Overrides YAML.")
    parser.add_argument("--n_hvg", type=int, default=None, help="Number of highly variable genes. Overrides YAML.")
    parser.add_argument("--min_counts", type=int, default=None, help="Min genes expressed for a cell. Overrides YAML.")
    parser.add_argument("--max_counts", type=int, default=None, help="Max genes expressed for a cell. Overrides YAML.")
    parser.add_argument("--min_cells", type=int, default=None, help="Min cells a gene must be in. Overrides YAML.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for more detailed logs and diagnostic plots.")
    return parser


# ===================================================================== #
# 2. Helper Functions
# ===================================================================== #
def parse_filename(filename):
    """
    Parses metadata (day, condition, replicate) from a 10x filename.
    """
    match = re.search(r"_(D(\d+\.?\d*)|DiPSC)_(serum|2i|Dox)_C(\d+)_", filename)
    if match:
        day_str, condition, replicate_num = match.group(1), match.group(3), match.group(4)
        day = 19.0 if day_str == "DiPSC" else float(day_str[1:])
        replicate = f"C{replicate_num}"
        return {"day": day, "condition": condition, "replicate": replicate}
    if "GDF9" in filename:
        return None
    print(f"Warning: Could not parse metadata from filename: {filename}")
    return None

def save_final_tensors(final_df, output_dir, output_file, trunk, n_hvg):
    """
    Saves the final processed data as PyTorch tensors.
    This version is self-contained and does not depend on an 'args' object.
    """
    print("\n--- Saving Final Tensors ---")
    metadata_cols = ['day', 'condition', 'path']
    metadata_df = final_df[metadata_cols]
    counts_df = final_df.drop(columns=metadata_cols)

    data_to_save = {
        'counts': torch.tensor(counts_df.values, dtype=torch.float32),
        'timepoints': torch.tensor(metadata_df['day'].values, dtype=torch.float32).unsqueeze(1),
        'cell_ids': counts_df.index.tolist(),
        'gene_names': counts_df.columns.tolist(),
        'metadata_df': metadata_df
    }

    os.makedirs(output_dir, exist_ok=True)
    if output_file is None:
        # Generate a default filename if one isn't provided
        output_file = f"schiebinger_hvg_tensor_trunk-{trunk}_{n_hvg}hvg.pt"

    output_path = os.path.join(output_dir, output_file)
    torch.save(data_to_save, output_path)
    print(f"Successfully saved data to: {output_path}")
    print(f"Saved {data_to_save['counts'].shape[0]} cells and {data_to_save['counts'].shape[1]} genes.")


# ===================================================================== #
# 3. Core Logic Function 
# ===================================================================== #
def run_hvg_extraction(config: Dict[str, Any], data_dir: str, debug: bool = True):
    """
    Loads raw 10x data, performs QC, filtering, and HVG selection based
    on a provided configuration dictionary.
    """
    # Pull all parameters from the config object
    cfg = config['hvg_extraction']
    output_dir = cfg['output_dir']
    output_file = cfg['output_file']
    trunk = cfg['trunk']
    n_hvg = cfg['n_hvg']
    min_counts = cfg['min_counts']
    max_counts = cfg['max_counts']
    min_cells = cfg['min_cells']

    # Defensive check: ensure the output filename is specified
    if not output_file:
        raise ValueError("Configuration must specify 'hvg_extraction.output_file'")

    fig_to_return = None
    # 1. Discover H5 files and prepare for loading
    h5_files = [f for f in os.listdir(data_dir) if f.endswith(".h5") and f.startswith("GSM")]
    if not h5_files:
        print(f"Error: No .h5 files found in {data_dir}"); return

    print(f"Found {len(h5_files)} H5 files to process.")
    adatas_list = []

    # 2. Loop through files, load data, and add metadata
    for filename in tqdm(h5_files, desc="Loading H5 files"):
        meta = parse_filename(filename)
        if meta is None: continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                adata_sample = sc.read_10x_h5(os.path.join(data_dir, filename))

            adata_sample.var_names_make_unique()

            adata_sample.obs["sample"] = filename
            adata_sample.obs["day"] = meta["day"]
            adata_sample.obs["condition"] = meta["condition"]
            adata_sample.obs["replicate"] = meta["replicate"]
            adatas_list.append(adata_sample)
        except Exception as e:
            print(f"Error reading file {filename}: {e}")

    # 3. Concatenate all AnnData objects into one
    print("Concatenating all samples into a single AnnData object...")
    adata = sc.concat(adatas_list, label="sample_id", index_unique="-")
    print(f"Initial combined data shape: {adata.n_obs} cells × {adata.n_vars} genes")
    adata.layers["counts"] = adata.X.copy()

    # 4. Filter data based on the selected trunk
    branch_point_day = 8.0
    if trunk == "serum":
        mask = (adata.obs['condition'] == 'Dox') | (adata.obs['condition'] == 'serum')
        adata = adata[mask, :].copy()
        adata.obs['path'] = 'Serum'
        print(f"Filtered down to {adata.n_obs} cells for the Serum trunk.")
    elif trunk == "2i":
        mask = (adata.obs['condition'] == 'Dox') | (adata.obs['condition'] == '2i')
        adata = adata[mask, :].copy()
        adata.obs['path'] = '2i'
        print(f"Filtered down to {adata.n_obs} cells for the 2i trunk.")
    elif trunk == "both":
        adata.obs['path'] = 'Dox'
        adata.obs.loc[(adata.obs['day'] > branch_point_day) & (adata.obs['condition'] == 'serum'), 'path'] = 'Serum'
        adata.obs.loc[(adata.obs['day'] > branch_point_day) & (adata.obs['condition'] == '2i'), 'path'] = '2i'
        adata.obs.loc[(adata.obs['day'] == 19.0) & (adata.obs['condition'] == 'serum'), 'path'] = 'Serum_iPSC'
        adata.obs.loc[(adata.obs['day'] == 19.0) & (adata.obs['condition'] == '2i'), 'path'] = '2i_iPSC'
        print("Kept all cells for 'both' trunks analysis. Added 'path' annotation.")

    # =================================================================== #
    # 5. Standard scRNA-seq preprocessing with detailed logging
    # =================================================================== #
    print("\n--- Starting Preprocessing and Quality Control ---")
    n_cells_initial = adata.n_obs
    print(f"Starting with {n_cells_initial} cells.")

    # --- Gene & Count Filtering ---
    adata.var['mt'] = adata.var_names.str.startswith('mt-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True, log1p=False)

    if debug:
        print("\n[Debug] Generating UMI count histogram...")
        umis_per_cell = adata.obs['total_counts']
        umis_log = umis_per_cell[umis_per_cell > 0]
        log_min, log_max = np.log10(umis_log.min()), np.log10(umis_log.max())
        log_bins = np.logspace(log_min, log_max, 75)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        counts, bins, patches = ax.hist(umis_log, bins=log_bins, edgecolor='black', linewidth=0.5)

        for patch, left_edge in zip(patches, bins[:-1]):
            if min_counts <= left_edge <= max_counts:
                patch.set_facecolor("C0")
            else:
                patch.set_facecolor("lightgrey")

        ax.set_xscale("log")
        ax.axvline(min_counts, color='red', linestyle='--', linewidth=1.5, label=f"Min Counts ({min_counts})")
        ax.axvline(max_counts, color='red', linestyle='--', linewidth=1.5, label=f"Max Counts ({max_counts})")
        
        ax.set_xlabel("Total UMIs per cell (log scale)")
        ax.set_ylabel("Number of cells")
        ax.set_title("UMI Distribution and Filtering Thresholds")
        ax.legend()
        ax.grid(False)
        plt.tight_layout()
        fig_to_return = fig #

    # Filter genes
    n_genes_before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print(f"Filtered {n_genes_before - adata.n_vars} genes expressed in < {min_cells} cells. Remaining: {adata.n_vars}.")
    
    # Filter cells by min/max counts
    n_cells_before = adata.n_obs
    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_cells(adata, max_counts=max_counts)
    print(f"Filtered {n_cells_before - adata.n_obs} cells by total counts. Remaining: {adata.n_obs}.")

    # Filter by mitochondrial content
    n_cells_before = adata.n_obs
    adata = adata[adata.obs.pct_counts_mt < 20, :].copy()
    print(f"Filtered {n_cells_before - adata.n_obs} cells with >20% mitochondrial counts. Remaining: {adata.n_obs}.")
    
    print("\n--- QC Summary ---")
    print(f"Finished QC. Kept {adata.n_obs} cells out of {n_cells_initial} ({adata.n_obs / n_cells_initial:.2%}).")
    
    # --- Normalization and HVG Selection ---
    print("\nNormalizing and finding variable genes...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg)
    if debug: print(f"[Debug] Found {adata.var.highly_variable.sum()} highly variable genes.")

    # 6. Assemble the raw count matrix for HVGs
    print("\nAssembling final raw count matrix for HVGs...")
    adata_hvg = adata[:, adata.var.highly_variable].copy()
    raw_counts_df = pd.DataFrame(
        adata_hvg.layers['counts'].toarray(),
        index=adata_hvg.obs.index,
        columns=adata_hvg.var.index
    )
    final_df = pd.concat([adata_hvg.obs[['day', 'condition', 'path']], raw_counts_df], axis=1)

    # 7. Save the final data to disk
    save_final_tensors(final_df, output_dir, output_file, trunk, n_hvg)
    print("\n--- Processing Complete ---")
    return fig_to_return 

# ===================================================================== #
# 4. Command-Line Execution Wrapper
# ===================================================================== #
def main():
    """
    Parses CLI arguments and launches the HVG extraction.
    """
    parser = setup_arg_parser()
    args = parser.parse_args()

    # 1. Prepare overrides from CLI arguments in the correct nested structure
    overrides = {
        "hvg_extraction": {
            "output_dir": args.output_dir,
            "output_file": args.output_file,
            "trunk": args.trunk,
            "n_hvg": args.n_hvg,
            "min_counts": args.min_counts,
            "max_counts": args.max_counts,
            "min_cells": args.min_cells,
        }
    }
    # Filter out None values so they don't overwrite defaults
    overrides['hvg_extraction'] = {k: v for k, v in overrides['hvg_extraction'].items() if v is not None}

    # 2. Load the base config and merge with CLI overrides
    final_config = load_config(
        default_config_path="Config/hvg_extraction.yaml",
        overrides=overrides
    )

    # 3. Call the core worker function
    run_hvg_extraction(
        config=final_config,
        data_dir=args.data_dir,
        debug=args.debug
    )

if __name__ == "__main__":
    main()

