import scanpy as sc
import pandas as pd
import numpy as np
import os
import re
import argparse
from tqdm import tqdm
import torch
import warnings

def setup_arg_parser():
    """
    Sets up the command-line argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Load, combine, and process Schiebinger dataset from 10x H5 files."
    )
    parser.add_argument(
        "--data_dir",
        default="../Data/Schiebinger/",
        help="Directory containing the raw .h5 files.",
    )
    parser.add_argument(
        "--output_dir",
        default="./HVGs/",
        help="Directory where the final .h5ad file will be saved.",
    )
    parser.add_argument(
        "--output_file",
        default=".HVGs.pt",
        help="Directory where the final .h5ad file will be saved.",  
    )
    parser.add_argument(
        "--trunk",
        type=str,
        default="2i",
        choices=["serum", "2i", "both"],
        help="Specify the developmental trunk to analyze: 'serum', '2i', or 'both'.",
    )
    parser.add_argument(
        "--n_hvg",
        type=int,
        default=2000,
        help="Number of highly variable genes to select.",
    )
    parser.add_argument(
        "--min_genes",
        type=int,
        default=200,
        help="Minimum number of genes expressed required for a cell to be kept.",
    )
    parser.add_argument(
        "--min_cells",
        type=int,
        default=3,
        help="Minimum number of cells a gene must be expressed in to be kept.",
    )
    return parser

def parse_filename(filename):
    """
    Parses metadata (day, condition, replicate) from a 10x filename.
    Example: 'GSM3195688_D8.5_serum_C1_gene_bc_mat.h5'
    Returns: {'day': 8.5, 'condition': 'serum', 'replicate': 'C1'}
    """
    # This pattern captures Day (D# or DiPSC), Condition (serum, 2i, Dox), and Replicate (C#)
    match = re.search(r"_(D(\d+\.?\d*)|DiPSC)_(serum|2i|Dox)_C(\d+)_", filename)
    
    if match:
        day_str = match.group(1)      # Full day string: e.g., 'D8.5' or 'DiPSC'
        condition = match.group(3)    # Condition: e.g., 'serum'
        replicate_num = match.group(4) # Replicate number: e.g., '1'
        
        if day_str == "DiPSC":
            # Assign a late pseudotime day for iPSCs for ordering
            day = 19.0 
        else:
            # Remove the 'D' prefix to get the number
            day = float(day_str[1:])
            
        replicate = f"C{replicate_num}"
        
        return {"day": day, "condition": condition, "replicate": replicate}
        
    # Ignore other files like the GDF9 experiments which have a different naming scheme
    if "GDF9" in filename:
        return None
        
    print(f"Warning: Could not parse metadata from filename: {filename}")
    return None

def save_final_tensors(final_df, output_dir, args, filename=None):
    """
    Saves the final processed data as PyTorch tensors.
    """
    print("\n--- Saving Final Tensors ---")
    
    # 1. Separate metadata from the count matrix
    metadata_cols = ['day', 'condition', 'path']
    metadata_df = final_df[metadata_cols]
    counts_df = final_df.drop(columns=metadata_cols)
    
    # 2. Convert to PyTorch tensors
    counts_tensor = torch.tensor(counts_df.values, dtype=torch.float32)
    timepoints_tensor = torch.tensor(metadata_df['day'].values, dtype=torch.float32).unsqueeze(1) # Reshape to [n_cells, 1]

    # 3. Prepare data bundle for saving
    data_to_save = {
        'counts': counts_tensor,
        'timepoints': timepoints_tensor,
        'cell_ids': counts_df.index.tolist(),
        'gene_names': counts_df.columns.tolist(),
        'metadata_df': metadata_df # Also save the pandas metadata for convenience
    }
    
    # 4. Save to file
    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        filename = f"schiebinger_hvg_tensor_trunk-{args.trunk}_{args.n_hvg}hvg.pt"
    
    output_path = os.path.join(output_dir, filename)
    
    torch.save(data_to_save, output_path)
    print(f"Successfully saved data to: {output_path}")
    print(f"Saved {counts_tensor.shape[0]} cells and {counts_tensor.shape[1]} genes.")


def main(args):
    """
    Main execution function.
    """
    # 1. Discover H5 files and prepare for loading
    h5_files = [f for f in os.listdir(args.data_dir) if f.endswith(".h5") and f.startswith("GSM")]
    if not h5_files:
        print(f"Error: No .h5 files found in {args.data_dir}")
        return

    print(f"Found {len(h5_files)} H5 files to process.")
    
    adatas_list = []
    # 2. Loop through files, load data, and add metadata
    for filename in tqdm(h5_files, desc="Loading H5 files"):
        file_path = os.path.join(args.data_dir, filename)
        
        # Extract metadata from the filename itself
        meta = parse_filename(filename)
        if meta is None:
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                adata_sample = sc.read_10x_h5(file_path)
            
            # For this data, gene symbols are loaded directly into the .var index.
            # We just need to make them unique as the warning suggests.
            adata_sample.var_names_make_unique()
            
            # Add the parsed metadata to the observation (cell) dataframe
            adata_sample.obs["sample"] = filename
            adata_sample.obs["day"] = meta["day"]
            adata_sample.obs["condition"] = meta["condition"]
            adata_sample.obs["replicate"] = meta["replicate"]
            adatas_list.append(adata_sample)
        except Exception as e:
            print(f"Error reading file {filename}: {e}")

    # 3. Concatenate all AnnData objects into one
    print("Concatenating all samples into a single AnnData object...")
    # The `index_unique` argument is critical to avoid cell barcode collisions
    adata = sc.concat(adatas_list, label="sample_id", index_unique="-")
    print(f"Initial combined data shape: {adata.n_obs} cells × {adata.n_vars} genes")

    # Keep a copy of the raw counts before normalization
    adata.layers["counts"] = adata.X.copy()

    # 4. Filter data based on the selected trunk (STRETCH GOAL)
    print(f"Filtering for trunk: '{args.trunk}'")
    
    # The experiment branches after day 8
    branch_point_day = 8.0
    
    if args.trunk == "serum":
        # Keep initial Dox cells and all subsequent serum cells
        serum_trunk_mask = (adata.obs['condition'] == 'Dox') | (adata.obs['condition'] == 'serum')
        adata = adata[serum_trunk_mask, :].copy()
        adata.obs['path'] = 'Serum'
        print(f"Filtered down to {adata.n_obs} cells for the Serum trunk.")
    
    elif args.trunk == "2i":
        # Keep initial Dox cells and all subsequent 2i cells
        i2_trunk_mask = (adata.obs['condition'] == 'Dox') | (adata.obs['condition'] == '2i')
        adata = adata[i2_trunk_mask, :].copy()
        adata.obs['path'] = '2i'
        print(f"Filtered down to {adata.n_obs} cells for the 2i trunk.")
        
    elif args.trunk == "both":
        # Keep all cells and create a 'path' annotation for coloring plots
        adata.obs['path'] = 'Dox' # Default path
        adata.obs.loc[(adata.obs['day'] > branch_point_day) & (adata.obs['condition'] == 'serum'), 'path'] = 'Serum'
        adata.obs.loc[(adata.obs['day'] > branch_point_day) & (adata.obs['condition'] == '2i'), 'path'] = '2i'
        # Handle iPSC endpoints
        adata.obs.loc[(adata.obs['day'] == 19.0) & (adata.obs['condition'] == 'serum'), 'path'] = 'Serum_iPSC'
        adata.obs.loc[(adata.obs['day'] == 19.0) & (adata.obs['condition'] == '2i'), 'path'] = '2i_iPSC'
        print("Kept all cells for 'both' trunks analysis. Added 'path' annotation.")

    # 5. Standard scRNA-seq preprocessing
    print("Starting preprocessing and quality control...")
    # Basic filtering of genes and cells
    sc.pp.filter_cells(adata, min_genes=args.min_genes)
    sc.pp.filter_genes(adata, min_cells=args.min_cells)
    print(f"Shape after basic gene/cell filtering: {adata.shape}")

    # Calculate mitochondrial gene percentage for QC
    adata.var['mt'] = adata.var_names.str.startswith('mt-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    
    # Add additional QC filtering to prevent errors with infinity values
    print("Applying additional QC filters for mitochondrial content and high UMI counts...")
    
    # Filter cells with high mitochondrial content
    adata = adata[adata.obs.pct_counts_mt < 20, :].copy()
    print(f"Shape after mitochondrial filtering: {adata.shape}")

    # Filter cells with outlier-high UMI counts (potential doublets or artifacts)
    upper_limit = np.quantile(adata.obs.total_counts.to_numpy(), 0.99)
    adata = adata[adata.obs.total_counts < upper_limit, :].copy()
    print(f"Shape after UMI count filtering: {adata.shape}")

    # Normalize and log-transform the data for HVG calculation
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Identify Highly Variable Genes (HVGs)
    print(f"Identifying top {args.n_hvg} highly variable genes...")
    # By default, this uses the log-normalized data in adata.X, which is what we want.
    # The previous error was caused by incorrectly telling it to use the raw 'counts' layer.
    sc.pp.highly_variable_genes(adata, n_top_genes=args.n_hvg)
    
    # 6. Assemble the raw count matrix for HVGs
    print("\nAssembling final raw count matrix for HVGs...")
    
    # Filter the anndata object to only the HVGs
    # Note: we are filtering the *original* adata object which still has the raw counts layer
    adata_hvg = adata[:, adata.var.highly_variable].copy()
    
    # Create a pandas DataFrame from the raw counts stored in the 'counts' layer
    raw_counts_df = pd.DataFrame(
        adata_hvg.layers['counts'].toarray(),
        index=adata_hvg.obs.index,
        columns=adata_hvg.var.index
    )
    
    # Combine the raw counts with the cell metadata
    final_df = pd.concat([adata_hvg.obs[['day', 'condition', 'path']], raw_counts_df], axis=1)
    
    # 7. Save the final data to disk
    save_final_tensors(final_df, args.output_dir, args, filename=args.output_file)
            
    print("\n--- Processing Complete ---")
    print(f"Final matrix constructed for {final_df.shape[0]} cells and {raw_counts_df.shape[1]} HVGs.")


if __name__ == "__main__":
    parser = setup_arg_parser()
    args = parser.parse_args()
    main(args)
