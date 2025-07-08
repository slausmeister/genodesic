#!/usr/bin/env python
import time
import os
import argparse
import torch
import torch.optim as optim
import yaml
from tqdm import tqdm
from typing import Dict, Any, List
from pathlib import Path

# --------------------------------------------------------------------- #
#  Local imports
# --------------------------------------------------------------------- #
from Genodesic.Dataloaders.CountLoader import create_count_dataloaders
from Genodesic.LatentCell.model        import LatentCell
from Genodesic.Utils.config_loader     import load_config


# ===================================================================== #
# 1. Command-Line Interface Definition
# ===================================================================== #
def setup_arg_parser() -> argparse.ArgumentParser:
    """Define CLI arguments."""
    p = argparse.ArgumentParser(description="Train the LatentCell Autoencoder.")
    # --- Required I/O arguments ---
    p.add_argument("--tensor_file", required=True, help="Path to the .pt file containing processed counts + timepoints")
    # --- Arguments with NO default in YAML (or that we want to force the user to provide for CLI) ---
    p.add_argument("--model_save_path", required=True, help="Path to save the trained model checkpoint")
    p.add_argument("--latent_save_path", default=None, help="Optional path to save the final latent space embedding")

    # --- Arguments that will OVERRIDE autoencoder.yaml ---
    # We set default=None so we can distinguish between a user-provided value and the argparse default.
    p.add_argument("--latent_dims", type=int, nargs='+', default=None, help="Overrides [intermediate_dims, bottleneck_dim]. E.g., --latent_dims 1000 300 100 30")
    p.add_argument("--num_epochs", type=int, default=None, help="Overrides num_epochs in YAML")
    p.add_argument("--batch_size", type=int, default=None, help="Overrides batch_size in YAML")
    p.add_argument("--lr", type=float, default=None, help="Overrides learning_rate in YAML")
    p.add_argument("--val_split", type=float, default=None, help="Overrides validation_split in YAML")
    p.add_argument("--overdispersion", type=float, default=None, help="Overrides overdispersion in YAML")
    p.add_argument("--debug", action="store_true", help="Print extra diagnostics during training")
    return p


# ===================================================================== #
# 2. Helper Functions
# ===================================================================== #
def _grad_global_norm(parameters) -> float:
    """Compute global L2 grad-norm (helper for debugging)."""
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


# ===================================================================== #
# 3. Core Logic Function 
# ===================================================================== #
def run_autoencoder_training(
    hvg_tensor_file: str,
    config: Dict[str, Any],
    debug: bool = False
):
    """
    Trains a LatentCell autoencoder based on a provided configuration.
    """
    ae_cfg = config['autoencoder']

    # Paths are now taken directly from the config, not generated here.
    model_save_path = ae_cfg['model_save_path']
    latent_save_path = ae_cfg.get('latent_save_path') # Use .get for optional path

    # Defensive checks
    if not model_save_path:
        raise ValueError("Config must specify 'autoencoder.model_save_path'")

    # 2) Device and Initial Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_dims = ae_cfg['intermediate_dims'] + [ae_cfg['bottleneck_dim']]
    print(f"[Init] device={device}  α={ae_cfg['overdispersion']}  latent_dims={latent_dims}")

    # 3) Data loading
    loaders = create_count_dataloaders(
        hvg_tensor_file,
        batch_size=ae_cfg['batch_size'],
        validation_split=ae_cfg['validation_split'],
        device=device,
    )
    train_loader, val_loader = (loaders[0], loaders[1]) if ae_cfg['validation_split'] > 0 else (loaders, None)

    # 4) Model, optimizer, and loss setup
    input_dim = next(iter(train_loader))[0].shape[1]
    print(f"[Init] detected HVGs: {input_dim}")
    model = LatentCell(input_dim, latent_dims, variational=ae_cfg['variational']).to(device)
    alpha = torch.tensor([ae_cfg['overdispersion']], device=device)
    optimizer = optim.Adam(model.parameters(), lr=ae_cfg['learning_rate'])
    train_losses, val_losses = [], []

    # 5) Training loop
    print("\n--- Training ---------------------------------------------------")
    for epoch in range(1, ae_cfg['num_epochs'] + 1):
        t0 = time.time()
        model.train()
        running_train_loss = 0.0

        for b_idx, (batch, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch:>3} [train]", leave=False)):
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = model.loss(recon, batch, alpha).mean()
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

            if debug and b_idx < 3:
                gnorm = _grad_global_norm(model.parameters())
                print(f"  [Debug] epoch={epoch} batch={b_idx} loss={loss.item():.4f} grad_norm={gnorm:.3g}")

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation step
        avg_val_loss = 0.0
        if val_loader:
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for batch, _ in val_loader:
                    recon, _ = model(batch)
                    loss = model.loss(recon, batch, alpha).mean()
                    running_val_loss += loss.item()
            avg_val_loss = running_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

        t1 = time.time() - t0
        print(f"[Epoch {epoch:>3}/{ae_cfg['num_epochs']}] train={avg_train_loss:.4f}  val={avg_val_loss:.4f}  time={t1:5.1f}s")

    print("--- Training complete ------------------------------------------")

    # 6) Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "latent_dims": latent_dims,
        "variational": ae_cfg['variational'],
        "epoch": ae_cfg['num_epochs'],
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1] if val_losses else 0.0,
    }, model_save_path)
    print(f"[Save] model → {model_save_path}")

    # 7) Optional: embed whole dataset and save
    latent_save_path = ae_cfg.get('latent_save_path') # Use .get() for optional keys

    if latent_save_path:
        print("\n--- Embedding full dataset ---------------------------------")
        model.eval()
        full_data = torch.load(hvg_tensor_file, map_location=device, weights_only=False)
        counts, timepoints = full_data["counts"], full_data["timepoints"]
        with torch.no_grad():
            _, latent_reps = model(counts)
        
        os.makedirs(os.path.dirname(latent_save_path), exist_ok=True)
        torch.save({"latent_reps": latent_reps.cpu(), "timepoints": timepoints.cpu()}, latent_save_path)
        print(f"[Save] latent space → {latent_save_path}")

# ===================================================================== #
# 4. Command-Line Execution Wrapper
# ===================================================================== #
def main():
    # 1. Initialize parser and parse CLI arguments
    parser = setup_arg_parser()
    args = parser.parse_args()

    # 2. Build the override dictionary ONLY from user-provided arguments
    cli_overrides = {}
    if args.model_save_path is not None:
        cli_overrides['model_save_path'] = args.model_save_path
    if args.latent_save_path is not None:
        cli_overrides['latent_save_path'] = args.latent_save_path
    if args.latent_dims is not None:
        if len(args.latent_dims) < 2:
            raise ValueError("--latent_dims must include at least one intermediate and one bottleneck dimension.")
        cli_overrides['intermediate_dims'] = args.latent_dims[:-1]
        cli_overrides['bottleneck_dim'] = args.latent_dims[-1]
    if args.num_epochs is not None:
        cli_overrides['num_epochs'] = args.num_epochs
    if args.batch_size is not None:
        cli_overrides['batch_size'] = args.batch_size
    if args.lr is not None:
        cli_overrides['learning_rate'] = args.lr 
    if args.val_split is not None:
        cli_overrides['validation_split'] = args.val_split 
    if args.overdispersion is not None:
        cli_overrides['overdispersion'] = args.overdispersion
    
    # Nest the overrides inside the 'autoencoder' key to match YAML structure
    overrides = {"autoencoder": cli_overrides}

    # 3. Use the central loader to prepare the final config
    final_config = load_config(
        default_config_path="Config/autoencoder.yaml",
        overrides=overrides
    )

    # 4. Call the worker with the final config
    run_autoencoder_training(
        hvg_tensor_file=args.tensor_file,
        config=final_config,
        debug=args.debug # Pass the debug flag
    )

if __name__ == "__main__":
    main()