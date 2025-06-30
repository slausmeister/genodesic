#!/usr/bin/env python
import time
import os
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm
from typing import List

# --------------------------------------------------------------------- #
#  Local imports
# --------------------------------------------------------------------- #
from Genodesic.Dataloaders.CountLoader import create_count_dataloaders
from Genodesic.LatentCell.model          import LatentCell
# --------------------------------------------------------------------- #


# ===================================================================== #
# 1. Command-Line Interface Definition
# ===================================================================== #
def setup_arg_parser() -> argparse.ArgumentParser:
    """Define CLI arguments."""
    p = argparse.ArgumentParser(description="Train the LatentCell Autoencoder.")
    p.add_argument("--tensor_file", required=True, help="Path to the .pt file containing processed counts + timepoints")
    p.add_argument("--model_save_path", required=True, help="Path to save the trained model checkpoint")
    p.add_argument("--latent_save_path", default=None, help="Optional path to save the final latent space embedding + timepoints")
    p.add_argument("--latent_dims", type=int, nargs='+', default=[1000, 300, 100, 30], help="List of hidden-layer sizes down to the bottleneck")
    p.add_argument("--num_epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--val_split", type=float, default=0.20)
    p.add_argument("--overdispersion", type=float, default=0.3, help="α in the NB reconstruction loss")
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
    tensor_file: str,
    model_save_path: str,
    latent_save_path: str = None,
    latent_dims: List[int] = [1000, 300, 100, 30],
    num_epochs: int = 30,
    batch_size: int = 512,
    lr: float = 5e-4,
    val_split: float = 0.20,
    overdispersion: float = 0.3,
    debug: bool = False
):
    """
    Trains a LatentCell autoencoder on count data.
    """
    # 1) Device and Initial Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Init] device={device}    α={overdispersion}    latent_dims={latent_dims}")

    # 2) Data loading
    loaders = create_count_dataloaders(
        tensor_file,
        batch_size=batch_size,
        validation_split=val_split,
        shuffle=True,
        device=device,
    )
    train_loader, val_loader = (loaders[0], loaders[1]) if val_split > 0 else (loaders, None)

    if debug:
        ds_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset) if val_loader else 0
        print(f"[Debug] dataset={ds_size}  val={val_size}  train_batches={len(train_loader)}  val_batches={len(val_loader) if val_loader else 0}")

    # 3) Model / optimiser
    first_batch, _ = next(iter(train_loader))
    input_dim = first_batch.shape[1]
    print(f"[Init] detected HVGs: {input_dim}")
    model = LatentCell(input_dim, latent_dims, variational=False).to(device)
    alpha = torch.tensor([overdispersion], device=device)
    optim_ = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []

    # 4) Training loop
    print("\n--- Training ---------------------------------------------------")
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        running_train = 0.0

        for b_idx, (batch, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch:>3} [train]", leave=False)):
            optim_.zero_grad()
            recon, _ = model(batch)
            loss = model.loss(recon, batch, alpha).mean()
            loss.backward()
            optim_.step()
            running_train += loss.item()

            if debug and b_idx < 3:
                gnorm = _grad_global_norm(model.parameters())
                print(f"  [Debug] epoch={epoch}  batch={b_idx}  loss={loss.item():.4f}  grad_norm={gnorm:.3g}")

        avg_train = running_train / len(train_loader)
        train_losses.append(avg_train)

        # Validation
        avg_val = 0.0
        if val_loader:
            model.eval()
            val_accum = 0.0
            with torch.no_grad():
                for batch, _ in val_loader:
                    recon, _ = model(batch)
                    loss = model.loss(recon, batch, alpha).mean()
                    val_accum += loss.item()
            avg_val = val_accum / len(val_loader)
            val_losses.append(avg_val)

        t1 = time.time() - t0
        print(f"[Epoch {epoch:>3}/{num_epochs}] train={avg_train:.4f}  val={avg_val:.4f}  time={t1:5.1f}s")

    print("--- Training complete ------------------------------------------")

    # 5) Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "latent_dims": latent_dims,
        "variational": False,
        "epoch": num_epochs,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1] if val_losses else 0.0,
    }, model_save_path)
    print(f"[Save] model → {model_save_path}")

    # 6) Optional: embed whole dataset
    if latent_save_path:
        print("\n--- Embedding full dataset ---------------------------------")
        model.eval()
        full = torch.load(tensor_file, map_location=device)
        counts, tpoints = full["counts"], full["timepoints"]
        with torch.no_grad():
            _, latent = model(counts)
        os.makedirs(os.path.dirname(latent_save_path), exist_ok=True)
        torch.save({"latent_reps": latent.cpu(), "timepoints": tpoints.cpu()}, latent_save_path)
        print(f"[Save] latent space → {latent_save_path}")

# ===================================================================== #
# 4. Command-Line Execution Wrapper
# ===================================================================== #
def main():
    """Parses CLI arguments and launches the training."""
    parser = setup_arg_parser()
    args = parser.parse_args()
    run_autoencoder_training(
        tensor_file=args.tensor_file,
        model_save_path=args.model_save_path,
        latent_save_path=args.latent_save_path,
        latent_dims=args.latent_dims,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        overdispersion=args.overdispersion,
        debug=args.debug
    )

if __name__ == "__main__":
    main()