import torch
import torch.optim as optim
import os
import argparse
from tqdm import tqdm


# We assume the Genodesic package is installed or in the PYTHONPATH
# This allows us to import our custom modules
from Genodesic.Dataloaders.CountLoader import create_count_dataloaders
from Genodesic.LatentCell.model import LatentCell

def setup_arg_parser():
    """
    Sets up the command-line argument parser for the training script.
    """
    parser = argparse.ArgumentParser(description="Train the LatentCell Autoencoder.")
    parser.add_argument(
        "--tensor_file",
        required=True,
        help="Path to the input .pt file containing processed counts and timepoints."
    )
    parser.add_argument(
        "--model_save_path",
        required=True,
        help="Path to save the final trained model checkpoint."
    )
    parser.add_argument(
        "--latent_dims",
        type=int,
        nargs='+',
        default=[1000, 300, 100, 30],
        help="A list of dimensions for the latent layers."
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=30,
        help="Number of epochs to train for."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for training and validation."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate for the Adam optimizer."
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Fraction of data to use for validation."
    )
    parser.add_argument(
        "--overdispersion",
        type=float,
        default=0.3,
        help="Overdispersion parameter alpha for the NB loss."
    )
    # Note: We are assuming a non-variational autoencoder for now, as in the notebook.
    # A '--variational' flag could be added here if needed.
    return parser

def main(args):
    """
    Main execution function for training the autoencoder.
    """
    # 1. Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Create DataLoaders
    # The dataloader needs to know the device to pre-load tensors
    loaders = create_count_dataloaders(args.tensor_file, args.batch_size, args.val_split, shuffle=True, device=device)
    if args.val_split > 0:
        train_loader, val_loader = loaders
    else:
        train_loader = loaders
        val_loader = None # No validation loader

    # 3. Initialize Model and Optimizer
    # Determine input dimension from the first batch of data
    first_batch_data, _ = next(iter(train_loader))
    input_dim = first_batch_data.shape[1]
    print(f"Detected input dimension (number of HVGs): {input_dim}")

    model = LatentCell(input_dim, args.latent_dims, variational=False)
    model.to(device)

    alpha = torch.tensor([args.overdispersion], device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    train_losses = []
    val_losses = []

    print("\n--- Starting Training ---")
    # 4. Training Loop
    for epoch in range(1, args.num_epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_data, _ in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
            optimizer.zero_grad()
            reconstructed, _ = model(batch_data)
            loss = model.loss(reconstructed, batch_data, alpha).mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        avg_val_loss = 0.0
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_data, _ in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                    reconstructed, _ = model(batch_data)
                    loss = model.loss(reconstructed, batch_data, alpha).mean()
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

        print(f"Epoch {epoch}/{args.num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    print("--- Training Complete ---")

    # 5. Save the final model
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'latent_dims': args.latent_dims,
        'variational': False,
        'epoch': args.num_epochs,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1] if val_losses else 0
    }

    torch.save(checkpoint, args.model_save_path)
    print(f"Model checkpoint saved successfully to: {args.model_save_path}")


if __name__ == '__main__':
    parser = setup_arg_parser()
    args = parser.parse_args()
    main(args)
