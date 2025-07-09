import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import os

from Genodesic.Utils.config_loader import load_config
from Genodesic.Dataloaders.LatentLoader import create_latent_meta_dataloader
from Genodesic.DensityModels import (
    MLP, TimeScoreNet, build_rq_nsf_model,
    OptimalFlowModel, ScoreSDEModel, RQNSFModel
)
from Genodesic.DensityModels.trainer import train_cfm_epoch, train_vpsde_epoch, train_rqnsf_epoch

def setup_cli_parser():
    """Defines the CLI for the training script."""
    parser = argparse.ArgumentParser(description="Unified training script for density models.")
    parser.add_argument("--config", type=str, default="Config/DensityModels.yaml", help="Path to the base YAML config file.")
    parser.add_argument("--data_file", type=str, help="Override path to the latent space data file.")
    parser.add_argument("--model_save_path", type=str, help="Override path to save the final model.")
    parser.add_argument("--model_type", type=str, choices=["vpsde", "otcfm", "rqnsf"], help="Override the model type.")
    parser.add_argument("--num_epochs", type=int, help="Override the number of training epochs.")
    parser.add_argument("--learning_rate", type=float, help="Override the learning rate.")
    parser.add_argument("--batch_size", type=int, help="Override the batch size.")
    return parser


def run_training(config: dict):
    """
    Core training function that operates on a single, fully-resolved config object.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = config["model_type"]
    model_specific_params = config["model_params"][model_type]
    
    print(f"--- Running Training for {model_type.upper()} ---")
    os.makedirs(os.path.dirname(config["model_save_path"]), exist_ok=True)

    print("INFO: Setting up dataloaders...")
    batch_size = model_specific_params.get("batch_size", config["batch_size"])
    print(f"INFO: Using effective batch size of {batch_size}")
    train_loader, val_loader = create_latent_meta_dataloader(
        data_file=config["data_file"],
        batch_size=batch_size,
        validation_split=config.get("validation_split", 0.2)
    )

    print(f"INFO: Initializing base network...")
    if model_type == "otcfm":
        network = MLP(
            dim=config["dim"], 
            w=model_specific_params["hidden_dim"], 
            time_varying=True, 
            activation=model_specific_params.get("activation", "selu"),
            num_layers=model_specific_params.get("num_layers", 4)
        ).to(device)
    elif model_type == "vpsde":
        network = TimeScoreNet(
            input_dim=config["dim"],
            hidden_dim=model_specific_params["hidden_dim"],
            num_layers=model_specific_params["num_layers"],
            activation=model_specific_params.get("activation", "selu")
        ).to(device)
    elif model_type == "rqnsf":
        network = build_rq_nsf_model(
            dim=config["dim"],
            n_blocks=model_specific_params["n_blocks"],
            bins=model_specific_params["bins"],
            subnet_width=model_specific_params["subnet_width"],
            subnet_activation=model_specific_params.get("subnet_activation", "selu")
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type in config: '{model_type}'")

    optimizer = optim.Adam(network.parameters(), lr=config["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)

    print("INFO: Starting training loop...")
    for epoch in range(config["num_epochs"]):
        if model_type == "otcfm":
            train_loss, val_loss = train_cfm_epoch(network, train_loader, val_loader, optimizer, device)
        elif model_type == "vpsde":
            train_loss, val_loss = train_vpsde_epoch(
                network, train_loader, val_loader, optimizer, device,
                beta_min=model_specific_params["beta_min"],
                beta_max=model_specific_params["beta_max"]
            )
        elif model_type == "rqnsf":
            train_loss, val_loss = train_rqnsf_epoch(
                network, train_loader, val_loader, optimizer, device, dim=config["dim"]
            )
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1:02d}/{config['num_epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Step 6: Save the checkpoint
    print(f"INFO: Training complete. Saving checkpoint to {config['model_save_path']}")
    checkpoint = {
        'model_state_dict': network.state_dict(),
        'model_type': config['model_type'],
        'hyperparameters': {'dim': config['dim'], **config['model_params'][config['model_type']]}
    }
    torch.save(checkpoint, config['model_save_path'])

    network.eval()
    if model_type == "otcfm":
        final_model = OptimalFlowModel(model=network, dim=config["dim"], device=device)
    elif model_type == "vpsde":
        sde_wrapper_params = {
            key: model_specific_params[key] 
            for key in ["beta_min", "beta_max"] 
            if key in model_specific_params
        }
        final_model = ScoreSDEModel(
            time_score_model=network, 
            dim=config['dim'], 
            device=device, 
            **sde_wrapper_params  
        )
    elif model_type == "rqnsf":
        final_model = RQNSFModel(model=network, dim=config['dim'], device=device)


    return final_model


def main_cli():
    """Handles command-line execution with our standard config pattern."""
    parser = setup_cli_parser()
    args = parser.parse_args()
    cli_overrides = {key: val for key, val in vars(args).items() if val is not None and key != 'config'}
    final_config = load_config(
        default_config_path=args.config,
        overrides=cli_overrides
    )
    run_training(final_config)

if __name__ == "__main__":
    main_cli()