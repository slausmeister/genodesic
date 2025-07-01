import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import yaml
import argparse
import os
from copy import deepcopy

from Genodesic.Dataloaders.LatentLoader import create_latent_meta_dataloader
from Genodesic.DensityModels import MLP, TimeScoreNet, build_rq_nsf_model
from Genodesic.DensityModels.trainer import train_cfm_epoch, train_vpsde_epoch, train_rqnsf_epoch

def _deep_merge(source, destination):
    """Helper function for merging nested dictionaries."""
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            _deep_merge(value, node)
        else:
            destination[key] = value
    return destination

def run_training(
    config_overrides: dict = None, 
    default_config_path: str = "Config/models.yaml"
):
    """
    Core training function that loads a default config and merges notebook overrides.
    """
    # 1. --- Configuration Loading and Merging ---
    print(f"INFO: Loading default configuration from {default_config_path}")
    with open(default_config_path, 'r') as f:
        config = yaml.safe_load(f)

    if config_overrides:
        print("INFO: Merging notebook overrides into config.")
        config = _deep_merge(config_overrides, config)

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = config["model_type"]
    model_specific_params = config["model_params"][model_type]
    
    print(f"--- Running Training for {model_type.upper()} ---")
    os.makedirs(os.path.dirname(config["model_save_path"]), exist_ok=True)

    # 2. --- Data Loading ---
    print("INFO: Setting up dataloaders...")
    train_loader, val_loader = create_latent_meta_dataloader(
        data_file=config["data_file"],
        batch_size=config["batch_size"],
        validation_split=config.get("validation_split", 0.2)
    )

    # 3. --- Model Initialization ---
    print(f"INFO: Initializing model...")
    if model_type == "otcfm":
        model = MLP(dim=config["dim"], w=model_specific_params["hidden_dim"], time_varying=True).to(device)
    elif model_type == "vpsde":
        model = TimeScoreNet(
            input_dim=config["dim"],
            hidden_dim=model_specific_params["hidden_dim"],
            num_layers=model_specific_params["num_layers"]
        ).to(device)
    elif model_type == "rqnsf":
        model = build_rq_nsf_model(dim=config["dim"]).to(device)
    else:
        raise ValueError(f"Unknown model_type in config: '{model_type}'")

    # 4. --- Optimizer and Scheduler ---
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)

    # 5. --- Training Loop ---
    print("INFO: Starting training loop...")
    # ... (The rest of the training logic is identical)
    for epoch in range(config["num_epochs"]):
        if model_type == "otcfm":
            train_loss, val_loss = train_cfm_epoch(model, train_loader, val_loader, optimizer, device)
        elif model_type == "vpsde":
            train_loss, val_loss = train_vpsde_epoch(
                model, train_loader, val_loader, optimizer, device,
                beta_min=model_specific_params["beta_min"],
                beta_max=model_specific_params["beta_max"]
            )
        elif model_type == "rqnsf":
            train_loss, val_loss = train_rqnsf_epoch(model, train_loader, val_loader, optimizer, device)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1:02d}/{config['num_epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # 6. --- Save Final Model ---
    print(f"INFO: Training complete. Saving model to {config['model_save_path']}")
    torch.save(model.state_dict(), config["model_save_path"])
    return model

def main_cli():
    """
    This function handles the command-line interface execution.
    The notebook will NOT run this part.
    """
    parser = argparse.ArgumentParser(description="Unified training script for density models.")
    parser.add_argument("--config", type=str, required=True, help="Path to the training YAML config file.")
    args = parser.parse_args()

    print(f"INFO: Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run the training with the loaded config
    run_training(config)

if __name__ == "__main__":
    main_cli()