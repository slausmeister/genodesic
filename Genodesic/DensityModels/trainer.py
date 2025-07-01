import torch
import numpy as np
from torchcfm.optimal_transport import OTPlanSampler
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher


def train_cfm_epoch(model, train_loader, val_loader, optimizer, device):
    """
    Performs one epoch of training and validation for an OT-CFM model.
    """
    # These can be instantiated here as they are stateless
    ot_sampler = OTPlanSampler(method="exact")
    cfm = ConditionalFlowMatcher(sigma=0.1) # As per your notebook
    
    # --- Training Phase ---
    model.train()
    train_losses = []
    for x1, _, _ in train_loader:
        x1 = x1.to(device)
        x0 = torch.randn_like(x1)
        x0, x1 = ot_sampler.sample_plan(x0, x1)
        t, xt, ut = cfm.sample_location_and_conditional_flow(x0, x1)
        
        optimizer.zero_grad()
        vt = model(torch.cat([xt, t[:, None]], dim=-1))
        loss = ((vt - ut)**2).mean()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # --- Validation Phase ---
    model.eval()
    val_losses = []
    with torch.no_grad():
        for x1, _, _ in val_loader:
            x1 = x1.to(device)
            x0 = torch.randn_like(x1)
            # No OT plan needed for validation loss calculation, can sample directly
            t, xt, ut = cfm.sample_location_and_conditional_flow(x0, x1)
            vt = model(torch.cat([xt, t[:, None]], dim=-1))
            loss = ((vt - ut)**2).mean()
            val_losses.append(loss.item())
            
    return np.mean(train_losses), np.mean(val_losses)


# ===================================================================== #
# 2. RQ-NSF Training Epoch
# ===================================================================== #

def train_rqnsf_epoch(model, train_loader, val_loader, optimizer, device):
    """
    Performs one epoch of training and validation for an RQ-NSF model.
    """
    # --- Training Phase ---
    model.train()
    train_losses = []
    for x1, _, _ in train_loader:
        x1 = x1.to(device)
        optimizer.zero_grad()
        
        z, log_jac = model(x1)
        log_p_z = -0.5 * (z**2).sum(dim=1) - 0.5 * model.dim * np.log(2.0 * np.pi)
        loss = -(log_p_z + log_jac).mean()

        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # --- Validation Phase ---
    model.eval()
    val_losses = []
    with torch.no_grad():
        for x1, _, _ in val_loader:
            x1 = x1.to(device)
            z, log_jac = model(x1)
            log_p_z = -0.5 * (z**2).sum(dim=1) - 0.5 * model.dim * np.log(2.0 * np.pi)
            loss = -(log_p_z + log_jac).mean()
            val_losses.append(loss.item())
            
    return np.mean(train_losses), np.mean(val_losses)


# ===================================================================== #
# 3. VP-SDE Training Epoch
# ===================================================================== #

def _analytical_mean_sigma(x0, t, beta_min, beta_max):
    """Helper for VP-SDE noise schedule."""
    integral_bt = beta_min * t + 0.5 * (beta_max - beta_min) * (t**2)
    alpha_t = torch.exp(-0.5 * integral_bt)
    mean = alpha_t * x0
    sigma = torch.sqrt(1.0 - alpha_t**2)
    return mean, sigma

def _sde_training_step(model, x0, beta_min, beta_max):
    """Calculates the loss for a single batch for the VP-SDE."""
    t = torch.rand(x0.shape[0], 1, device=x0.device) * (1 - 1e-5) + 1e-5
    mean, sigma = _analytical_mean_sigma(x0, t, beta_min, beta_max)
    
    eps = torch.randn_like(x0)
    xt = mean + sigma * eps
    
    score_true = -eps / (sigma + 1e-7)
    score_pred = model(xt, t)
    
    # Denoising score matching loss weighted by lambda(t) = sigma^2
    loss = ((score_pred - score_true)**2 * (sigma**2)).sum(dim=-1).mean()
    return loss

def train_vpsde_epoch(model, train_loader, val_loader, optimizer, device, beta_min, beta_max):
    """
    Performs one epoch of training and validation for a VP-SDE model.
    """
    # --- Training Phase ---
    model.train()
    train_losses = []
    for x0, _, _ in train_loader:
        x0 = x0.to(device)
        optimizer.zero_grad()
        loss = _sde_training_step(model, x0, beta_min, beta_max)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # --- Validation Phase ---
    model.eval()
    val_losses = []
    with torch.no_grad():
        for x0, _, _ in val_loader:
            x0 = x0.to(device)
            loss = _sde_training_step(model, x0, beta_min, beta_max)
            val_losses.append(loss.item())
            
    return np.mean(train_losses), np.mean(val_losses)