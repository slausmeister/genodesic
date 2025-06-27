import torch
import torch.nn as nn
import torch.distributions as dist

class LatentCell(nn.Module):
    def __init__(self, data_dim, latent_dims, variational=False):
        super(LatentCell, self).__init__()
        self.data_dim = data_dim
        self.latent_dims = [data_dim] + latent_dims
        self.variational = variational

        # Encoder
        encoder_layers = []
        for i in range(len(self.latent_dims) - 1):
            encoder_layers.append(nn.Linear(self.latent_dims[i], self.latent_dims[i + 1]))
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent parameters for variational case
        if self.variational:
            self.fc_mu = nn.Linear(self.latent_dims[-1], self.latent_dims[-1])
            self.fc_logvar = nn.Linear(self.latent_dims[-1], self.latent_dims[-1])

        # Decoder
        decoder_layers = []
        for i in range(len(self.latent_dims) - 1, 0, -1):
            decoder_layers.append(nn.Linear(self.latent_dims[i], self.latent_dims[i - 1]))
            decoder_layers.append(nn.SELU())
        # Ensure the output is positive
        decoder_layers.append(nn.Softplus())
        self.decoder = nn.Sequential(*decoder_layers)

        # Dispersion parameter θ

    def forward(self, x):
        # Normalize the input data
        umi_sums = x.sum(dim=1, keepdim=True)  # sum_j k_ij for each cell i
        x_normalized = torch.log1p((x / umi_sums) * 1e4)
    
        # Encode the normalized data
        z = self.encoder(x_normalized)

        # If variational, sample from the latent distribution
        if self.variational:
            mu = self.fc_mu(z)
            logvar = self.fc_logvar(z)
            std = torch.exp(0.5 * logvar)
            q = dist.Normal(mu, std)
            z = q.rsample()  # Reparameterization trick
        
        # Decode the latent representation
        x_decoded = self.decoder(z)
        # x_decoded is already non-negative due to Softplus

        # Reverse the normalization
        x_reconstructed = (torch.expm1(x_decoded) / 1e4) * umi_sums

        if self.variational:
            return x_reconstructed, z, mu, logvar
        else:
            return x_reconstructed, z

    def loss(self, y_pred, y_true, alpha, mu=None, logvar=None, current_epoch=None, total_epochs=None):
        eps = 1e-8  # Small constant to prevent division by zero

        # Ensure positivity
        y_pred = y_pred.clamp(min=eps)

        t1 = torch.lgamma(alpha + y_true) - torch.lgamma(alpha) - torch.lgamma(y_true + 1)
        t2 = alpha * (torch.log(alpha + eps) - torch.log(y_pred + alpha + eps))
        t3 = y_true * (torch.log(y_pred + eps) - torch.log(y_pred + alpha + eps))

        nb_loss = -(t1 + t2 + t3)

        # Add KL divergence loss for variational case with annealing
        if self.variational and mu is not None and logvar is not None:
            kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            if current_epoch is not None and total_epochs is not None:
                beta = min(1.0, 0.1 * current_epoch / total_epochs)  # Linear ramp-up for KL weight
                return nb_loss + beta * kl_divergence
            return nb_loss + kl_divergence

        return nb_loss
