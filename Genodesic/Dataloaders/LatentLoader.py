import torch
from torch.utils.data import DataLoader, random_split, Dataset

class LatentMetaDataset(Dataset):
    def __init__(self, data_file):
        data = torch.load(data_file)
        self.latent_representations = data["latent_reps"].detach().clone().to(dtype=torch.float32)
        self.pseudotimes = data["timepoints"].detach().clone().to(dtype=torch.long).squeeze()
        if "clusters" in data:
            self.clusters = torch.tensor(data["clusters"], dtype=torch.long)
        else:
            self.clusters = torch.zeros_like(self.pseudotimes)

    def __len__(self):
        return len(self.latent_representations)

    def __getitem__(self, idx):
        return (
            self.latent_representations[idx],
            self.pseudotimes[idx],
            self.clusters[idx]
        )

def create_latent_meta_dataloader(data_file, batch_size=32, shuffle=True, validation_split=0.0):
    dataset = LatentMetaDataset(data_file)
    
    if validation_split <= 0.0:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), None

    train_size = int((1 - validation_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader