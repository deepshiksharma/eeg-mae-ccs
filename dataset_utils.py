import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split


class LazyLoaderEEG(Dataset):
    """ Memory-efficient EEG data loading for huge .npy files. """
    
    def __init__(self, data_dir, labels=None):
        self.npy_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        if len(self.npy_files) == 0:
            raise FileNotFoundError(f"No .npy files found in {data_dir}")
        
        self.labels = labels
        self.cum_sizes = []
        total = 0
        for f in self.npy_files:
            shape = np.load(f, mmap_mode='r').shape
            total += shape[0]
            self.cum_sizes.append(total)
        self.total_samples = total

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        file_idx = next(i for i, cum_size in enumerate(self.cum_sizes) if idx < cum_size)
        idx_in_file = idx if file_idx == 0 else idx - self.cum_sizes[file_idx - 1]

        data_file = np.load(self.npy_files[file_idx], mmap_mode='r')
        x = torch.tensor(data_file[idx_in_file], dtype=torch.float32)

        if self.labels is not None:
            y = torch.tensor(self.labels[idx], dtype=torch.long)
            return x, y
        return x


def split_indices(dataset, val_size=0.1, test_size=None, labels=None, seed=25):
    """ Split dataset indices for train/val or train/val/test sets. """
    
    idx_all = np.arange(len(dataset))

    # train and val splits
    if test_size is None:
        idx_train, idx_val = train_test_split(
            idx_all,
            test_size=val_size,
            random_state=seed,
            stratify=labels if labels is not None else None
        )
        return idx_train, idx_val

    else:
        # test split
        idx_temp, idx_test = train_test_split(
            idx_all,
            test_size=test_size,
            random_state=seed,
            stratify=labels if labels is not None else None
        )
        # train and val splits
        adjusted_val_size = val_size / (1 - test_size)
        idx_train, idx_val = train_test_split(
            idx_temp,
            test_size=adjusted_val_size,
            random_state=seed,
            stratify=labels[idx_temp] if labels is not None else None
        )
        return idx_train, idx_val, idx_test


def create_dataloaders(data_dir, labels=None, batch_size=64, val_size=None, test_size=None, seed=25):
    dataset = LazyLoaderEEG(data_dir, labels=labels)
    
    split = split_indices(dataset, val_size=val_size, test_size=test_size, labels=labels, seed=seed)

    if test_size is None:
        idx_train, idx_val = split
        train_loader = DataLoader(Subset(dataset, idx_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, idx_val), batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
    
    else:
        idx_train, idx_val, idx_test = split
        train_loader = DataLoader(Subset(dataset, idx_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, idx_val), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(Subset(dataset, idx_test), batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader
