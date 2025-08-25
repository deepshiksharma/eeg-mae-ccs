import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from models.masked_autoencoder import EEG_MAE
import matplotlib.pyplot as plt

def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }, path)

def split_dataset(data, labels=None, val_size=0.1, seed=42):
    idx_train, idx_val = train_test_split(
        np.arange(len(data)),
        test_size=val_size,
        random_state=seed,
        stratify=labels if labels is not None else None
    )
    return idx_train, idx_val


class EEGDataset(Dataset):
    def __init__(self, data, labels=None):
        # data: np.array (N, C, T)
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        if self.labels is not None:
            y = torch.tensor(self.labels[idx], dtype=torch.long)
            return x, y
        return x


data = np.load('eeg_data.npy')  # prototype shape: (N, 64, 2000)  actual shape of data: (N, 59, 2000)
idx_train, idx_val = split_dataset(data)

train_dataset = EEGDataset(data[idx_train])
val_dataset = EEGDataset(data[idx_val])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)


model = EEG_MAE(num_channels=59)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("using device:", device, end="\n\n")
model.to(device)


# Training loop
epochs = 300

Train_Loss, Val_Loss = [], []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x in train_loader:
        x = x.to(device)

        # forward pass
        pred, target, mask = model(x)
        
        # loss on masked patches only
        loss = F.mse_loss(pred, target, reduction="mean")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"\n[Epoch {epoch+1}]")
    print(f"Train Loss: {avg_train_loss:.4f}")
    Train_Loss.append(avg_train_loss)

    # val
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)
            pred, target, mask = model(x)

            val_loss += F.mse_loss(pred, target, reduction="mean").item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}\n")
    Val_Loss.append(avg_val_loss)


save_checkpoint(model, optimizer, epoch, avg_val_loss, f"pretrained_epoch{epoch+1}.pt")

plt.plot(Train_Loss)
plt.plot(Val_Loss)
plt.savefig("pretraining_loss.png")
