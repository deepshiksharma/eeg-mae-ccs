import torch
import torch.nn.functional as F
from models.masked_autoencoder import EEG_MAE
from dataset_utils import create_dataloaders
from utils import save_checkpoint

import yaml
with open("config.yaml") as f:
    config = yaml.safe_load(f)
config = config['pretrain']


train_loader, val_loader = create_dataloaders(
    config['data_dir'],
    batch_size = config['batch_size'],
    val_size = config['validation_split_size']
)


num_channels = 0
seq_len = 0

model = EEG_MAE(num_channels=num_channels, T=seq_len)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)


# Training loop
num_epochs = config['train_epochs']

Train_Loss, Val_Loss = [], []

for epoch in range(num_epochs):
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
