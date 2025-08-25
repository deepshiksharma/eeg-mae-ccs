import torch
import torch.nn as nn
from .masked_autoencoder import PatchEmbed, MAE_Encoder


class EEG_Classifier(nn.Module):
    def __init__(self, num_channels=64, patch_size=100, embed_dim=128, encoder_depth=6,
                 nhead=4, ff_dim=256, num_classes=4, T=32):
        super().__init__()
        self.patch_embed = PatchEmbed(num_channels, patch_size, embed_dim)

        # positional embeddings
        # note to self:
        #   refer to masked_autoencoder.py implementation for why num_patches are calculated dynamically.
        #   (tdlr; unlike text, eeg segments are fixed at 2000ms, no need for any slack)
        num_patches = T // patch_size

        print("seq len T:", T)
        print("patch size:", patch_size)
        print("num_patches:", num_patches)

        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        # encoder
        self.encoder = MAE_Encoder(embed_dim, encoder_depth, nhead, ff_dim)

        # classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x: (B, C, T)
        x_patches = self.patch_embed(x)  # (B, N, D)
        B, N, D = x_patches.shape

        # add positional embeddings
        x_patches = x_patches + self.pos_embed[:, :N]

        # encode all patches
        encoded = self.encoder(x_patches)  # (B, N, D)

        # global average pooling
        pooled = encoded.mean(dim=1)  # (B, D)
        
        # classifier
        logits = self.classifier(pooled)  # (B, num_classes)
        return logits
