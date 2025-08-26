import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, in_chans, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, T)
        x = self.proj(x)        # (B, embed_dim, N)
        x = x.permute(0, 2, 1)  # (B, N, embed_dim)
        return x


class MAE_Encoder(nn.Module):
    def __init__(self, embed_dim, depth, nhead, ff_dim):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        return self.encoder(x)
    

class MAE_Decoder(nn.Module):
    def __init__(self, embed_dim, decoder_dim, depth, nhead, ff_dim, patch_size, num_channels):
        super().__init__()
        self.decoder_input_proj = nn.Linear(embed_dim, decoder_dim)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            batch_first=True,
            activation='gelu'
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=depth)
        self.output_proj = nn.Linear(decoder_dim, patch_size * num_channels)

    def forward(self, x):
        x = self.decoder_input_proj(x)
        x = self.decoder(x)
        x = self.output_proj(x)
        return x    # (B, N, patch_size * C)


class EEG_MAE(nn.Module):
    def __init__(self, num_channels=64, patch_size=100, embed_dim=128, encoder_depth=6,
                 decoder_dim=64, decoder_depth=4, nhead=4, ff_dim=256, mask_ratio=0.5, T=32):
        super().__init__()
        self.patch_embed = PatchEmbed(num_channels, patch_size, embed_dim)
        self.encoder = MAE_Encoder(embed_dim, encoder_depth, nhead, ff_dim)
        self.decoder = MAE_Decoder(embed_dim, decoder_dim, decoder_depth, nhead, ff_dim,
                                  patch_size, num_channels)

        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        
        # positional embeddings
            # num_patches = T // patch_size
            # (where T is the length of each EEG segment)
            #             = 2000 / 100 = 20
        num_patches = T // patch_size
        
        print(f"\nseq_len (T):\t{T}")
        print(f"patch_size:\t{patch_size}")
        print(f"num_patches:\t{num_patches}\n")
        
        self.pos_embed_enc = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        self.pos_embed_dec = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        # mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x):
        # x: (B, C, T)
        x_patches = self.patch_embed(x)  # (B, N, D)
        B, N, D = x_patches.shape
        device = x.device

        # add encoder positional embeddings
        x_patches = x_patches + self.pos_embed_enc[:, :N]

        # number of masked tokens
        num_mask = int(N * self.mask_ratio)
        rand_idx = torch.rand(B, N, device=device).argsort(dim=1)   # (B, N)
        keep_idx = rand_idx[:, num_mask:]       # visible
        masked_idx = rand_idx[:, :num_mask]     # masked

        # boolean mask (True = masked)
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        mask.scatter_(1, masked_idx, True)

        # gather visible tokens
        keep_idx_exp = keep_idx.unsqueeze(-1).expand(-1, -1, D)
        visible_tokens = torch.gather(x_patches, dim=1, index=keep_idx_exp)  # (B, N_visible, D)

        # encode only visible tokens
        encoded = self.encoder(visible_tokens)  # (B, N_visible, D)

        # prepare decoder input: combine encoded visible + mask tokens in original order
        decoder_input = torch.zeros_like(x_patches)
        # place encoded visible tokens
        keep_idx_exp_dec = keep_idx.unsqueeze(-1).expand(-1, -1, D)
        decoder_input.scatter_(1, keep_idx_exp_dec, encoded)
        # place mask tokens
        mask_tokens = self.mask_token.expand(B, num_mask, D)
        masked_idx_exp_dec = masked_idx.unsqueeze(-1).expand(-1, -1, D)
        decoder_input.scatter_(1, masked_idx_exp_dec, mask_tokens)

        # add decoder positional embeddings
        decoder_input = decoder_input + self.pos_embed_dec[:, :N]
        # decode
        decoded = self.decoder(decoder_input)  # (B, N, patch_size*C)

        # only take masked positions for loss
        masked_idx_exp_out = masked_idx.unsqueeze(-1).expand(-1, -1, self.patch_size*self.num_channels)
        decoded_masked = torch.gather(decoded, 1, masked_idx_exp_out)

        # target patches (for masked only)
        target = x.unfold(2, self.patch_size, self.patch_size)  # (B, C, N, patch_size)
        target = target.permute(0, 2, 1, 3).reshape(B, N, -1)   # (B, N, patch_dim)
        target_masked = torch.gather(target, 1, masked_idx_exp_out)  # (B, num_mask, patch_dim)
        
        return decoded_masked, target_masked, mask
