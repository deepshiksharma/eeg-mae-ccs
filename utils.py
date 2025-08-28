import torch
import matplotlib.pyplot as plt


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }, save_path)


def plot_loss(a, b):
    ...


def plot_acc(a, b):
    ...


def plot_lr(a, b):
    ...


def get_params(model, type=None):
    def count_params(module):
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable
    
    # patch embedding
    patch_embed_0, patch_embed_1 = count_params(model.patch_embed)
    print(f"patch embedding params: {patch_embed_0}, trainable: {patch_embed_1}")

    # encoder
    enc_0, enc_1 = count_params(model.encoder)
    print(f"encoder params: {enc_0}, trainable: {enc_1}")
    # encoder positional embedding
    enc_pos_embed_0, enc_pos_embed_1 = count_params(model.pos_embed_enc)
    print(f"encoder positional embedding params: {enc_pos_embed_0}, trainable: {enc_pos_embed_1}")
    
    if type == "mae":
        # decoder
        dec_0, dec_1 = count_params(model.decoder)
        print(f"decoder params: {dec_0}, trainable: {dec_1}")
        # decoder positional embedding
        dec_pos_embed_0, dec_pos_embed_1 = count_params(model.pos_embed_dec) 
        print(f"decoder positional embedding params: {dec_pos_embed_0}, trainable: {dec_pos_embed_1}")

    elif type == "classifier":
        # classifier head
        clsf_0, clsf_1 = count_params(model.classifier)
        print(f"classifier params: {clsf_0}, trainable: {clsf_1}")
    
    # complete
    total_0, total_1 = count_params(model)
    print(f"model params: {total_0}, trainable: {total_1}")
