import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import resize
from PIL import Image
import glob
import random

# === VQVAE, VAR assumed to be already built and loaded (as in your code) ===
################## 1. Download checkpoints and build models
import os
if os.path.exists('/content/VAR'): os.chdir('/content/VAR')
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import VQVAE, build_vae_var

# we recommend using imagenet-512-d36 model to do the in-painting & out-painting & class-condition editing task
MODEL_DEPTH = 36    # TODO: =====> please specify MODEL_DEPTH <=====

assert MODEL_DEPTH in {16, 20, 24, 30, 36}


# download checkpoint
# hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
vae_ckpt, var_ckpt = './var_d36.pth', f'./var_d36.pth'
# if not osp.exists(vae_ckpt): os.system(f'wget {hf_home}/{vae_ckpt}')
# if not osp.exists(var_ckpt): os.system(f'wget {hf_home}/{var_ckpt}')

# build vae, var
FOR_512_px = MODEL_DEPTH == 36
if FOR_512_px:
    patch_nums = (1, 2, 3, 4, 6, 9, 13, 18, 24, 32)
else:
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vae, var = build_vae_var(
    V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
    device=device, patch_nums=patch_nums,
    num_classes=1000, depth=MODEL_DEPTH, shared_aln=FOR_512_px,
)

# load checkpoints
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
print(f'preparation finished.')

# === Flickr2K dataset loader with patch cropping ===
class Flickr2KSRDataset(Dataset):
    def __init__(self, root_dir, scale=2, patch_size=512, stride=256):
        self.hr_paths = sorted(glob.glob(os.path.join(root_dir, '*.png')))
        self.scale = scale
        self.patch_size = patch_size
        self.stride = stride
        self.to_tensor = transforms.ToTensor()
        self.patch_coords = self._build_patch_index()

    def _build_patch_index(self):
        patch_list = []
        for img_path in self.hr_paths:
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            for top in range(0, h - self.patch_size + 1, self.stride):
                for left in range(0, w - self.patch_size + 1, self.stride):
                    patch_list.append((img_path, left, top))
        return patch_list


    def __len__(self):
        return len(self.patch_coords)

    def __getitem__(self, idx):
        path, left, top = self.patch_coords[idx]
        hr_img = Image.open(path).convert('RGB')
        hr_patch = hr_img.crop((left, top, left + self.patch_size, top + self.patch_size))

        lr_patch = hr_patch.resize((self.patch_size // self.scale, self.patch_size // self.scale), Image.BICUBIC)
        lr_patch_up = lr_patch.resize((self.patch_size, self.patch_size), Image.BICUBIC)

        return self.to_tensor(lr_patch_up), self.to_tensor(hr_patch)

# === Loss: token-level prediction ===
def token_level_loss(pred_logits, target_idx):
    pred = pred_logits.reshape(-1, pred_logits.size(-1))
    target = target_idx.view(-1)
    return nn.CrossEntropyLoss()(pred, target)

# === Forward method with no class condition ===
def forward_autoregressive_teacher_forcing_no_class(var_model, input_tokens):
    B = input_tokens[-1].size(0)
    f_hat = torch.zeros(B, var_model.Cvae, var_model.patch_nums[-1], var_model.patch_nums[-1], device=input_tokens[-1].device)

    cur_L = 0
    logits = None
    for si, pn in enumerate(var_model.patch_nums):
        lvl_pos = var_model.lvl_embed(var_model.lvl_1L) + var_model.pos_1LC
        token_map = input_tokens[si].view(B, pn * pn).long()
        x = var_model.word_embed(var_model.vae_quant_proxy[0].embedding(token_map)).view(B, pn * pn, -1)
        x = x + lvl_pos[:, cur_L:cur_L + pn * pn]

        cond_BD = torch.zeros(B, var_model.class_emb.embedding_dim, device=x.device)  # no class condition
        cond_BD = var_model.shared_ada_lin(cond_BD)

        for blk in var_model.blocks:
            x = blk(x=x, cond_BD=cond_BD, attn_bias=None)

        logits = var_model.get_logits(x, cond_BD)  # overwrite until last stage
        cur_L += pn * pn

    return logits  # logits from final stage only

# === Training loop ===
from tqdm import tqdm

def train_var_for_sr(var, vae, dataloader, device='cuda', epochs=10, lr=1e-4):
    var.train()
    for p in vae.parameters():
        p.requires_grad_(False)

    optimizer = optim.Adam(var.parameters(), lr=lr)

    for epoch in range(epochs):
        print(f"[Epoch {epoch+1}/{epochs}]")
        total_loss = 0
        for i, (lr_img, hr_img) in enumerate(tqdm(dataloader, desc=f"Training", leave=False)):
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            with torch.no_grad():
                input_tokens = vae.img_to_idxBl(lr_img, var.patch_nums)
                target_tokens = vae.img_to_idxBl(hr_img, var.patch_nums)

            pred_logits = forward_autoregressive_teacher_forcing_no_class(var, input_tokens)
            loss = token_level_loss(pred_logits, target_tokens[-1])  # Only fine-tune final stage (32x32)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Step [{i+1}/{len(dataloader)}] Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}] Average Loss: {total_loss / len(dataloader):.4f}")


# === Usage Example ===
dataset = Flickr2KSRDataset(root_dir='./data/kagglehub/datasets/daehoyang/flickr2k/versions', scale=2, patch_size=512, stride=256)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
train_var_for_sr(var, vae, dataloader, device='cuda', epochs=5)
