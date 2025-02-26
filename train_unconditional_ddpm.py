''' 
This script does unconditional image generation on MNIST using a diffusion model

This code is modified from,
https://github.com/cloneofsimo/minDiffusion

Diffusion model is based on DDPM,
https://arxiv.org/abs/2006.11239
'''

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

class RotatedMNIST(Dataset):
    def __init__(self, root="./data", train=True, download=True, angles=None):
        """
        Custom Dataset for Rotated MNIST that applies rotations on-the-fly.
        """
        self.angles = angles if angles is not None else [0]
        
        # Load regular MNIST
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ])
        )
        
    def __len__(self):
        return len(self.mnist) * len(self.angles)
    
    def __getitem__(self, idx):
        # Map idx to original image index and angle index
        mnist_idx = idx // len(self.angles)
        angle_idx = idx % len(self.angles)
        angle = self.angles[angle_idx]
        
        # Get original image (ignore label)
        image, _ = self.mnist[mnist_idx]
        
        # Apply rotation
        if angle != 0:
            image = transforms.functional.rotate(image, angle)
            
        return image, torch.tensor(0)  # Return dummy label

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_res: bool = False) -> None:
        super().__init__()
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class Unet(nn.Module):
    def __init__(self, in_channels, n_feat = 256):
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, t):
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # Time embeddings
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(temb1*up1, down2)
        up3 = self.up2(temb2*up2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out

def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }

class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device):
        super(DDPM, self).__init__()
        self.nn_model = nn_model
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.loss_mse = nn.MSELoss()

    def forward(self, x):
        """
        this method is used in training, so samples t and noise randomly
        """
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )
        
        return self.loss_mse(noise, self.nn_model(x_t, _ts / self.n_T))

    def sample(self, n_sample, size, device):
        x_i = torch.randn(n_sample, *size).to(device)
        x_i_store = []

        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            eps = self.nn_model(x_i, t_is)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            
            if i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store

    def train_step(self, x):
        """Single training step with equivariance loss"""
        # Sample random timesteps
        t = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)
        
        # Path 1: Original -> Rotate -> Denoise
        x_t, noise, _ = self.forward(x, t)
        angles = torch.rand(x.shape[0], device=self.device) * 360  # Random angle for each image
        
        # Apply rotation individually to each image in the batch
        x_t_rotated = torch.stack([
            transforms.functional.rotate(x_t[i:i+1], angles[i].item()) 
            for i in range(x.shape[0])
        ])
        
        noise_pred_path1 = self.nn_model(x_t_rotated, t / self.n_T)
        
        # Path 2: Original -> Denoise -> Rotate
        noise_pred = self.nn_model(x_t, t / self.n_T)
        # Apply rotation individually to each predicted noise
        noise_pred_path2 = torch.stack([
            transforms.functional.rotate(noise_pred[i:i+1], angles[i].item())
            for i in range(x.shape[0])
        ])
        
        # Calculate losses
        denoising_loss = self.loss_mse(noise_pred, noise)
        equivariance_loss = self.loss_mse(noise_pred_path1, noise_pred_path2)
        
        # Combine losses
        lambda_equiv = 0.5  # Adjust this weight as needed
        total_loss = denoising_loss + lambda_equiv * equivariance_loss
        
        return {
            'total_loss': total_loss,
            'denoising_loss': denoising_loss.item(),
            'equivariance_loss': equivariance_loss.item()
        }

def train_mnist():
    # hardcoding these here
    torch.cuda.empty_cache()
    n_epoch = 50
    batch_size = 256
    n_T = 400
    device = "cuda:1"
    n_feat = 512
    lrate = 1e-4
    save_model = True
    save_dir = './data/diffusion_outputs_uncond/'

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Initialize unconditional model
    ddpm = DDPM(nn_model=Unet(in_channels=1, n_feat=n_feat), 
                betas=(1e-4, 0.02), n_T=n_T, device=device)
    ddpm.to(device)

    # Setup dataset with rotations
    angles = [10 * i for i in range(36)]  # 0 to 350 degrees in steps of 10
    dataset = RotatedMNIST(
        root="./data",
        train=True,
        download=True,
        angles=angles
    )
    
    # Calculate total dataset size and samples per epoch
    total_samples = len(dataset)
    samples_per_epoch = total_samples // 10  # Use 10% of data per epoch
    
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # Create random indices for this epoch's subset
        indices = torch.randperm(total_samples)[:samples_per_epoch]
        epoch_subset = torch.utils.data.Subset(dataset, indices)
        
        # Create dataloader for this epoch's subset
        dataloader = DataLoader(
            epoch_subset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=5
        )

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        
        for x, _ in pbar:  # Ignore labels
            optim.zero_grad()
            x = x.to(device)
            x = x.view(-1, 1, 28, 28)
            loss = ddpm.train_step(x)['total_loss']
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
        
        # Generate samples for visualization
        ddpm.eval()
        with torch.no_grad():
            n_sample = 40  # Total number of samples to generate
            x_gen, x_gen_store = ddpm.sample(n_sample, (1, 28, 28), device)

            # Append some real images at bottom
            x_real = x[:n_sample].to(device)
            x_all = torch.cat([x_gen, x_real])
            grid = make_grid(x_all*-1 + 1, nrow=8)
            save_image(grid, save_dir + f"image_ep{ep}.png")
            print('saved image at ' + save_dir + f"image_ep{ep}.png")

            if ep%5==0 or ep == int(n_epoch-1):
                # Create gif of images evolving over time
                fig, axs = plt.subplots(nrows=5, ncols=8, sharex=True, sharey=True, figsize=(8,5))
                def animate_diff(i, x_gen_store):
                    print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                    plots = []
                    for row in range(5):
                        for col in range(8):
                            idx = row * 8 + col
                            axs[row, col].clear()
                            axs[row, col].set_xticks([])
                            axs[row, col].set_yticks([])
                            plots.append(axs[row, col].imshow(-x_gen_store[i,idx,0], 
                                                            cmap='gray',
                                                            vmin=(-x_gen_store[i]).min(), 
                                                            vmax=(-x_gen_store[i]).max()))
                    return plots
                
                ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store],  
                                  interval=200, blit=False, repeat=True, 
                                  frames=x_gen_store.shape[0])    
                ani.save(save_dir + f"gif_ep{ep}.gif", dpi=100, writer=PillowWriter(fps=5))
                print('saved image at ' + save_dir + f"gif_ep{ep}.gif")
                
        # Save final model
        if save_model and ep == int(n_epoch-1):
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")

if __name__ == "__main__":
    train_mnist()
