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
import gc

class RotatedMNIST(Dataset):
    def __init__(self, root="./data", train=True, download=True, angles=None):
        """
        Custom Dataset for Rotated MNIST that applies rotations on-the-fly.
        
        Args:
            root (str): Path to store the MNIST data
            train (bool): If True, creates dataset from training set
            download (bool): If True, downloads the dataset
            angles (list): List of rotation angles in degrees. If None, uses [0]
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
        """
        Returns a rotated MNIST digit.
        """
        # Map idx to original image index and angle index
        mnist_idx = idx // len(self.angles)
        angle_idx = idx % len(self.angles)
        angle = self.angles[angle_idx]
        
        # Get original image (ignore label)
        image, _ = self.mnist[mnist_idx]
        
        # Apply rotation
        if angle != 0:
            image = transforms.functional.rotate(image, angle)
            
        return image.flatten()

class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
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
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
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
        '''
        generic one layer FC NN for embedding things  
        '''
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
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,
    }

class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # Register DDPM schedule
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.loss_mse = nn.MSELoss()

    def forward(self, x):
        """
        Makes forward diffusion x_t, and tries to guess noise added
        """
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )

        return self.loss_mse(noise, self.nn_model(x_t, _ts / self.n_T))

    def sample_steps(self, x_i, n_sample, size, device, start_timestep, end_timestep):
        """
        Samples from the learned model
        """
        with torch.no_grad():
            x_i_store = []
            for i in range(end_timestep, start_timestep, -1):
                print(f'sampling timestep {i}',end='\r')
                t_is = torch.tensor([i / self.n_T]).to(device)
                t_is = t_is.repeat(n_sample,1,1,1)

                z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
                eps = self.nn_model(x_i, t_is)
                x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
                )
                x_i_store.append(x_i.detach().cpu().numpy())

            return x_i, x_i_store

def rotate_image_naive(images, angle):
    """
    Rotate a batch of images by rearranging pixels for angles 0, 90, 180, or 270 degrees.
    """
    if angle == 0:
        return images
        
    if angle not in [90, 180, 270]:
        raise ValueError("Angle must be 0, 90, 180, or 270 degrees")
    
    B, C, H, W = images.shape
    rotated = torch.zeros_like(images)
    
    for b in range(B):
        for c in range(C):
            if angle == 90:
                rotated[b,c] = images[b,c].transpose(-2,-1).flip(dims=[-1])
            elif angle == 180:
                rotated[b,c] = images[b,c].flip(dims=[-2,-1])
            elif angle == 270:
                rotated[b,c] = images[b,c].transpose(-2,-1).flip(dims=[-2])
                
    return rotated

def rotate_image(images, angle):
    """
    Rotate a batch of images
    Args:
        images: Tensor of shape [batch_size, channels, height, width]
        angle: Rotation angle in degrees
    """
    return torch.stack([transforms.functional.rotate(img, angle) for img in images])

def generate_figure():
    n_feat = 512  
    n_T = 400
    device = "cuda:1"

    # Memory management
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    
    # Set global random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    ddpm = DDPM(nn_model=Unet(in_channels=1, n_feat=n_feat), 
                betas=(1e-4, 0.02), n_T=n_T, device=device)

    ddpm.load_state_dict(torch.load("./data/diffusion_outputs_uncond/model_24.pth", 
                                   map_location=device))

    ddpm.n_T = 400
    ddpm.eval()

    INDEX = 10

    # Sample multiple examples
    n_sample = 20
    x_i = torch.randn(n_sample, 1, 28, 28).to(device)

    # Reset seed before generating intermediate
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    print("generating intermediate")
    intermediate_i, intermediate_store = ddpm.sample_steps(x_i, n_sample, (1, 28, 28), device, INDEX, 400)
    print("finished with intermediate")

    # Reset seed before generating both paths
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    print("generating original")
    x_i, x_i_store = ddpm.sample_steps(intermediate_i, n_sample, (1, 28, 28), device, 1, INDEX)
    print("finished with original")
    x_i = rotate_image_naive(x_i.cpu(), 90)

    # Reset seed to use same noise sequence
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    print("generating transformed")
    transformed_i = rotate_image_naive(intermediate_i, 90)
    x_i_transformed, x_i_transformed_store = ddpm.sample_steps(transformed_i, n_sample, (1, 28, 28), device, 1, INDEX)
    print("finished with transformed")

    # Create publication-quality comparison plot
    fig = plt.figure(figsize=(20, 8))
    plt.suptitle(f'Effect of 90° Rotation at Step {INDEX}', fontsize=16, y=0.95)

    # Plot original samples on top
    for i in range(n_sample):
        plt.subplot(2, n_sample, i + 1)
        plt.imshow(x_i[i, 0].numpy(), cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        if i == 0:
            plt.title('Original Samples', pad=10)

    # Plot transformed samples on bottom
    for i in range(n_sample):
        plt.subplot(2, n_sample, n_sample + i + 1)
        plt.imshow(x_i_transformed[i, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        if i == 0:
            plt.title('Transformed Samples', pad=10)

    plt.tight_layout()
    plt.savefig(f'./figures/test/comparison_step{INDEX}.png', bbox_inches='tight', dpi=300)
    plt.close()

def mse_experiment(rotation_angle, use_naive=True):
    n_feat = 512  
    n_T = 400  
    device = "cuda:1"

    # Memory management
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    ddpm = DDPM(nn_model=Unet(in_channels=1, n_feat=n_feat), 
                betas=(1e-4, 0.02), n_T=n_T, device=device)

    ddpm.load_state_dict(torch.load("./data/diffusion_outputs_uncond/model_24.pth", 
                                   map_location=device))

    ddpm.n_T = 400
    ddpm.eval()

    # Set global random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    n_sample = 100
    x_i = torch.randn(n_sample, 1, 28, 28).to(device)

    indices = [i * 10 for i in range(40)]
    mses = []

    # Choose rotation function based on argument
    rotate_fn = rotate_image_naive if use_naive else rotate_image
    rotation_type = "naive" if use_naive else "interpolated"

    for INDEX in indices:
        print("running for index", INDEX)
        
        # Reset seed for each index to ensure reproducibility
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        print("generating intermediate")
        intermediate_i, intermediate_store = ddpm.sample_steps(x_i, n_sample, (1, 28, 28), device, INDEX, 400)
        
        # Reset seed before generating both paths
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        print("generating original")
        x_i, x_i_store = ddpm.sample_steps(intermediate_i, n_sample, (1, 28, 28), device, 1, INDEX)
        x_i = rotate_fn(x_i, rotation_angle)

        # Reset seed to use same noise sequence
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        print("generating transformed")
        transformed_i = rotate_fn(intermediate_i, rotation_angle)
        x_i_transformed, x_i_transformed_store = ddpm.sample_steps(transformed_i, n_sample, (1, 28, 28), device, 1, INDEX)

        with torch.no_grad():
            loss = nn.MSELoss()(x_i, x_i_transformed)
            mses.append(loss.cpu().item())

    plt.plot([ddpm.n_T - i for i in indices], mses)
    plt.savefig(f'./figures/unconditional/mse_experiment_{rotation_angle}_{rotation_type}.png', 
                bbox_inches='tight', dpi=300)
    plt.close()

    return mses

# if __name__ == "__main__":
#     # Dictionary to store MSEs for all angles and both rotation types
#     all_mses = {}
#     rotation_angles = [0, 5, 10, 25, 50, 75, 90, 135, 180, 225, 270, 315, 355]
    
#     # Run experiments for both naive and interpolated rotations
#     use_naive = False
#     rotation_type = "naive" if use_naive else "interpolated"
#     all_mses[rotation_type] = {}
    
#     for rotation_angle in rotation_angles:
#         print(f"\nRunning {rotation_type} experiment for rotation angle: {rotation_angle}°")
#         mses = mse_experiment(rotation_angle, use_naive)
#         all_mses[rotation_type][rotation_angle] = mses

#     # Save all MSEs to a single .npy file
#     np.save('./data/all_rotation_mses_both_types.npy', all_mses)
    
#     # Create comprehensive visualization comparing both methods
#     plt.figure(figsize=(15, 10))
    
#     # Calculate x-axis values (timesteps) once
#     indices = [i * 10 for i in range(40)]
#     timesteps = [400 - i for i in indices]
    
#     # Color maps for different angles
#     colors = plt.cm.viridis(np.linspace(0, 1, len(rotation_angles)))
    
#     # Create two subplots
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
#     # Plot naive rotations
#     for angle, color in zip(rotation_angles, colors):
#         ax1.plot(timesteps, all_mses["naive"][angle], color=color, 
#                 label=f'{angle}°', linewidth=2, alpha=0.8)
#     ax1.set_xlabel('Timestep', fontsize=12)
#     ax1.set_ylabel('Mean Squared Error', fontsize=12)
#     ax1.set_title('Naive Rotation', fontsize=14, pad=20)
#     ax1.grid(True, linestyle='--', alpha=0.7)
#     ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, title='Angles')
    
#     # Plot interpolated rotations
#     for angle, color in zip(rotation_angles, colors):
#         ax2.plot(timesteps, all_mses["interpolated"][angle], color=color, 
#                 label=f'{angle}°', linewidth=2, alpha=0.8)
#     ax2.set_xlabel('Timestep', fontsize=12)
#     ax2.set_ylabel('Mean Squared Error', fontsize=12)
#     ax2.set_title('Interpolated Rotation', fontsize=14, pad=20)
#     ax2.grid(True, linestyle='--', alpha=0.7)
#     ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, title='Angles')
    
#     plt.tight_layout()
#     plt.savefig('./figures/unconditional/combined_mse_experiment_comparison.png', 
#                 bbox_inches='tight', dpi=300)
#     plt.close()


if __name__ == "__main__":

    all_mses = {}
    rotation_angles = [0, 5, 10, 25, 50, 75, 90, 135, 180, 225, 270, 315, 355]
    all_mses["naive"] = {}
    all_mses["interpolated"] = {}

    # Load the data
    arr = np.load('./data/all_rotation_mses_both_types.npy', allow_pickle=True)

    # Convert the numpy array to dictionary
    data_dict = arr.item()

    # For interpolated rotations at a specific angle:
    interpolated_90_mses = data_dict['interpolated'][90]

    all_interpolated_results = data_dict['interpolated']

    print(all_interpolated_results.keys())

    colors = plt.cm.viridis(np.linspace(0, 1, len(rotation_angles)))

    plt.figure(figsize=(12, 8))
    indices = [i * 10 for i in range(40)]
    timesteps = [400 - i for i in indices]
    for angle, color in zip(rotation_angles, colors):
        plt.plot(timesteps, all_interpolated_results[angle], color=color, 
                label=f'{angle}', linewidth=2, alpha=0.8)


    # Customize plot
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.title('MSE vs Timestep for Different Rotation Angles (Unconditional Diffusion Model)', fontsize=14, pad=20)
    
    # Add legend with multiple columns
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
              ncol=2, title='Rotation Angles')
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    plt.savefig('./figures/unconditional/mse_experiment_total.png', bbox_inches='tight', dpi=300)
    plt.close()
