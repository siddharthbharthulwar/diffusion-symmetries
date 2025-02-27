"""
Refactored MNIST Equivariant Diffusion Model

This script trains a rotation-equivariant diffusion model on MNIST,
mirroring the class-based structure in the 2D point diffusion script.
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import math
import concurrent.futures

# --------------------------------------------------------------------------------
# Utility function for rotating MNIST images
# --------------------------------------------------------------------------------
def rotate_image(images, angles):
    """
    Rotate a batch of images with different angles per image
    Args:
        images: Tensor of shape [batch_size, channels, height, width]
        angles: Tensor of shape [batch_size] containing rotation angles in degrees
    """
    batch_size = images.shape[0]
    angles = angles * math.pi / 180.0  # Convert to radians
    
    # Create rotation matrices
    cos_theta = torch.cos(angles)
    sin_theta = torch.sin(angles)
    
    # Build affine transformation matrix for each image
    rotation_matrix = torch.zeros(batch_size, 2, 3, device=images.device)
    rotation_matrix[:, 0, 0] = cos_theta
    rotation_matrix[:, 0, 1] = -sin_theta
    rotation_matrix[:, 1, 0] = sin_theta
    rotation_matrix[:, 1, 1] = cos_theta
    
    # Create sampling grid and rotate images
    grid = F.affine_grid(rotation_matrix, images.size(), align_corners=False)
    rotated_images = F.grid_sample(images, grid, align_corners=False)
    
    return rotated_images

# --------------------------------------------------------------------------------
# Custom Dataset for applying rotations to MNIST on-the-fly
# --------------------------------------------------------------------------------
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
            transform=transforms.ToTensor()
        )
        
    def __len__(self):
        return len(self.mnist) * len(self.angles)
    
    def __getitem__(self, idx):
        # Map idx to original image index and angle index
        mnist_idx = idx // len(self.angles)
        angle_idx = idx % len(self.angles)
        angle = self.angles[angle_idx]
        
        # Get original image (ignore label for unconditional training)
        image, _ = self.mnist[mnist_idx]
        
        # Apply rotation
        if angle != 0:
            image = transforms.functional.rotate(image, angle)
            
        return image, torch.tensor(0)  # Return dummy label for convenience

# --------------------------------------------------------------------------------
# Network modules for the UNet
# --------------------------------------------------------------------------------
class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_res: bool = False) -> None:
        super().__init__()
        self.same_channels = (in_channels == out_channels)
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
            return out / 1.414  # approximate scaling factor
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
        super().__init__()
        # Make sure we set self.input_dim
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
    """
    A simple UNet-based architecture for image diffusion.
    """
    def __init__(self, in_channels, n_feat = 256):
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        # Time embeddings
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
        """
        x: (B, 1, 28, 28)
        t: normalized time (e.g. t / self.n_T), shape (B,)
        """
        x0 = self.init_conv(x)
        down1 = self.down1(x0)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # Time embeddings
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(temb1 * up1, down2)
        up3 = self.up2(temb2 * up2, down1)
        out = self.out(torch.cat((up3, x0), 1))
        return out

# --------------------------------------------------------------------------------
# DDPM Schedules and a helpful function
# --------------------------------------------------------------------------------
def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
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

# --------------------------------------------------------------------------------
# Main class that encapsulates the MNIST diffusion model + training
# --------------------------------------------------------------------------------
class MNISTDiffusion(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_feat=512,
        betas=(1e-4, 0.02),
        n_T=400,
        device="cuda:0",
        lrate=1e-4,
    ):
        super().__init__()
        self.device = device
        self.n_T = n_T
        
        # Create the underlying UNet
        self.nn_model = Unet(in_channels=in_channels, n_feat=n_feat).to(device)
        
        # Precompute DDPM schedules/buffers
        sched = ddpm_schedules(betas[0], betas[1], n_T)
        for k, v in sched.items():
            self.register_buffer(k, v)
        
        # Define MSE loss
        self.loss_mse = nn.MSELoss()

        # Adam optimizer
        self.optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=lrate)

    def forward(self, x, t=None):
        """
        Forward diffusion step to produce noisy x_t from x_0,
        optionally for a specific t. If t is None, picks a random t.
        
        Returns: (x_t, noise, predicted_noise)
        """
        if t is None:
            t = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)
        
        noise = torch.randn_like(x)
        x_t = (
            self.sqrtab[t, None, None, None] * x
            + self.sqrtmab[t, None, None, None] * noise
        )
        pred = self.nn_model(x_t, t.float() / self.n_T)
        return x_t, noise, pred

    def train_step(self, x, lambda_equiv=0.0):
        """
        Single training step: 
        1) Sample random t
        2) Create noisy x_t
        3) Compute standard denoising loss
        4) Compute equivariance loss by comparing rotated paths
        5) Return dictionary of losses

        Args:
            x (Tensor): Batch of images (B, 1, 28, 28)
            lambda_equiv (float): Weight for rotation equivariance loss
        """
        self.optimizer.zero_grad()

        # Sample timesteps
        t = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)

        # Forward diffuse
        noise = torch.randn_like(x)
        x_t = (
            self.sqrtab[t, None, None, None] * x
            + self.sqrtmab[t, None, None, None] * noise
        )

        # Path 1: Original -> Rotate -> Denoise
        angles = torch.rand(x.shape[0], device=self.device) * 360  # random angle
        x_t_rotated = rotate_image(x_t, angles)
        noise_pred_path1 = self.nn_model(x_t_rotated, t.float()/self.n_T)
        
        # Path 2: Original -> Denoise -> Rotate
        noise_pred = self.nn_model(x_t, t.float()/self.n_T)
        noise_pred_path2 = rotate_image(noise_pred, angles)

        # Denoising loss
        denoising_loss = self.loss_mse(noise_pred, noise)
        
        # Equivariance loss
        equivariance_loss = self.loss_mse(noise_pred_path1, noise_pred_path2)

        # Combine
        total_loss = denoising_loss + lambda_equiv * equivariance_loss
        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "denoising_loss": denoising_loss.item(),
            "equivariance_loss": equivariance_loss.item(),
        }

    @torch.no_grad()
    def sample(self, n_sample, size):
        """
        Sample from the model using DDPM (ancestral sampling).
        Returns:
            - final sample
            - list of intermediate steps (for visualization)
        """
        x_i = torch.randn(n_sample, *size).to(self.device)
        x_i_store = []

        for i in range(self.n_T, 0, -1):
            t_is = torch.tensor([i / self.n_T], device=self.device).repeat(n_sample,1,1,1)
            z = torch.randn(n_sample, *size, device=self.device) if i > 1 else 0

            eps = self.nn_model(x_i, t_is)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            
            # Save some frames
            if i % 20 == 0 or i == self.n_T or i < 8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        return x_i, np.array(x_i_store)

    def sample_steps(self, x_i, n_sample, size, device, start_timestep, end_timestep):
        """
        Samples from the learned model
        """
        with torch.no_grad():
            x_i_store = []
            for i in range(end_timestep, start_timestep, -1):
                print(f'sampling timestep {i}',end='\r')
                t_is = torch.tensor([i / self.n_T], device=device).repeat(n_sample,1,1,1)
                z = torch.randn(n_sample, *size, device=device) if i > 1 else 0
                eps = self.nn_model(x_i, t_is)
                x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
                )
                x_i_store.append(x_i.detach().cpu().numpy())
            return x_i, x_i_store

# --------------------------------------------------------------------------------
# Score comparison function
# --------------------------------------------------------------------------------
def compare_scores_highdim(
    modelA, 
    modelB,
    data_loader,                # PyTorch DataLoader for MNIST/CIFAR
    alpha_bar,                  # Precomputed alpha_bar[t], same shape as # timesteps
    n_steps: int,
    n_batches: int = 10,        # How many batches to sample for the comparison
    device: str = "cuda"
):
    """
    Compare the predicted scores of modelA and modelB on real high-dimensional data.
    Returns average L2 difference and average angular difference across sampled data.
    """
    modelA.eval()
    modelB.eval()
    
    all_l2_diffs = []
    all_angles = []
    eps = 1e-8
    
    for batch_idx, (images, _) in enumerate(data_loader):
        if batch_idx >= n_batches:
            break
        
        # Images shape: (B, C, H, W)
        # Move to GPU
        images = images.to(device)
        batch_size = images.shape[0]
        
        # Pick a random timestep for each sample in the batch
        t = torch.randint(low=0, high=n_steps, size=(batch_size,), device=device, dtype=torch.long)
        
        # Forward-diffuse images to x_t
        alpha_bar_t = alpha_bar[t]  # shape: (B,)
        noise = torch.randn_like(images)
        alpha_t_sqrt = alpha_bar_t.sqrt().view(-1, 1, 1, 1)
        one_minus_alpha_t = (1 - alpha_bar_t).sqrt().view(-1, 1, 1, 1)
        x_t = alpha_t_sqrt * images + one_minus_alpha_t * noise
        
        # Get predicted noise eps_theta from each model
        with torch.no_grad():
            eps_pred_A = modelA(x_t, t)  # Pass integer timesteps directly
            eps_pred_B = modelB(x_t, t)
            
            # If the models return tuples, get just the predicted noise (first element)
            if isinstance(eps_pred_A, tuple):
                eps_pred_A = eps_pred_A[0]
            if isinstance(eps_pred_B, tuple):
                eps_pred_B = eps_pred_B[0]
        
        # Convert predicted noise to predicted score
        denom = torch.sqrt(1 - alpha_bar_t).view(-1,1,1,1)
        scoreA = -eps_pred_A / denom
        scoreB = -eps_pred_B / denom
        
        # Flatten scores into vectors
        scoreA_vec = scoreA.view(batch_size, -1)  # (B, D)
        scoreB_vec = scoreB.view(batch_size, -1)  # (B, D)
        
        # L2 difference
        diff_vec = scoreA_vec - scoreB_vec
        l2_diff = torch.norm(diff_vec, dim=1)  # shape (B,)
        
        # Angular difference
        dot_products = (scoreA_vec * scoreB_vec).sum(dim=1)
        normA = torch.norm(scoreA_vec, dim=1)
        normB = torch.norm(scoreB_vec, dim=1)
        cos_theta = dot_products / (normA * normB + eps)
        
        # clip to [-1, 1] to avoid NaNs
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        angle = torch.acos(cos_theta)  # in [0, pi]
        
        # Accumulate
        all_l2_diffs.append(l2_diff.cpu().numpy())
        all_angles.append(angle.cpu().numpy())
    
    # Concatenate over all batches
    all_l2_diffs = np.concatenate(all_l2_diffs, axis=0)
    all_angles = np.concatenate(all_angles, axis=0)
    
    # Compute average metrics
    avg_l2 = all_l2_diffs.mean()
    avg_angle = all_angles.mean()
    
    return avg_l2, avg_angle

# --------------------------------------------------------------------------------
# Main training routine
# --------------------------------------------------------------------------------

def train_mnist(device="cuda:0"):
    # Hyperparameters
    torch.cuda.empty_cache()
    n_epoch = 10
    batch_size = 128
    n_T = 400
    n_feat = 512
    lrate = 1e-4
    lambda_equiv = 0.0   # adjust if you want rotation equiv. to matter more
    angles = [10 * i for i in range(10)]  # rotation angles for the dataset
    save_dir = "./equivariant_mnist/standard/"
    os.makedirs(save_dir, exist_ok=True)

    # Create dataset/dataloader
    dataset = RotatedMNIST(root="./data", train=True, download=True, angles=angles)
    total_samples = len(dataset)
    samples_per_epoch = total_samples // 10  # Using a smaller subset each epoch
    ddpm = MNISTDiffusion(
        in_channels=1, 
        n_feat=n_feat, 
        betas=(1e-4, 0.02), 
        n_T=n_T, 
        device=device, 
        lrate=lrate
    ).to(device)

    # Prepare arrays to track losses
    total_losses = []
    denoising_losses = []
    equivariance_losses = []

    for ep in range(n_epoch):
        print(f"Epoch {ep}")
        ddpm.train()

        # Subsample data for this epoch
        indices = torch.randperm(total_samples)[:samples_per_epoch]
        epoch_subset = torch.utils.data.Subset(dataset, indices)
        dataloader = DataLoader(
            epoch_subset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4
        )

        # Simple linear lr decay
        ddpm.optimizer.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None

        for x, _ in pbar:
            x = x.to(device)
            x = x.view(-1, 1, 28, 28)  # ensure shape
            loss_dict = ddpm.train_step(x, lambda_equiv=0)
            
            if loss_ema is None:
                loss_ema = loss_dict["total_loss"]
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss_dict["total_loss"]
            
            total_losses.append(loss_dict["total_loss"])
            denoising_losses.append(loss_dict["denoising_loss"])
            equivariance_losses.append(loss_dict["equivariance_loss"])
            pbar.set_description(f"loss: {loss_ema:.4f}")

        # Generate samples to visualize after each epoch
        ddpm.eval()
        with torch.no_grad():
            n_sample = 40
            x_gen, x_gen_store = ddpm.sample(n_sample, (1, 28, 28))
            
            # Combine generated + real images
            x_real = x[:n_sample].to(device)
            x_all = torch.cat([x_gen, x_real])
            grid = make_grid(x_all * -1 + 1, nrow=8)
            save_image(grid, os.path.join(save_dir, f"image_ep{ep}.png"))
            print("Saved sample image:", os.path.join(save_dir, f"image_ep{ep}.png"))

            # Create a gif every 5 epochs or in the last epoch
            if ep % 5 == 0 or ep == (n_epoch - 1):
                fig, axs = plt.subplots(nrows=5, ncols=8, sharex=True, sharey=True, figsize=(8, 5))

                def animate_diff(i, x_store):
                    for row in range(5):
                        for col in range(8):
                            idx = row * 8 + col
                            axs[row, col].clear()
                            axs[row, col].set_xticks([])
                            axs[row, col].set_yticks([])
                            axs[row, col].imshow(-x_store[i, idx, 0], cmap='gray',
                                                 vmin=(-x_store[i]).min(),
                                                 vmax=(-x_store[i]).max())
                
                def update(frame):
                    animate_diff(frame, x_gen_store)

                ani = FuncAnimation(fig, update, frames=len(x_gen_store), interval=200, repeat=True)
                gif_path = os.path.join(save_dir, f"gif_ep{ep}.gif")
                ani.save(gif_path, dpi=100, writer=PillowWriter(fps=5))
                plt.close(fig)
                print("Saved GIF:", gif_path)

    # Save final model
    final_model_path = os.path.join(save_dir, f"model_{n_epoch-1}.pth")
    torch.save(ddpm.state_dict(), final_model_path)
    print("Saved final standard model:", final_model_path)

    return ddpm

def evaluate_single_step_equivariance(rotation_angles, device, n_T, n_sample, model_path, lrate, n_feat, betas):
    """
    Loads a copy of the equivariant model on the specified device,
    then for each rotation angle in rotation_angles, performs a single-step
    equivariance test over all timesteps (except t=0) and returns the results.
    """
    # Create and load the model on the given device
    ddpm_model = MNISTDiffusion(
        in_channels=1, 
        n_feat=n_feat, 
        betas=betas, 
        n_T=n_T, 
        device=device, 
        lrate=lrate
    ).to(device)
    ddpm_model.load_state_dict(torch.load(model_path, map_location=device))
    ddpm_model.eval()

    results = {}
    for rotation_angle in rotation_angles:
        print(f"[Device {device}] Testing single-step rotation angle: {rotation_angle}°")
        single_step_mses = []
        for index in range(n_T):
            if index == 0:
                single_step_mses.append(0)
                continue

            # Set seeds for reproducibility
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)

            # Generate a batch of initial latent states (proxy for x_t)
            initial_latent = torch.randn(n_sample, 1, 28, 28).to(device)
            t_tensor = torch.full((n_sample,), index / n_T, device=device)

            # Sample noise for the one-step update (use 0 if index==1)
            z = torch.randn(n_sample, 1, 28, 28, device=device) if index > 1 else 0

            # Create a tensor for the current rotation angle
            angle_tensor = torch.full((n_sample,), rotation_angle, device=device, dtype=torch.float)

            # ----- Path 1: Rotate then perform one denoising step -----
            rotated_latent = rotate_image(initial_latent, angle_tensor)
            eps1 = ddpm_model.nn_model(rotated_latent, t_tensor)
            x1 = ddpm_model.oneover_sqrta[index] * (rotated_latent - eps1 * ddpm_model.mab_over_sqrtmab[index]) \
                 + ddpm_model.sqrt_beta_t[index] * z

            # ----- Path 2: Denoise then rotate -----
            eps2 = ddpm_model.nn_model(initial_latent, t_tensor)
            x_temp = ddpm_model.oneover_sqrta[index] * (initial_latent - eps2 * ddpm_model.mab_over_sqrtmab[index]) \
                     + ddpm_model.sqrt_beta_t[index] * z
            x2 = rotate_image(x_temp, angle_tensor)

            # Compute the MSE between the two paths
            loss = F.mse_loss(x1, x2)
            single_step_mses.append(loss.item())

        results[rotation_angle] = {
            'timesteps': [n_T - i for i in range(n_T)],
            'mses': single_step_mses
        }
    return results

if __name__ == "__main__":

    device = "cuda:0"
    
    # Hyperparameters that will be used for both models
    n_epoch = 200
    batch_size = 128
    n_T = 400
    n_feat = 512
    lrate = 1e-4
    angles = [10 * i for i in range(5)]  # rotation angles for the dataset

    # Load standard model
    standard_model_dir = "./equivariant_mnist/baseline/"
    standard_model_path = os.path.join(standard_model_dir, "model_199.pth")  # Load final model
    
    print(f"Loading standard model from {standard_model_path}")
    ddpm_standard = MNISTDiffusion(
        in_channels=1, 
        n_feat=n_feat, 
        betas=(1e-4, 0.02), 
        n_T=n_T, 
        device=device, 
        lrate=lrate
    ).to(device)
    ddpm_standard.load_state_dict(torch.load(standard_model_path))
    
    # Load equivariant model
    equivariant_model_dir = "./equivariant_mnist/equivariant/"
    equivariant_model_path = os.path.join(equivariant_model_dir, "model_199.pth")  # Load final model
    
    print(f"Loading equivariant model from {equivariant_model_path}")
    ddpm_equivariant = MNISTDiffusion(
        in_channels=1, 
        n_feat=n_feat, 
        betas=(1e-4, 0.02), 
        n_T=n_T, 
        device=device, 
        lrate=lrate
    ).to(device)
    ddpm_equivariant.load_state_dict(torch.load(equivariant_model_path))

    # -----------------------------
    # Single-step equivariance error evaluation on both models
    # -----------------------------
    print("Testing single-step equivariance error on both models...")

    # We'll test over a range of rotation angles (in degrees) and sample timesteps
    rotation_angles = list(range(0, 361, 30))
    # Sample fewer timesteps to speed up evaluation
    indices = list(range(0, n_T, 20)) + [n_T-1]  # Sample every 20 steps + the last step
    n_sample = 100  # number of samples per evaluation

    # Dictionary to store results for both models
    standard_results = {}
    equivariant_results = {}

    # Function to evaluate single-step equivariance for a model
    def evaluate_model_equivariance(model, model_name):
        results = {}
        for rotation_angle in rotation_angles:
            print(f"Testing {model_name} model with rotation angle: {rotation_angle}°")
            single_step_mses = []
            
            for index in indices:
                # For index==0, we cannot take a denoising step, so record 0 error
                if index == 0:
                    single_step_mses.append(0)
                    print(f"  Timestep {index}: MSE = 0 (skipped)")
                    continue

                # Ensure reproducibility
                torch.manual_seed(42)
                torch.cuda.manual_seed(42)

                # Generate initial latent state
                initial_latent = torch.randn(n_sample, 1, 28, 28).to(device)
                t_tensor = torch.full((n_sample,), index / n_T, device=device)
                z = torch.randn(n_sample, 1, 28, 28, device=device) if index > 1 else 0

                # ----- Path 1: Rotate then perform one denoising step -----
                angle_tensor = torch.full((n_sample,), rotation_angle, device=device, dtype=torch.float)
                rotated_latent = rotate_image(initial_latent, angle_tensor)
                eps1 = model.nn_model(rotated_latent, t_tensor)
                x1 = model.oneover_sqrta[index] * (rotated_latent - eps1 * model.mab_over_sqrtmab[index]) \
                    + model.sqrt_beta_t[index] * z

                # ----- Path 2: Perform one denoising step then rotate -----
                eps2 = model.nn_model(initial_latent, t_tensor)
                x_temp = model.oneover_sqrta[index] * (initial_latent - eps2 * model.mab_over_sqrtmab[index]) \
                        + model.sqrt_beta_t[index] * z
                x2 = rotate_image(x_temp, angle_tensor)

                # Compute the mean squared error between the two paths
                with torch.no_grad():
                    loss = F.mse_loss(x1, x2)
                single_step_mses.append(loss.item())
                print(f"  Timestep {index}: MSE = {loss.item():.6f}")
            
            # Store the results
            results[rotation_angle] = {
                'timesteps': [n_T - i for i in indices],
                'mses': single_step_mses
            }
        return results

    # Evaluate both models
    print("\n=== Evaluating Standard Model ===")
    standard_results = evaluate_model_equivariance(ddpm_standard, "standard")
    
    print("\n=== Evaluating Equivariant Model ===")
    equivariant_results = evaluate_model_equivariance(ddpm_equivariant, "equivariant")

    # Print a summary of the results
    print("\n=== Summary of Results ===")
    print("Average MSE per rotation angle:")
    
    print("\nStandard Model:")
    for angle, data in standard_results.items():
        avg_mse = sum(data['mses']) / len(data['mses'])
        print(f"  {angle}°: {avg_mse:.6f}")
    
    print("\nEquivariant Model:")
    for angle, data in equivariant_results.items():
        avg_mse = sum(data['mses']) / len(data['mses'])
        print(f"  {angle}°: {avg_mse:.6f}")

    # Plot the results for the standard model
    plt.figure(figsize=(10, 6))
    for angle, data in standard_results.items():
        plt.plot(data['timesteps'], data['mses'], label=f'{angle}° rotation')
    plt.xlabel('Timestep (n_T - index)')
    plt.ylabel('Single-Step MSE between paths')
    plt.title('Single-Step Standard Error on MNIST (Standard Model)')
    plt.legend()
    standard_plot_path = os.path.join("./equivariant_mnist/", "single_step_standard_model.png")
    plt.savefig(standard_plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved standard model plot to: {standard_plot_path}")

    # Plot the results for the equivariant model
    plt.figure(figsize=(10, 6))
    for angle, data in equivariant_results.items():
        plt.plot(data['timesteps'], data['mses'], label=f'{angle}° rotation')
    plt.xlabel('Timestep (n_T - index)')
    plt.ylabel('Single-Step MSE between paths')
    plt.title('Single-Step Standard Error on MNIST (Equivariant Model)')
    plt.legend()
    equivariant_plot_path = os.path.join("./equivariant_mnist/", "single_step_equivariant_model.png")
    plt.savefig(equivariant_plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved equivariant model plot to: {equivariant_plot_path}")

    plt.figure(figsize=(12, 8))
    
    # Plot standard model with solid lines
    for angle in rotation_angles:
        if angle in standard_results:
            data = standard_results[angle]
            plt.plot(data['timesteps'], data['mses'], 
                    label=f'Standard {angle}°', 
                    linestyle='-')
    
    # Plot equivariant model with dashed lines
    for angle in rotation_angles:
        if angle in equivariant_results:
            data = equivariant_results[angle]
            plt.plot(data['timesteps'], data['mses'], 
                    label=f'Equivariant {angle}°', 
                    linestyle='--')
    
    plt.xlabel('Timestep (n_T - index)')
    plt.ylabel('Single-Step MSE between paths')
    plt.title('Comparison of Single-Step Equivariance Error: Standard vs. Equivariant Model')
    plt.legend()
    comparison_plot_path = os.path.join("./equivariant_mnist/", "single_step_model_comparison.png")
    plt.savefig(comparison_plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved comparison plot to: {comparison_plot_path}")

