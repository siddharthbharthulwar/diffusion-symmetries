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
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import gc
from sklearn.datasets import make_moons

def moons_dataset(n=8000):
    X, _ = make_moons(n_samples=n, random_state=42, noise=0.03)
    X[:, 0] = (X[:, 0] + 0.3) * 2 - 1
    X[:, 1] = (X[:, 1] + 0.3) * 3 - 1
    return torch.from_numpy(X.astype(np.float32))

def circle_dataset(n=8000, radius=1.0, noise=0.025, angle_intervals=None):
    """Creates a dataset of points arranged in a circle with optional noise
    
    Args:
        n (int): Number of points to generate
        radius (float): Radius of the circle
        noise (float): Standard deviation of Gaussian noise to add
        angle_intervals (list of tuples): List of (start_angle, end_angle) in degrees.
            If None or [(0,360)], generates full circle.
            Example: [(0,90), (270,360)] for first and fourth quadrants
    
    Returns:
        torch.Tensor: Tensor of shape (n, 2) containing 2D points
    """
    if angle_intervals is None:
        angle_intervals = [(0, 360)]
        
    # Convert degrees to radians
    angle_intervals_rad = [(start * np.pi / 180, end * np.pi / 180) 
                          for start, end in angle_intervals]
    
    # Calculate total angle coverage
    total_angle = sum(end - start for start, end in angle_intervals_rad)
    
    # Calculate points per interval proportionally
    points_list = []
    for start, end in angle_intervals_rad:
        # Calculate number of points for this interval
        interval_angle = end - start
        n_interval = int(n * interval_angle / total_angle)
        
        # Generate evenly spaced angles for this interval
        theta = torch.linspace(start, end, n_interval)
        
        # Convert to cartesian coordinates
        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)
        
        # Stack into points
        interval_points = torch.stack([x, y], dim=1)
        points_list.append(interval_points)
    
    # Combine all intervals
    points = torch.cat(points_list, dim=0)
    
    # Add random noise
    points = points + torch.randn_like(points) * noise
    
    return points


class SimplePointMLP(nn.Module):
    def __init__(self, point_dim, time_embed_dim=64, hidden_dim=256):  # increased dimensions
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        self.net = nn.Sequential(
            nn.Linear(point_dim + time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),  # added another layer
            nn.SiLU(),
            nn.Linear(hidden_dim, point_dim)
        )

    def forward(self, x, t):
        # Create time embeddings
        t_emb = self.time_embed(t.unsqueeze(-1))
        
        # Concatenate point data with time embedding
        x_t = torch.cat([x, t_emb], dim=-1)
        
        # Predict noise
        return self.net(x_t)

class PointDiffusion:
    def __init__(self, point_dim, n_steps=1000, device="cuda"):
        self.point_dim = point_dim
        self.n_steps = n_steps
        self.device = device
        
        # Define noise schedule (linear beta schedule)
        self.beta = torch.linspace(1e-4, 0.02, n_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        # Create noise prediction network
        self.model = SimplePointMLP(point_dim=point_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def diffuse_points(self, x_0, t):
        """Add noise to points according to diffusion schedule"""
        a_bar = self.alpha_bar[t]
        
        # Reshape for broadcasting
        a_bar = a_bar.view(-1, 1)
        
        # Sample noise
        eps = torch.randn_like(x_0)
        
        # Create noisy points
        x_t = torch.sqrt(a_bar) * x_0 + torch.sqrt(1 - a_bar) * eps
        
        return x_t, eps

    def train_step(self, x_0):
        """Single training step"""
        # Sample random timesteps
        t = torch.randint(0, self.n_steps, (x_0.shape[0],)).to(self.device)
        
        # Add noise to points
        x_t, noise = self.diffuse_points(x_0, t)
        
        # Predict noise
        noise_pred = self.model(x_t, t.float() / self.n_steps)
        
        # Calculate loss
        loss = nn.MSELoss()(noise_pred, noise)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    @torch.no_grad()
    def sample(self, n_points, shape, t=0):
        """Sample new points from the diffusion model"""
        # Start from random noise
        x = torch.randn(n_points, shape).to(self.device)
        
        # Gradually denoise
        for t in range(t, -1, -1):
            t_tensor = torch.ones(n_points).to(self.device) * t
            
            # Predict noise
            eps_theta = self.model(x, t_tensor.float() / self.n_steps)
            
            # Get alpha values for current timestep
            alpha = self.alpha[t]
            alpha_bar = self.alpha_bar[t]
            
            # Sample noise for stochastic sampling (except at t=0)
            z = torch.randn_like(x) if t > 0 else 0
            
            # Update point estimates
            x = 1 / torch.sqrt(alpha) * (
                x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * eps_theta
            ) + torch.sqrt(self.beta[t]) * z
            
        return x

def compute_score_field(diffusion_model: PointDiffusion, 
                       x_range: Tuple[float, float]=(-1.5, 1.5), 
                       y_range: Tuple[float, float]=(-1.5, 1.5),
                       grid_size: int=50,
                       timestep: int=500) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute score vectors over a grid of points.
    """
    # Create grid points
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.stack([xx, yy], axis=-1)
    
    # Convert to tensor
    grid_tensor = torch.from_numpy(grid_points.reshape(-1, 2)).float().to(diffusion_model.device)

    # Ensure timestep is within bounds
    t = torch.ones(grid_tensor.shape[0], device=diffusion_model.device) * (timestep % diffusion_model.n_steps)
    
    # Compute scores using noise prediction
    with torch.no_grad():
        eps_theta = diffusion_model.model(grid_tensor, t.float() / diffusion_model.n_steps)
        alpha_bar_t = diffusion_model.alpha_bar[t.long()]
        scores = -eps_theta / torch.sqrt(1 - alpha_bar_t.view(-1, 1))
    
    # Reshape scores back to grid
    score_x = scores[:, 0].cpu().numpy().reshape(grid_size, grid_size)
    score_y = scores[:, 1].cpu().numpy().reshape(grid_size, grid_size)
    
    return xx, yy, score_x, score_y

def true_score(points, R=5.0, sigma=0.1, eps=1e-12):
    """
    Computes the 'true' score function for a ring-like distribution
    of radius R (centered at the origin), with radial 'width' sigma.

    points: np.ndarray of shape (N, 2) for N (x,y) locations.
    R:      float, the ring radius
    sigma:  float, controls how concentrated the ring is
    eps:    small offset to avoid division by zero at r=0

    Returns: np.ndarray of shape (N, 2) with the score vectors.
    """
    # Radii
    rs = np.sqrt(points[:,0]**2 + points[:,1]**2 + eps)
    
    # The factor (-(r-R)/(sigma^2 * r))
    factors = -(rs - R) / (sigma**2 * rs)
    
    # Multiply pointwise by (x, y)
    score_vals = points * factors.reshape(-1, 1)
    
    return score_vals

def plot_score_field(diffusion_model: PointDiffusion, 
                    data_points: torch.Tensor,
                    timestep: int=500,
                    x_range: Tuple[float, float]=(-3.5, 3.5),
                    y_range: Tuple[float, float]=(-3.5, 3.5),
                    title: str="Score Vector Field") -> plt.Figure:
    """
    Plot the score vector field along with original data points.
    Returns the figure object instead of saving to disk.
    """
    # Compute score field
    xx, yy, score_x, score_y = compute_score_field(diffusion_model, 
                                                  x_range=x_range, 
                                                  y_range=y_range,
                                                  timestep=timestep)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    
    # Plot original data points - but clip them to our desired range
    data_np = data_points.cpu().numpy()
    mask = ((data_np[:, 0] >= x_range[0]) & (data_np[:, 0] <= x_range[1]) &
            (data_np[:, 1] >= y_range[0]) & (data_np[:, 1] <= y_range[1]))
    ax.scatter(data_np[mask, 0], data_np[mask, 1], c='blue', alpha=0.1, label='Data points')
    
    # Plot score vectors
    ax.quiver(xx, yy, score_x, score_y, alpha=0.5, color='red', width=0.005)
    
    # Set axis limits and properties
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title(f'Score Vector Field (t={timestep})')
    ax.legend()
    
    return fig

def create_score_field_animation(diffusion_model: PointDiffusion, 
                               timesteps: list,
                               output_path: str = 'score_field_animation.gif'):
    """
    Create an animated GIF of score fields across different timesteps.
    """
    # Create figure for animation
    fig = plt.figure(figsize=(10, 10))
    
    frames = []
    # Reverse timesteps to go from noise to clean
    for t in tqdm(timesteps[::-1], desc="Generating frames"):
        # Clear previous frame
        plt.clf()
        data_points = diffusion_model.sample(n_points=1000, shape=2, t=t)
        # Create new frame
        fig = plot_score_field(diffusion_model, data_points, timestep=t)
        
        # Convert figure to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        frames.append(image)
        plt.close(fig)
    
    # Save as GIF
    import imageio
    imageio.mimsave(output_path, frames, fps=5)
    print(f"Animation saved to {output_path}")

def compute_score_errors(diffusion_model: PointDiffusion,
                        x_range: Tuple[float, float]=(-3.5, 3.5),
                        y_range: Tuple[float, float]=(-3.5, 3.5),
                        grid_size: int=50,
                        timestep: int=0,
                        R: float=0.8,
                        sigma: float=0.025):
    """
    Compute angle and magnitude errors between true and learned score functions
    
    Returns:
        Tuple containing:
        - xx, yy: Grid coordinates
        - theta: Angle errors
        - delta_mag: Magnitude errors
        - combined: Combined error (with alpha=0.5)
    """
    # Get learned scores on grid
    xx, yy, score_x, score_y = compute_score_field(diffusion_model, 
                                                  x_range=x_range,
                                                  y_range=y_range,
                                                  grid_size=grid_size,
                                                  timestep=timestep)
    
    # Reshape grid points for true score computation
    grid_points = np.stack([xx.flatten(), yy.flatten()], axis=1)
    
    # Get true scores
    true_scores = true_score(grid_points, R=R, sigma=sigma)
    learned_scores = np.stack([score_x.flatten(), score_y.flatten()], axis=1)
    
    # Compute angle error (theta)
    # Handle numerical stability with small scores
    eps = 1e-8
    dot_products = np.sum(true_scores * learned_scores, axis=1)
    true_norms = np.linalg.norm(true_scores, axis=1)
    learned_norms = np.linalg.norm(learned_scores, axis=1)
    
    # Clip for numerical stability
    cos_theta = np.clip(dot_products / (true_norms * learned_norms + eps), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    
    # Compute magnitude error (delta_mag)
    delta_mag = np.abs(learned_norms - true_norms)
    
    # Compute combined error with alpha=0.5
    alpha = 0.997
    combined_error = alpha * theta + (1 - alpha) * delta_mag

    
    # Reshape back to grid
    theta = theta.reshape(grid_size, grid_size)
    delta_mag = delta_mag.reshape(grid_size, grid_size)
    combined_error = combined_error.reshape(grid_size, grid_size)
    
    return xx, yy, theta, delta_mag, combined_error

def plot_score_errors(diffusion_model: PointDiffusion,
                     timestep: int=0,
                     R: float=0.8,
                     sigma: float=0.025, xlim: Tuple[float, float]=(-5.2, 5.2), ylim: Tuple[float, float]=(-5.2, 5.2)):
    """
    Plot angle and magnitude errors between true and learned score functions
    """
    xx, yy, theta, delta_mag, combined = compute_score_errors(
        diffusion_model, timestep=timestep, R=R, sigma=sigma, x_range=xlim, y_range=ylim
    )
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot angle error
    im0 = axes[0].pcolormesh(xx, yy, theta, shading='auto', vmin=0, vmax=4)
    axes[0].set_title('Angle Error (radians)')
    plt.colorbar(im0, ax=axes[0])
    
    # Plot magnitude error
    im1 = axes[1].pcolormesh(xx, yy, delta_mag, shading='auto')
    axes[1].set_title('Magnitude Error')
    plt.colorbar(im1, ax=axes[1])
    
    # Plot combined error
    im2 = axes[2].pcolormesh(xx, yy, combined, shading='auto')
    axes[2].set_title('Combined Error (α=0.5)')
    plt.colorbar(im2, ax=axes[2])
    
    # Set properties for all subplots
    for ax in axes:
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    
    plt.suptitle(f'Score Function Errors at t={timestep}')
    plt.tight_layout()
    return fig

def compute_score_errors_between_models(model1: PointDiffusion,
                                      model2: PointDiffusion,
                                      x_range: Tuple[float, float]=(-3.5, 3.5),
                                      y_range: Tuple[float, float]=(-3.5, 3.5),
                                      grid_size: int=50,
                                      timestep: int=0):
    """
    Compute angle and magnitude errors between two diffusion models' score functions
    
    Returns:
        Tuple containing:
        - xx, yy: Grid coordinates
        - theta: Angle errors
        - delta_mag: Magnitude errors
        - combined: Combined error (with alpha=0.5)
    """
    # Get scores from both models on grid
    xx, yy, score1_x, score1_y = compute_score_field(model1, 
                                                    x_range=x_range,
                                                    y_range=y_range,
                                                    grid_size=grid_size,
                                                    timestep=timestep)
    
    _, _, score2_x, score2_y = compute_score_field(model2,
                                                  x_range=x_range,
                                                  y_range=y_range,
                                                  grid_size=grid_size,
                                                  timestep=timestep)
    
    # Reshape scores
    scores1 = np.stack([score1_x.flatten(), score1_y.flatten()], axis=1)
    scores2 = np.stack([score2_x.flatten(), score2_y.flatten()], axis=1)
    
    # Compute angle error (theta)
    eps = 1e-8
    dot_products = np.sum(scores1 * scores2, axis=1)
    norms1 = np.linalg.norm(scores1, axis=1)
    norms2 = np.linalg.norm(scores2, axis=1)
    
    # Clip for numerical stability
    cos_theta = np.clip(dot_products / (norms1 * norms2 + eps), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    
    # Compute magnitude error (delta_mag)
    delta_mag = np.abs(norms2 - norms1)
    
    # Compute combined error with alpha=0.997
    alpha = 0.997
    combined_error = alpha * theta + (1 - alpha) * delta_mag
    
    # Reshape back to grid
    theta = theta.reshape(grid_size, grid_size)
    delta_mag = delta_mag.reshape(grid_size, grid_size)
    combined_error = combined_error.reshape(grid_size, grid_size)
    
    return xx, yy, theta, delta_mag, combined_error

def plot_score_errors_between_models(model1: PointDiffusion,
                                   model2: PointDiffusion,
                                   timestep: int=0,
                                   xlim: Tuple[float, float]=(-5.2, 5.2),
                                   ylim: Tuple[float, float]=(-5.2, 5.2)):
    """
    Plot angle and magnitude errors between two diffusion models
    """
    xx, yy, theta, delta_mag, combined = compute_score_errors_between_models(
        model1, model2, timestep=timestep, x_range=xlim, y_range=ylim
    )
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot angle error
    im0 = axes[0].pcolormesh(xx, yy, theta, shading='auto', vmin=0, vmax=4)
    axes[0].set_title('Angle Error (radians)')
    plt.colorbar(im0, ax=axes[0])
    
    # Plot magnitude error
    im1 = axes[1].pcolormesh(xx, yy, delta_mag, shading='auto')
    axes[1].set_title('Magnitude Error')
    plt.colorbar(im1, ax=axes[1])
    
    # Plot combined error
    im2 = axes[2].pcolormesh(xx, yy, combined, shading='auto')
    axes[2].set_title('Combined Error (α=0.997)')
    plt.colorbar(im2, ax=axes[2])
    
    # Set properties for all subplots
    for ax in axes:
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    
    plt.suptitle(f'Score Function Differences at t={timestep}')
    plt.tight_layout()
    return fig

# Example usage:
if __name__ == "__main__":
    num_steps = 50
    xlim = (-5.2, 5.2)
    ylim = (-5.2, 5.2)
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    
    # Create full circle dataset
    points_full = circle_dataset(n=1000, radius=5, noise=0.1, angle_intervals=[(0, 360)])
    
    # Create partial circle dataset with 10-degree gaps
    points_partial = circle_dataset(n=1000, radius=5, noise=0.1,
                                  angle_intervals=[(0, 20), (30, 50), (60, 80), (90, 110),
                                                 (120, 140), (150, 170), (180, 200),
                                                 (210, 230), (240, 260), (270, 290),
                                                 (300, 320), (330, 350)])

    # points_partial = circle_dataset(n=1000, radius=5, noise=0.1, angle_intervals=[(0, 160), (180, 340)])
    
    # Initialize two diffusion models
    diffusion_full = PointDiffusion(point_dim=2, n_steps=num_steps, device="cuda")
    diffusion_partial = PointDiffusion(point_dim=2, n_steps=num_steps, device="cuda")
    
    # Train on full circle
    points_full = points_full.cuda()
    for epoch in range(10000):
        loss = diffusion_full.train_step(points_full)
        if epoch % 100 == 0:
            print(f"Full Circle - Epoch {epoch}, Loss: {loss:.6f}")
    
    # Train on partial circle
    points_partial = points_partial.cuda()
    for epoch in range(10000):
        loss = diffusion_partial.train_step(points_partial)
        if epoch % 100 == 0:
            print(f"Partial Circle - Epoch {epoch}, Loss: {loss:.6f}")

    timestep = 48

    # Plot score errors between models
    fig = plot_score_errors_between_models(diffusion_full, diffusion_partial, 
                                         timestep=timestep, xlim=xlim, ylim=ylim)
    plt.savefig(os.path.join(save_dir, "score_errors_between_models.png"), 
                bbox_inches='tight', dpi=300)
    plt.close()

    # Plot score fields for both models
    for model, name in [(diffusion_full, "full"), (diffusion_partial, "partial")]:
        # Generate points
        points = model.sample(n_points=1000, shape=2, t=timestep)
        points_np = points.cpu().numpy()
        
        # Plot score field with points
        xx, yy, score_x, score_y = compute_score_field(model, 
                                                      timestep=timestep,
                                                      x_range=xlim,
                                                      y_range=ylim)
        
        plt.figure(figsize=(10, 10))
        plt.quiver(xx, yy, score_x, score_y, alpha=0.5, color='red', width=0.005)
        plt.scatter(points_np[:, 0], points_np[:, 1], s=10, c='blue', 
                   alpha=0.9, label='Generated Points')
        plt.title(f'{name.capitalize()} Model Score Field with Points (t={timestep})')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"score_field_{name}.png"), 
                    bbox_inches='tight', dpi=300)
        plt.close()
