import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import matplotlib.pyplot as plt

# -----------------------
# 1. CONFIG
# -----------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 2000

# Diffusion hyperparameters (toy schedule)
T_MAX = 100  # discrete timesteps for the toy forward process
BETA_START = 1e-4
BETA_END = 0.02
betas = torch.linspace(BETA_START, BETA_END, T_MAX).to(DEVICE)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# -----------------------
# 2. SIMPLE DATASET: CIRCLE
# -----------------------
# Generate points on a circle of radius ~1 (with small random radius variation, if desired)
def sample_circle_data(batch_size):
    angles = 2 * math.pi * torch.rand(batch_size)
    x = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    return x  # shape: (batch_size, 2)

# -----------------------
# 3. ROTATION UTILITY
# -----------------------
def random_2d_rotation(x):
    """
    x: (batch_size, 2), a batch of 2D points.
    Returns x rotated by a random angle (one angle per batch).
    """
    # One random angle per batch
    angles = 2 * math.pi * torch.rand(x.shape[0], device=x.device)
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)
    # Rotation matrix for each batch entry
    R = torch.stack([
        torch.stack([ cos_vals, -sin_vals], dim=1),
        torch.stack([ sin_vals,  cos_vals], dim=1)
    ], dim=1)  # shape: (batch_size, 2, 2)

    # x shape: (batch_size, 2) -> expand to (batch_size, 2, 1)
    x_expanded = x.unsqueeze(2)
    # Rotated = R * x
    x_rot = torch.bmm(R, x_expanded).squeeze(2)
    return x_rot, R

def apply_rotation(x, R):
    """
    Apply a precomputed rotation matrix R to x.
    x: (batch_size, 2)
    R: (batch_size, 2, 2)
    """
    x_expanded = x.unsqueeze(2)
    x_rot = torch.bmm(R, x_expanded).squeeze(2)
    return x_rot

# -----------------------
# 4. FORWARD NOISING
# -----------------------
def q_sample(x0, t):
    """
    x0: (batch_size, 2)
    t: (batch_size,) integer timesteps in [0, T_MAX-1]
    Return: noisy sample x_t according to
      x_t = sqrt(alpha_cumprod_t)*x0 + sqrt(1 - alpha_cumprod_t)*eps
    """
    alphas_cumprod_t = alphas_cumprod[t].unsqueeze(1)  # shape: (batch_size, 1)
    eps = torch.randn_like(x0)
    return (torch.sqrt(alphas_cumprod_t) * x0 +
            torch.sqrt(1 - alphas_cumprod_t) * eps)

# -----------------------
# 5. MODEL: A tiny MLP to predict noise
# -----------------------
class DenoiserMLP(nn.Module):
    """
    Denoiser that takes (x_t, t) as input and predicts the noise eps that was added.
    We'll encode t with a simple positional embedding or just an MLP on t.
    """
    def __init__(self, hidden_dim=64, max_t=T_MAX):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Embedding for t
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # MLP for x
        self.x_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Merge & out
        self.merge = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2)  # predict 2D noise
        )

    def forward(self, x, t):
        # x: (batch_size, 2), t: (batch_size,)
        t_embed = self.time_mlp(t.unsqueeze(1).float())
        x_embed = self.x_mlp(x)
        h = x_embed + t_embed
        return self.merge(h)

# Instantiate model
model = DenoiserMLP(hidden_dim=64).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -----------------------
# 6. TRAINING LOOP
# -----------------------
def training_loop():
    model.train()
    for epoch in range(EPOCHS):
        # 1) Sample data
        x0 = sample_circle_data(BATCH_SIZE).to(DEVICE)  # (B,2)
        # 2) Pick random t
        t = torch.randint(0, T_MAX, (BATCH_SIZE,), device=DEVICE)

        # 3) Forward noising
        x_t = q_sample(x0, t)  # (B,2)
        # 4) Predict noise
        eps_pred = model(x_t, t)  # (B,2)

        # 5) Compute standard noise-prediction loss (MSE)
        #    We can "reconstruct" the noise if we know how x_t was formed:
        #    x_t = sqrt(alpha_cumprod_t)*x0 + sqrt(1 - alpha_cumprod_t)*eps
        #    so eps = (x_t - sqrt(alpha_cumprod_t)*x0)/sqrt(1 - alpha_cumprod_t).
        alpha_cumprod_t = alphas_cumprod[t].unsqueeze(1)
        eps_target = (x_t - torch.sqrt(alpha_cumprod_t)*x0) / torch.sqrt(1 - alpha_cumprod_t)
        loss_mse = ((eps_pred - eps_target)**2).mean()

        # 6) Compute rotational equivariance loss
        #    We want: model(R(x_t), t) ~ R(model(x_t, t)).
        x_t_rot, R = random_2d_rotation(x_t)  # rotated inputs, plus their rotation matrices
        eps_pred_rot = model(x_t_rot, t)     # (B,2)
        # Also rotate the original eps_pred
        eps_pred_rot_expected = apply_rotation(eps_pred, R)
        loss_rot = ((eps_pred_rot - eps_pred_rot_expected)**2).mean()

        # Combine
        loss = loss_mse + 0.0 * loss_rot  # Weighted combination

        # 7) Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 8) Logging
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f} | "
                  f"Loss MSE: {loss_mse.item():.4f}, Loss Rot: {loss_rot.item():.4f}")

# -----------------------
# 7. RUN TRAINING
# -----------------------
if __name__ == "__main__":
    training_loop()

    # After training, let's do a small visualization:
    model.eval()
    with torch.no_grad():
        # Sample some points from the circle
        x0_vis = sample_circle_data(200).to(DEVICE)
        t_vis = torch.randint(0, T_MAX, (200,), device=DEVICE)
        x_t_vis = q_sample(x0_vis, t_vis)
        eps_pred_vis = model(x_t_vis, t_vis)
        # Reconstructed x0_pred = (x_t - sqrt(1 - alpha_cumprod_t)*eps_pred) / sqrt(alpha_cumprod_t)
        alpha_cumprod_tv = alphas_cumprod[t_vis].unsqueeze(1)
        x0_pred_vis = (x_t_vis - torch.sqrt(1 - alpha_cumprod_tv)*eps_pred_vis) / torch.sqrt(alpha_cumprod_tv)

    x0_vis_np = x0_vis.cpu().numpy()
    x0_pred_vis_np = x0_pred_vis.cpu().numpy()

    plt.figure(figsize=(6,6))
    plt.scatter(x0_vis_np[:,0], x0_vis_np[:,1], c='blue', alpha=0.5, label='True x0')
    plt.scatter(x0_pred_vis_np[:,0], x0_pred_vis_np[:,1], c='red', alpha=0.5, label='Reconstructed x0')
    plt.legend()
    plt.title("Toy Diffusion Model with Rotational Equivariance Penalty")
    plt.axis('equal')
    plt.show()
