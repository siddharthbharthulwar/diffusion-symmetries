import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import gc
from train_equivariant_mnist import MNISTDiffusion
import matplotlib.pyplot as plt

def rotate_image(images, angle):
    """
    Rotate a batch of images
    Args:
        images: Tensor of shape [batch_size, channels, height, width]
        angle: Rotation angle in degrees
    """
    return torch.stack([transforms.functional.rotate(img, angle) for img in images])

@torch.no_grad()
def sample_steps(model, x_i, n_sample, size, start_timestep, end_timestep):
    """
    Sample from the model for a specific range of timesteps.
    
    Args:
        model (MNISTDiffusion): The diffusion model
        x_i (torch.Tensor): Initial latent state to start sampling from. If None, starts from random noise
        n_sample (int): Number of samples to generate
        size (tuple): Size of each sample (channels, height, width)
        start_timestep (int): Starting timestep (must be less than end_timestep)
        end_timestep (int): Ending timestep
        
    Returns:
        tuple: (final latent tensor, list of intermediate latents)
    """
    device = model.device
    # If no initial latent provided, start from random noise
    if x_i is None:
        x_i = torch.randn(n_sample, *size).to(device)
    else:
        x_i = x_i.to(device)
    
    x_i_store = []
    
    for i in range(end_timestep, start_timestep - 1, -1):
        print(f'sampling timestep {i}', end='\r')
        t_is = torch.tensor([i / model.n_T]).to(device)
        t_is = t_is.repeat(n_sample)
        
        # Add noise only if not at the final step
        z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
        
        # Get model prediction and update x_i
        eps = model.nn_model(x_i, t_is)
        x_i = (
            model.oneover_sqrta[i] * (x_i - eps * model.mab_over_sqrtmab[i])
            + model.sqrt_beta_t[i] * z
        )
        x_i_store.append(x_i.detach().cpu().numpy())
    
    return x_i, x_i_store

def load_diffusion_model(model_path, device="cuda:0"):
    """
    Load a trained diffusion model from a .pth file
    
    Args:
        model_path (str): Path to the .pth model file
        device (str): Device to load the model on ('cuda:0' or 'cpu')
        
    Returns:
        MNISTDiffusion: Loaded model
    """
    # Initialize model with same hyperparameters used during training
    model = MNISTDiffusion(
        in_channels=1,
        n_feat=n_feat,  # Using n_feat from outer scope
        betas=(1e-4, 0.02),
        n_T=n_T,  # Using n_T from outer scope
        device=device,
        lrate=lrate  # Using lrate from outer scope
    ).to(device)
    
    # Load the state dict
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None
    
    # Set to eval mode by default
    model.eval()
    return model

if __name__ == "__main__":
    # Hyperparameters
    torch.cuda.empty_cache()
    n_epoch = 10
    batch_size = 128
    n_T = 400
    n_feat = 512
    lrate = 1e-4
    lambda_equiv = 0.0   # adjust if you want rotation equiv. to matter more
    angles = [10 * i for i in range(10)]  # rotation angles for the dataset

    rotation_angle = 90

    # Load the model
    model_path = "./equivariant_mnist/equivariant_0/model_199.pth"
    ddpm = load_diffusion_model(model_path, device="cuda:0")

    ddpm.eval()

    indices = [i * 10 for i in range(40)]
    mses = []
    n_sample = 4  # number of samples to generate
    size = (1, 28, 28)

    for INDEX in indices:
        print("running for index", INDEX)

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        print("generating intermediate")
        # Sample from timestep 400 down to INDEX
        initial_noise = torch.randn(n_sample, *size).to(ddpm.device)
        intermediate_i, intermediate_store = sample_steps(
            model=ddpm,
            x_i=initial_noise,
            n_sample=n_sample,
            size=size,
            start_timestep=INDEX,
            end_timestep=400
        )

        print("generating original")
        # Continue sampling from INDEX to 1
        x_i, x_i_store = sample_steps(
            model=ddpm,
            x_i=intermediate_i,  # Use the intermediate result
            n_sample=n_sample,
            size=size,
            start_timestep=1,
            end_timestep=INDEX
        )

        x_i = rotate_image(x_i, rotation_angle)

        print("generating transformed")
        # First rotate the intermediate result
        transformed_i = rotate_image(intermediate_i, rotation_angle)
        # Then continue sampling
        x_i_transformed, x_i_transformed_store = sample_steps(
            model=ddpm,
            x_i=transformed_i,  # Use the rotated intermediate result
            n_sample=n_sample,
            size=size,
            start_timestep=1,
            end_timestep=INDEX
        )

        with torch.no_grad():
            loss = nn.MSELoss()(x_i, x_i_transformed)
            mses.append(loss.cpu().item())

    plt.plot([ddpm.n_T - i for i in indices], mses)
    plt.xlabel('Timestep')
    plt.ylabel('MSE between paths')
    plt.savefig(f'./equivariant_mnist/{rotation_angle}_nonequivariant.png', 
                bbox_inches='tight', dpi=300)
    plt.close()



    