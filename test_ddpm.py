''' 
This script does conditional image generation on MNIST, using a diffusion model

This code is modified from,
https://github.com/cloneofsimo/minDiffusion

Diffusion model is based on DDPM,
https://arxiv.org/abs/2006.11239

The conditioning idea is taken from 'Classifier-Free Diffusion Guidance',
https://arxiv.org/abs/2207.12598

This technique also features in ImageGen 'Photorealistic Text-to-Image Diffusion Modelswith Deep Language Understanding',
https://arxiv.org/abs/2205.11487

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
        
        The index is mapped to both the original image index and the rotation angle.
        """
        # Map idx to original image index and angle index
        mnist_idx = idx // len(self.angles)
        angle_idx = idx % len(self.angles)
        angle = self.angles[angle_idx]
        
        # Get original image and label
        image, label = self.mnist[mnist_idx]
        
        # Apply rotation
        if angle != 0:
            image = transforms.functional.rotate(image, angle)
            
        return image.flatten(), torch.tensor(label)

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
            # this adds on correct residual in case channels have increased
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


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_classes=10):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)

        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7), # otherwise just have 2*n_feat
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

    def forward(self, x, c, t, context_mask):
        # x is (noisy) image, c is context label, t is timestep, 
        # context_mask says which samples to block the context on

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # convert context to one hot embedding
        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
        
        # mask out context if context_mask == 1
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1,self.n_classes)
        context_mask = (-1*(1-context_mask)) # need to flip 0 <-> 1
        c = c * context_mask
        
        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

        up1 = self.up0(hiddenvec)
        # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
        up2 = self.up1(cemb1*up1+ temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2*up2+ temb2, down1)
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


def rotate_image(image, angle):
    return transforms.functional.rotate(image, angle)

class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob).to(self.device)
        
        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample_from_noise(self, n_sample, size, device, guide_w, x_i, timesteps):
        with torch.no_grad():
            c_i = torch.arange(0,10).to(device) # context for us just cycles throught the mnist labels
            c_i = c_i.repeat(int(n_sample/c_i.shape[0]))

            # don't drop context at test time
            context_mask = torch.zeros_like(c_i).to(device)

            # double the batch
            c_i = c_i.repeat(2)
            context_mask = context_mask.repeat(2)
            context_mask[n_sample:] = 1. # makes second half of batch context free

            x_i_store = [] # keep track of generated steps in case want to plot something 
            print()
            for i in range(timesteps, 0, -1):
                print(f'sampling timestep {i}',end='\r')
                t_is = torch.tensor([i / timesteps]).to(device)
                t_is = t_is.repeat(n_sample,1,1,1)

                # double batch
                x_i = x_i.repeat(2,1,1,1)
                t_is = t_is.repeat(2,1,1,1)

                z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

                # split predictions and compute weighting
                eps = self.nn_model(x_i, c_i, t_is, context_mask)
                eps1 = eps[:n_sample]
                eps2 = eps[n_sample:]
                eps = (1+guide_w)*eps1 - guide_w*eps2
                x_i = x_i[:n_sample]
                x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
                )
                if i%20==0 or i==timesteps or i<8:
                    x_i_store.append(x_i.detach().cpu().numpy())
        
            x_i_store = np.array(x_i_store)
            return x_i, x_i_store

    def sample_from_noise_with_transformation(self, n_sample, size, device, guide_w, transformation_fn, transformation_idx, x_i):
        with torch.no_grad():
            c_i = torch.arange(0,10).to(device)
            c_i = c_i.repeat(int(n_sample/c_i.shape[0]))
            context_mask = torch.zeros_like(c_i).to(device)

            # double the batch
            c_i = c_i.repeat(2)
            context_mask = context_mask.repeat(2)
            context_mask[n_sample:] = 1.

            x_i_store = []
            print()  # Add blank line before progress starts
            for i in range(self.n_T, 0, -1):
                print(f'sampling timestep {i}', end='\r')

                t_is = torch.tensor([i / self.n_T]).to(device)
                t_is = t_is.repeat(n_sample,1,1,1)

                if transformation_fn is not None and transformation_idx is not None:
                    if i == transformation_idx:
                        x_i = transformation_fn(x_i)

                # double batch
                x_i_doubled = x_i.repeat(2,1,1,1)
                t_is = t_is.repeat(2,1,1,1)

                z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

                # split predictions and compute weighting
                eps = self.nn_model(x_i_doubled, c_i, t_is, context_mask)
                eps1 = eps[:n_sample]
                eps2 = eps[n_sample:]
                eps = (1+guide_w)*eps1 - guide_w*eps2

                x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
                )

                if i%20==0 or i==self.n_T or i<8:
                    x_i_store.append(x_i.detach().cpu().numpy())

                # Clean up temporary tensors
                del x_i_doubled, eps, eps1, eps2, t_is
                if i > 1:
                    del z
                torch.cuda.empty_cache()
                
            x_i_store = np.array(x_i_store)
            return x_i, x_i_store

    def sample(self, n_sample, size, device, guide_w = 0.0):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        c_i = torch.arange(0,10).to(device) # context for us just cycles throught the mnist labels
        c_i = c_i.repeat(int(n_sample/c_i.shape[0]))

        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(device)

        # double the batch
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1. # makes second half of batch context free

        x_i_store = [] # keep track of generated steps in case want to plot something 
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)

            # double batch
            x_i = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store

    def sample_with_transforation(self, n_sample, size, device, guide_w = 0.0, transformation_fn = None, transformation_idx = None):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        c_i = torch.arange(0,10).to(device) # context for us just cycles throught the mnist labels
        c_i = c_i.repeat(int(n_sample/c_i.shape[0]))

        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(device)

        # double the batch
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1. # makes second half of batch context free

        x_i_store = [] # keep track of generated steps in case want to plot something 
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)

            if transformation_fn is not None and transformation_idx is not None:
                if i == transformation_idx:
                    x_i = transformation_fn(x_i)

            # double batch
            x_i = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store

    def create_animation(self, transformation_fn, transformation_idx):

        save_dir = './figures/animations/'

        ddpm.eval()
        ws_test = [0.5]
        with torch.no_grad():
            n_sample = 4*n_classes
            for w_i, w in enumerate(ws_test): #TODO: change to ws_test[0] for no guidance
                x_i = torch.randn(n_sample, 1, 28, 28).to(device)
                x_gen, x_gen_store = ddpm.sample_from_noise_with_transformation(n_sample, (1, 28, 28), device, guide_w=w, transformation_fn=transformation_fn, transformation_idx=transformation_idx, x_i=x_i)

                x_gen_real, x_gen_real_store = ddpm.sample_from_noise(n_sample, (1, 28, 28), device, guide_w=w, x_i=x_i)
                # x_gen, x_gen_store = ddpm.(n_sample, (1, 28, 28), device, guide_w=w)

            fig, axs = plt.subplots(nrows=int(n_sample/n_classes), ncols=n_classes,sharex=True,sharey=True,figsize=(8,3))

            for row in range(int(n_sample/n_classes)):
                for col in range(n_classes):
                    # Convert numpy array to tensor with correct dimensions for rotation
                    img = torch.tensor(x_gen_real[row*n_classes+col, 0].detach().cpu().numpy()).unsqueeze(0)
                    rotated_img = transformation_fn(img)
                    axs[row, col].imshow(rotated_img.squeeze().numpy(), cmap='gray')
                    axs[row, col].axis('off')

            plt.savefig(save_dir + f"ddpm_animation_{transformation_idx}_real.png")

            fig, axs = plt.subplots(nrows=int(n_sample/n_classes), ncols=n_classes,sharex=True,sharey=True,figsize=(8,3))
            def animate_diff(i, x_gen_store):
                print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                plots = []
                for row in range(int(n_sample/n_classes)):
                    for col in range(n_classes):
                        axs[row, col].clear()
                        axs[row, col].set_xticks([])
                        axs[row, col].set_yticks([])
                        plots.append(axs[row, col].imshow(x_gen_store[i,(row*n_classes)+col,0],cmap='gray'))
                return plots

            ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store],  interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0])    
            ani.save(save_dir + f"ddpm_animation_{transformation_idx}.gif", dpi=100, writer=PillowWriter(fps=5))
            print('saved image at ' + save_dir + f"ddpm_animation_{transformation_idx}.gif")

def print_gpu_memory():
    device_idx = 1  # Since you're using cuda:1
    print(f'Allocated: {torch.cuda.memory_allocated(device_idx)/1024**2:.1f}MB')
    print(f'Cached: {torch.cuda.memory_reserved(device_idx)/1024**2:.1f}MB')
    print(f'Max allocated: {torch.cuda.max_memory_allocated(device_idx)/1024**2:.1f}MB')
    print(f'Max cached: {torch.cuda.max_memory_reserved(device_idx)/1024**2:.1f}MB')

def transformation_fn(x):
    return rotate_image(x, 90)

if __name__ == "__main__":
    # Must match training parameters exactly
    n_feat = 512  
    n_classes = 10
    n_T = 400  # Initialize with original n_T
    device = "cuda:1"
    ws_test = [0.0, 0.5, 2.0]

    # Memory management
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    
    print("Initial GPU memory:")
    print_gpu_memory()
    
    # Create model on CPU first
    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), 
                betas=(1e-4, 0.02), n_T=n_T, device='cpu', drop_prob=0.1)
    
    # Load weights on CPU
    ddpm.load_state_dict(torch.load("./data/diffusion_outputs10/model_24.pth", 
                                   map_location='cpu'))
    
    # Now we can modify n_T after loading
    ddpm.n_T = 400  # Change to desired smaller value
    
    # Move to GPU
    ddpm = ddpm.to(device)
    
    print("After moving to GPU:")
    print_gpu_memory()
    ddpm.eval()
    # Generate initial noise once and reuse
    x_i = torch.randn(n_classes, 1, 28, 28).to(device)
    n_samples = n_classes
    angles = [36 * i for i in range(9)]  # 0, 36, 72, ..., 288 degrees

    for transformation_idx in range(0, 401, 25):
        ddpm.create_animation(transformation_fn, transformation_idx)
    
    # # Get baseline samples (no rotation) - only need to do this once
    # x_baseline, x_baseline_store = ddpm.sample_from_noise(n_samples, (1, 28, 28), device, guide_w=ws_test[1], x_i=x_i)
    
    # # Loop through different transformation indices
    # total_indices = len(range(0, 401, 25))
    # for idx, transform_idx in enumerate(range(0, 401, 25)):
    #     print(f"\n[{idx + 1}/{total_indices}] Processing transformation index {transform_idx}")
    #     rotated_samples = []
        
    #     # Generate samples with different rotation angles at current transform_idx
    #     for angle_idx, angle in enumerate(angles):
    #         print(f"  - Generating samples for angle {angle}° ({angle_idx + 1}/{len(angles)})")
    #         def rotation_fn(x):
    #             return rotate_image(x, angle)
                
    #         x_rot, x_rot_store = ddpm.sample_from_noise_with_transformation(
    #             n_samples, (1, 28, 28), device, 
    #             guide_w=ws_test[1],
    #             transformation_fn=rotation_fn,
    #             transformation_idx=transform_idx,
    #             x_i=x_i
    #         )
    #         rotated_samples.append(x_rot)
            
    #         # Clear some memory
    #         torch.cuda.empty_cache()
    #         gc.collect()

    #     print(f"  Saving grid for transformation index {transform_idx}")
    #     # Create grid plot
    #     fig, axs = plt.subplots(10, n_samples, figsize=(2*n_samples, 20))
        
    #     # Plot baseline (no rotation) in first row
    #     for j in range(n_samples):
    #         axs[0, j].imshow(x_baseline[j, 0].detach().cpu().numpy(), cmap='gray')
    #         axs[0, j].axis('off')
    #         if j == 0:
    #             axs[0, j].set_ylabel('No rotation', rotation=0, labelpad=40)
        
    #     # Plot rotated versions in remaining rows
    #     for i, (angle, x_rot) in enumerate(zip(angles, rotated_samples), 1):
    #         for j in range(n_samples):
    #             axs[i, j].imshow(x_rot[j, 0].detach().cpu().numpy(), cmap='gray')
    #             axs[i, j].axis('off')
    #             if j == 0:
    #                 axs[i, j].set_ylabel(f'Rotation {angle}°', rotation=0, labelpad=40)
        
    #     # Add title showing transformation index
    #     plt.suptitle(f'Transformation applied at step {transform_idx}', y=1.02, fontsize=16)
        
    #     plt.tight_layout()
    #     plt.savefig(f'./figures/rotation_grid{transform_idx}.png', bbox_inches='tight')
    #     plt.close()
        
    #     # Clear memory after each grid
    #     del rotated_samples
    #     torch.cuda.empty_cache()
    #     gc.collect()