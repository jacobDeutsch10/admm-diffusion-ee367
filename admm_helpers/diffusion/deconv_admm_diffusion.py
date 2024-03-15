import numpy as np
from tqdm import tqdm
from numpy.fft import fft2, ifft2
from pypher.pypher import psf2otf
import torch
from skimage.transform import resize
from matplotlib import pyplot as plt
import torch

# modified from credit https://github.com/ZakariaPZ/ADMM-Deconvolution-with-Diffusion-Prior/tree/master
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def deconv_admm_guided_diffusion_custom(b, cls, c, lam, rho, num_iters, noise_variance, model, diffusion, diffusion_iters=7, cond_fn=None):

    # initialize x,z,u with all zeros
    x = np.zeros_like(b)
    z = np.zeros_like(b)
    u = np.zeros_like(b)
    v = np.zeros_like(b)

    # set up DDPM
    if cls is not None:
        cls = torch.tensor((cls,)).long().to(device)
        model_kwargs ={"y": cls}
    else:
        model_kwargs = {}

    num_channels = b.shape[2]
    for it in tqdm(range(num_iters)):
        for channel in range(num_channels):

            # Blur kernel
            cFT = psf2otf(c, b[:, :, channel].shape)
            cTFT = np.conj(cFT)

            # Fourier transform of b
            bFT = fft2(b[:, :, channel])

            # pre-compute denominator of x update
            denom = cTFT * cFT + rho

            # x update - inverse filtering: Fourier multiplications and divisions
            v[:, :, channel] = z[:, :, channel] - u[:, :, channel]
            vFT = fft2(v[:, :, channel])
            x[:, :, channel] = np.real(ifft2((cTFT * bFT + rho * vFT) / denom))

        # z update
        v = x + u

        # run diffusion denoiser

        v_resized = resize(v, (256, 256))

        v_tensor_denoised = torch.from_numpy(v_resized[None, :]).permute(0, 3, 1, 2).float().to(device)

        
        for t in range(diffusion_iters,0,-1):
            with torch.no_grad():
                result = diffusion.ddim_sample(
                    model,
                    v_tensor_denoised,
                    torch.tensor((t,)).to(device),
                    clip_denoised=True,
                    model_kwargs=model_kwargs,
                    cond_fn =cond_fn
                )
                v_tensor_denoised = result['sample']

        z = torch.squeeze(v_tensor_denoised).permute(1, 2, 0).cpu().numpy()
        z = resize(z, (v.shape[0], v.shape[1]))


        # u update
        u = u + x - z

    return x