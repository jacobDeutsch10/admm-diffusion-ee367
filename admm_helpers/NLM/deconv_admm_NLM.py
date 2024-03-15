import numpy as np
from tqdm import tqdm
from numpy.fft import fft2, ifft2
from pypher.pypher import psf2otf
from skimage.restoration import denoise_nl_means, estimate_sigma
# credit https://github.com/ZakariaPZ/ADMM-Deconvolution-with-Diffusion-Prior/tree/master
# PyTorch
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def deconv_admm_NLM(b, c, lam, rho, num_iters, searchWindowRadius, sigma, nlmSigma):

    # Blur kernel
    cFT = psf2otf(c, b.shape)
    cTFT = np.conj(cFT)

    # Fourier transform of b
    bFT = fft2(b)

    # initialize x,z,u with all zeros
    x = np.zeros_like(b)
    z = np.zeros_like(b)
    u = np.zeros_like(b)


    # set up NLM model
    averageFilterRadius = int(sigma)

    # pre-compute denominator of x update
    denom = cTFT * cFT + rho 

    for it in tqdm(range(num_iters)):

        # x update - inverse filtering: Fourier multiplications and divisions

        v = z - u
        vFT = fft2(v)
        x = np.real(ifft2((cTFT * bFT + rho*vFT)/denom))

        # z update
        v = x + u

        # v_tensor_denoised = nonlocalmeans(
        #     v, searchWindowRadius, averageFilterRadius, sigma, nlmSigma
        #     )
        
        v_tensor_denoised = denoise_nl_means(
            v, sigma=sigma, fast_mode=True, patch_size=3, patch_distance=6
        )
        z = np.squeeze(v_tensor_denoised)

        # u update
        u = u + x - z

    return x
