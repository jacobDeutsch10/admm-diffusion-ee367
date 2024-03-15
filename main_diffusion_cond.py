
# import packages
import numpy as np
from numpy.fft import fft2, ifft2
from fspecial import fspecial_gaussian_2d
from skimage.metrics import peak_signal_noise_ratio  as psnr
from pypher.pypher import psf2otf
import matplotlib.pyplot as plt
import glob
from admm_helpers.diffusion.deconv_admm_diffusion import  deconv_admm_guided_diffusion_custom
import torch
from pathlib import Path
from PIL import Image
import json
from skimage.metrics import structural_similarity as ssim
from classes import IMAGENET2012_CLASSES
from guided_diffusion_models import create_256_cond, create_256_uncond

def get_class_from_fname(fname):
    keys = list(IMAGENET2012_CLASSES.keys())
    hsh = fname.split("_")[-1].split(".")[0]
    
    class_name = IMAGENET2012_CLASSES[hsh]
    idx = keys.index(hsh)
    return class_name, idx
results = Path("results")
results.mkdir(exist_ok=True)


images = sorted(list(glob.glob('imagenet_val/*.JPEG')))
img_size=256
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

kernel_size = 15 
kernel_sigma = 1.25 
c = fspecial_gaussian_2d((kernel_size, kernel_size), kernel_sigma)

name = "Guided Diffusion"
# blur kernel
model, diffusion = create_256_cond()
lam =0.05
rho = 0.5
num_iters = 10
for idx, img_path in enumerate(images[:20]):
    img_dir = results / f"{idx}"
    img_dir.mkdir(exist_ok=True)
    img = Image.open(img_path)
    img.save(img_dir / "GT.png")
    _, caption = get_class_from_fname(img_path)
    img = np.array(img).astype(float)/255
    stats = {}
    stats_path = img_dir / "stats.json"
    if stats_path.exists():
        with open(stats_path, "r") as f:
            stats = json.load(f)
    
    for sigma in [0.2, 0.1, 0.01,]:
        diffusion_iters = np.argmin(np.abs(1 - diffusion.alphas_cumprod -sigma**2))
        sigma_dict = stats.get(str(sigma), {})
        # Blur kernel
        cFT = psf2otf(c, (img.shape[0], img.shape[1]))
        Afun = lambda x: np.real(ifft2(fft2(x) * cFT))
        if name in sigma_dict:
           continue

        # simulated measurements
        b = np.zeros(np.shape(img))
        for it in range(3):
            b[:, :, it] = Afun(img[:, :, it]) + sigma * np.random.randn(img.shape[0], img.shape[1])
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        Image.fromarray((b*255).astype(np.uint8)).save(img_dir / f"noisy_{sigma}.png")
        ax[0].imshow(img)
        ax[0].set_title('Original')
        ax[0].axis('off')
        ax[1].imshow(b)
        ax[1].set_title('Noisy')
        ax[1].axis('off')

        x = deconv_admm_guided_diffusion_custom(b, caption, c, lam, rho, num_iters, sigma, model, diffusion, diffusion_iters)
        x = np.clip(x, 0.0, 1.0)


        
        
        
        psnr_ = psnr(img, x)
        ssim_ = ssim(img, x, channel_axis=-1, data_range= x.max() - x.min())
        Image.fromarray((x*255).astype(np.uint8)).save(img_dir / f"{name}_{sigma}.png")
        sigma_dict[name] = {"psnr": psnr_, "ssim": ssim_}
        print(f"{name} - {psnr_:0.2f}, {ssim_:0.2f}")
        with open(stats_path, "w") as f:
           stats[str(sigma)] = sigma_dict
           json.dump(stats, f, indent=2)


