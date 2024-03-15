
# import packages
import numpy as np
from numpy.fft import fft2, ifft2
from fspecial import fspecial_gaussian_2d
from skimage.metrics import peak_signal_noise_ratio  as psnr
from skimage.transform import resize
from pypher.pypher import psf2otf
import matplotlib.pyplot as plt
import glob
from admm_helpers.TV.deconv_admm_tv import deconv_admm_tv
from admm_helpers.DnCNN.deconv_admm_dncnn import deconv_admm_dncnn, deconv_admm_dncnn_cap
from admm_helpers.DnCNN.DnCNN import DnCNN, DnCNNCaptions
from admm_helpers.bilateral.deconv_admm_bilateral import deconv_admm_bilateral
from admm_helpers.NLM.deconv_admm_NLM import deconv_admm_NLM

import torch

from pathlib import Path
from PIL import Image
import json
from skimage.metrics import structural_similarity as ssim
from classes import IMAGENET2012_CLASSES
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
# noise parameter - standard deviation
def admm_tv_helper(b, rho, lam, num_iters,sigma=0.1):
    x_admm_tv = np.zeros(np.shape(b))
    for it in range(3):
        x_admm_tv[:, :, it] = deconv_admm_tv(b[:, :, it], c, lam, rho, num_iters)
    x_admm_tv = np.clip(x_admm_tv, 0.0, 1.0)
    return x_admm_tv

def admm_bilateral_help(b, rho, lam, num_iters,sigma=0.1):
    x_admm_bilateral = np.zeros(np.shape(b))
    for it in range(3):
        x_admm_bilateral[:, :, it] = deconv_admm_bilateral(b[:, :, it], c, lam, rho, num_iters, sigma, 0.25)
    x_admm_bilateral = np.clip(x_admm_bilateral, 0.0, 1.0)
    return x_admm_bilateral

def admm_NLM_helper(b, rho, lam, num_iters, nlmSigma, searchWindowRadius,sigma=0.1):
    x_admm_NLM = np.zeros(np.shape(b))
    for it in range(3):
        x_admm_NLM[:, :, it] = deconv_admm_NLM(b[:, :, it],  c, lam, rho, num_iters, searchWindowRadius, sigma, nlmSigma)
    x_admm_NLM = np.clip(x_admm_NLM, 0.0, 1.0)
    return x_admm_NLM

model = DnCNN(use_bias=True, hidden_channels=64, hidden_layers=2)
model.load_state_dict(torch.load(f'dncnn_{0.2}_{64}_4.pth'))
def admm_DCNN_helper(b, rho, lam, num_iters, sigma=0.1):

    
    
    x_admm_dncnn = deconv_admm_dncnn(b, c, lam, rho, num_iters,model)
    x_admm_dncnn = np.clip(x_admm_dncnn, 0, 1)
    return x_admm_dncnn

def admm_DCNN_caps_helper(b, cap, lam, rho, num_iters,sigma=0.1):
    model = DnCNNCaptions(use_bias=True, hidden_channels=64, hidden_layers=2)
    model.load_state_dict(torch.load(f'dncnn_cap_{sigma}_{64}_4.pth'))
    x_admm_dncnn = deconv_admm_dncnn_cap(b, cap, c, lam, rho, num_iters,model)
    x_admm_dncnn = np.clip(x_admm_dncnn, 0, 1)
    x_admm_dncnn = resize(x_admm_dncnn, (img_size, img_size))
    return x_admm_dncnn

techniques = {
    "TV": lambda b, sigma: admm_tv_helper(b, 5, 0.025, 75, sigma=sigma),
    "bilateral": lambda b, sigma: admm_bilateral_help(b, 5, 0.025, 5, sigma=sigma),
    "NLM": lambda b, sigma: admm_NLM_helper(b, 5, 0.025, 5, 0.1, 5, sigma=sigma),
    "DnCNN": lambda b, sigma: admm_DCNN_helper(b, 0.5, 0.05, 10, sigma=sigma),
}


# blur kernel

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
        sigma_dict = stats.get(str(sigma), {})
        # Blur kernel
        cFT = psf2otf(c, (img.shape[0], img.shape[1]))
        Afun = lambda x: np.real(ifft2(fft2(x) * cFT))


        # simulated measurements
        b = np.zeros(np.shape(img))
        for it in range(3):
            b[:, :, it] = Afun(img[:, :, it]) + sigma * np.random.randn(img.shape[0], img.shape[1])
        # save out inputs
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        Image.fromarray((b*255).astype(np.uint8)).save(img_dir / f"noisy_{sigma}.png")
        ax[0].imshow(img)
        ax[0].set_title('Original')
        ax[0].axis('off')
        ax[1].imshow(b)
        ax[1].set_title('Noisy')
        ax[1].axis('off')
        plt.savefig(img_dir / f"noisy_{sigma}_comparison.png")

        num_techniques = len(techniques)


        for i, (name, technique) in enumerate(techniques.items()):
            if name in sigma_dict:
                continue
            # run technique
            x = technique(b,sigma)
            # calculate and saveout stats
            psnr_ = psnr(img, x)
            ssim_ = ssim(img, x, channel_axis=-1, data_range= x.max() - x.min())
            Image.fromarray((x*255).astype(np.uint8)).save(img_dir / f"{name}_{sigma}.png")
            sigma_dict[name] = {"psnr": psnr_, "ssim": ssim_}
            print(f"{name} - {psnr_:0.2f}, {ssim_:0.2f}")
            with open(stats_path, "w") as f:
                stats[str(sigma)] = sigma_dict
                json.dump(stats, f, indent=2)


