# ADMM-Based Image Deconvolution with Conditioned Diffusion Prior
### EE367 Final Project
#### Jacob Deutsch (jadeuts@stanford.edu)

## Install

1. Install pip dependencies
```bash
pip install -r requirements.txt
```
2. Install guided_diffusion
```bash
    git clone https://github.com/openai/guided-diffusion.git
    cd guided-diffusion
    pip install -e .
    cd ..
```
3. Download files
    - Download [ImageNet validation](https://huggingface.co/datasets/imagenet-1k/tree/main/data) set and unzip into a directory named `imagenet_val`
    - Download 256x256 conditional and unconditional diffusion models from [guided-diffusion](https://github.com/openai/guided-diffusion/tree/main)

## Recreating Results
Run the following scripts
```bash
# DnCNN and Traditional Priors
python main.py

# unconditional diffusion
python main_diffusion.py

# conditional diffusion
python main_diffusion_cond.py
```
Tables and figures can be generated using `results.ipynb`

Credit to [Deconvolution using ADMM with Diffusion Denoising Prior](https://github.com/ZakariaPZ/ADMM-Deconvolution-with-Diffusion-Prior/tree/master) as I heavily leveraged their codebase.