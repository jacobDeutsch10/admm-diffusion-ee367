# %%
from admm_helpers.DnCNN.DnCNN import *

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import glob
import numpy as np
class ImageOnlyDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        
        if self.transform:
            image = self.transform(image.convert("RGB"))
        return image, torch.tensor([])

image_paths =sorted(glob.glob("imagenet_val/*.JPEG"))
image_paths_train = image_paths[int(len(image_paths)*0.4):]
image_paths_val = image_paths[:int(len(image_paths)*0.4)]
img_size=256
train_transforms = transforms.Compose(
    [
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(img_size) ,
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)
val_transforms = transforms.Compose(
    [
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(img_size) ,
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def plot_summary(idx, model, sigma, losses, psnrs, baseline_psnrs,
                 val_losses, val_psnrs, val_iters, train_dataset,
                 val_dataset, val_dataloader):

    with torch.no_grad():
        model.eval()

        # evaluate on training dataset sample
        train_dataset.use_patches = False
        train_image, cap = train_dataset[0]
        train_image = train_image[None, ...].to(device)
        cap = torch.tensor(cap)[None, ...].long().to(device)
        train_dataset.use_patches = True

        noisy_train_image = add_noise(train_image, sigma=sigma)
        denoised_train_image = model(noisy_train_image)

        # evaluate on validation dataset sample
        val_dataset.use_patches = False
        val_image, _= val_dataset[6]
        val_image = val_image[None, ...].to(device)
        val_dataset.use_patches = True
        val_patch_samples, val_cap_samples = next(iter(val_dataloader))
        val_patch_samples = val_patch_samples.to(device)
        val_cap_samples = val_cap_samples.long().to(device)
        # calculate validation metrics
        noisy_val_patch_samples = add_noise(val_patch_samples, sigma=sigma)
        denoised_val_patch_samples = model(noisy_val_patch_samples)
        val_loss = torch.mean((val_patch_samples - denoised_val_patch_samples)**2)
        val_psnr = calc_psnr(denoised_val_patch_samples, val_patch_samples)

        val_losses.append(val_loss.item())
        val_psnrs.append(val_psnr)
        val_iters.append(idx)

        noisy_val_image = add_noise(val_image, sigma=sigma)
        denoised_val_image = model(noisy_val_image)

    plt.clf()
    plt.figure(figsize=(12, 6))
    plt.subplot(241)
    plt.plot(losses, label='Train loss')
    plt.plot(val_iters, val_losses, '.', label='Val. loss')
    plt.yscale('log')
    plt.legend()
    plt.title('loss')

    plt.subplot(245)
    plt.plot(psnrs, label='Train PSNR')
    plt.plot(val_iters, val_psnrs, '.', label='Val. PSNR')
    plt.plot(baseline_psnrs, label='Baseline PSNR')
    plt.ylim((0, 32))
    plt.legend()
    plt.title('psnr')

    plt.subplot(242)
    plt.imshow(img_to_numpy(train_image))
    plt.ylabel('Training Set')
    plt.title('GT')

    plt.subplot(243)
    plt.imshow(img_to_numpy(noisy_train_image))
    plt.title('Noisy Image')

    plt.subplot(244)
    plt.imshow(img_to_numpy(denoised_train_image))
    plt.title('Denoised Image')

    plt.subplot(246)
    plt.imshow(img_to_numpy(val_image))
    plt.ylabel('Validation Set')
    plt.title('GT')

    plt.subplot(247)
    plt.imshow(img_to_numpy(noisy_val_image))
    plt.title('Noisy Image')

    plt.subplot(248)
    plt.imshow(img_to_numpy(denoised_val_image))
    plt.title('Denoised Image')
    plt.tight_layout()
    plt.pause(0.1)
    plt.show()

def train(sigma=0.1, use_bias=True, hidden_channels=32, epochs=2, batch_size=32, plot_every=20):

    print(f'==> Training on noise level {sigma:.02f} | use_bias: {use_bias} | hidden_channels: {hidden_channels}')

    # create datasets
    train_dataset = ImageOnlyDataset(image_paths_train, transform=train_transforms)
    val_dataset = ImageOnlyDataset(image_paths_val, transform=val_transforms)

    # create dataloaders & seed for reproducibility
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = DnCNN(use_bias=use_bias, hidden_channels=hidden_channels, hidden_layers=2).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=2e-3)

    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=20, gamma=0.5)

    losses = []
    psnrs = []
    baseline_psnrs = []
    val_losses = []
    val_psnrs = []
    val_iters = []
    idx = 0

    pbar = tqdm(total=len(train_dataset) * epochs // batch_size)
    for epoch in range(epochs):
        for sample, _ in train_dataloader:

            model.train()
            sample = sample.to(device)

            # add noise
            noisy_sample = add_noise(sample, sigma=sigma)

            # denoise
            denoised_sample = model(noisy_sample)

            # loss function
            loss = torch.mean((denoised_sample - sample)**2)
            psnr = calc_psnr(denoised_sample, sample)
            baseline_psnr = calc_psnr(noisy_sample, sample)

            losses.append(loss.item())
            psnrs.append(psnr)
            baseline_psnrs.append(baseline_psnr)

            # update model
            loss.backward()
            optim.step()
            optim.zero_grad()

            # plot results
            if not idx % plot_every:
                plot_summary(idx, model, sigma, losses, psnrs, baseline_psnrs,
                             val_losses, val_psnrs, val_iters, train_dataset,
                             val_dataset, val_dataloader)

            idx += 1
            pbar.update(1)
        scheduler.step()

    pbar.close()
    return model
def evaluate_model(model, sigma=0.1, output_filename='out.png'):
    dataset = ImageOnlyDataset(image_paths_val, transform=val_transforms)
    model.eval()

    psnrs = []
    for idx, (image,_ )in enumerate(dataset):
        image = image[None, ...].to(device)  # add batch dimension
        noisy_image = add_noise(image, sigma)
        denoised_image = model(noisy_image)
        psnr = calc_psnr(denoised_image, image)
        psnrs.append(psnr)

        # include the tiger image in your homework writeup
        if idx == 6:
            skimage.io.imsave(output_filename, (img_to_numpy(denoised_image)*255).astype(np.uint8))

    return np.mean(psnrs)


for sigma in [0.2, 0.1, 0.01]:
    model = train(sigma=sigma, use_bias=True, hidden_channels=64, epochs=4, batch_size=32, plot_every=500)
    print(evaluate_model(model, sigma=sigma, output_filename='out.png'))
    torch.save(model.state_dict(), f'dncnn_{sigma}_{64}_{4}.pth')


