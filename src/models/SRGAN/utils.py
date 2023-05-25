import torch
import os
import config
import numpy as np
from PIL import Image
from torchvision.utils import save_image
from math import log10, sqrt

repo_path = os.environ.get('THESIS_PATH')

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 1.0
    psnr = 20 * log10((max_pixel / sqrt(mse)))
    return psnr

def gradient_penalty(critic, real, fake, device):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake.detach() * (1 - alpha)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr, flag):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    if flag:
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        model.load_state_dict(checkpoint)

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def plot_examples(low_res_folder, gen, flag):
    files = os.listdir(low_res_folder)

    gen.eval()
    for file in files:
        image = Image.open(low_res_folder + file)
        with torch.no_grad():
            lr_im = (config.lowres_transform(image=np.asarray(image))["image"]
                    .unsqueeze(0)
                    .to(config.DEVICE))
            
            upscaled_img = gen(lr_im)

        if flag:
            save_image(upscaled_img * 0.5 + 0.5, f"{repo_path}/src/models/saved/{file}")
        else:
            save_image(upscaled_img * 0.5 + 0.5, f"{repo_path}/src/models/SRGAN/test_images/{file}")
    gen.train()

    return upscaled_img * 0.5 + 0.5, image