import os
import torch
import config
import wandb
import numpy as np
from PIL import Image
from torch import nn
from torch import optim
from utils import load_checkpoint, save_checkpoint, plot_examples, PSNR
from loss import VGGLoss
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.transforms as T
from srgan import Generator, Discriminator
from tqdm import tqdm
from dataset import MyImageFolder
from skimage.metrics import structural_similarity as ssim

torch.backends.cudnn.benchmark = True
repo_path = os.environ.get('THESIS_PATH')

def train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss):
    loop = tqdm(loader, leave=True)

    disc_losses_real = []
    disc_losses_fake = []

    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)
        
        if idx == 0:
            save_image(high_res, repo_path + '/src/models/saved/high_res.png')

            save_image(low_res, repo_path + '/src/models/saved/low_res.png')
            
        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        fake = gen(low_res)
        disc_real = disc(high_res)
        disc_fake = disc(fake.detach())
        disc_loss_real = bce(
            disc_real, torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real)
        )
        disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = disc_loss_fake + disc_loss_real

        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        disc_fake = disc(fake)
        l2_loss = mse(fake, high_res)
        adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
        # 2e-6 * vgg VS 0.006
        loss_for_vgg = 2e-6 * vgg_loss(fake, high_res)
        gen_loss = loss_for_vgg + adversarial_loss + l2_loss
        # + l2_loss

        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

        ## Log discriminator losses for histogram
        disc_losses_real = np.append(disc_losses_real, disc_real.detach().cpu().numpy().flatten())
        disc_losses_fake = np.append(disc_losses_fake, disc_fake.detach().cpu().numpy().flatten())

        if idx % 100 == 0:
            saved_im, original = plot_examples(repo_path + "/data/processed/test_crops/", gen, True)
            
            ## log wandb loss
            #wandb.log({"disc_loss":loss_disc, "adv_loss":adversarial_loss, "vgg_loss":loss_for_vgg, "mse_loss":l2_loss, "gen_loss":gen_loss})
        
            if idx % 1000 == 0:
                orig, svd = np.asarray(original), np.squeeze(np.transpose(saved_im.detach().cpu().numpy(), (2,3,1,0)))
                psnr =  PSNR(orig/255, svd)
                struc_sim = ssim(orig/255, svd, multichannel=True, channel_axis=2, data_range=1)
                
                #wandb.log({"example":wandb.Image(saved_im, caption=f'PSNR: {psnr}, SSIM: {struc_sim}')})

    ## plot examples at the end of epoch
    saved_im, original = plot_examples(repo_path + "/data/processed/test_crops2/", gen, False)

    ## log histogram
    #real_hist = wandb.Histogram(disc_losses_real)
    #fake_hist = wandb.Histogram(disc_losses_fake)

    #wandb.log({'real_hist':real_hist, 'fake_hist':fake_hist})
        

def evaluate(disc, gen):
    return 0

def main():
    '''wandb.init(
        project="thesis_srgan", entity='s164397',

        config={
            "learning_rate": config.LEARNING_RATE,
            "architecture": "SRGAN",
            "epochs": config.NUM_EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "imsize_hr": config.HIGH_RES
        }
    )'''
    dataset = MyImageFolder(root_dir=repo_path + "/data/processed/crops3")
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=config.NUM_WORKERS,
    )
    gen = Generator(in_channels=3).to(config.DEVICE)
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    
    ## watch models for wandb
    #wandb.watch(gen, log_freq=100)
    #wandb.watch(disc, log_freq=100)
    
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    vgg_loss = VGGLoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN,
            gen,
            opt_gen,
            config.LEARNING_RATE,
            True,
        )
        load_checkpoint(
           config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE, True,
        )

    for epoch in range(config.NUM_EPOCHS):
        train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss)

        if config.SAVE_MODEL:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)
    
    # log interpolated and original image in wandb to compare
    test_path = repo_path + '/data/processed/test_crops/'
    files = os.listdir(test_path)
    image = Image.open(repo_path + "/data/processed/test_crops/" + files[-1])

    lr_im = (config.lowres_transform(image=np.asarray(image))["image"]
                    .unsqueeze(0)
                    .to(config.DEVICE))

    size = (config.HIGH_RES, config.HIGH_RES)
    #interpolated_image = lr_im.resize(size, resample=Image.Resampling.BICUBIC)
    interpolated_image = T.functional.resize(lr_im, size=size, interpolation=T.InterpolationMode.BICUBIC)
    transform = T.ToPILImage()
    trans_int_img = transform(interpolated_image[0])
    
    #wandb.log({"bicubic":wandb.Image(trans_int_img, caption='interpolated image'), "original":wandb.Image(image, caption='original image')})
    save_image(interpolated_image, repo_path + '/src/models/saved/interpolated_img.png')
    image.save(repo_path + '/src/models/saved/original_img.png', 'PNG')

if __name__ == "__main__":
    try_model = True
    gen_path = '/work3/s164397/Thesis/Oblique2019/saved_models/SRGAN/3/gen.pth.tar'
    #gen_path = repo_path + '/../gen.pth'

    if try_model:
        # Will just use pretrained weights and run on images
        # in test_images/ and save the ones to SR in saved/
        gen = Generator(in_channels=3).to(config.DEVICE)
        opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))

        #checkpoint = torch.load(gen_path, map_location=config.DEVICE)
        #print(checkpoint.keys())
        
        load_checkpoint(
            gen_path,
            gen,
            opt_gen,
            config.LEARNING_RATE,
            True,
        )
        saved_im, original = plot_examples(repo_path + "/data/processed/test_crops2/", gen, False)
    else:
        main()


