import os
import torch
import config
import wandb
import numpy as np
from torch import nn
from torch import optim
from utils import gradient_penalty, load_checkpoint, save_checkpoint, plot_examples, PSNR
from loss import VGGLoss
from torch.utils.data import DataLoader
from esrgan import Generator, Discriminator, initialize_weights
from tqdm import tqdm
from dataset import MyImageFolder
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import structural_similarity as ssim

torch.backends.cudnn.benchmark = True
repo_path = os.environ.get('THESIS_PATH')

def train_fn(
    loader,
    disc,
    gen,
    opt_gen,
    opt_disc,
    l1,
    vgg_loss,
    g_scaler,
    d_scaler,
    writer,
    tb_step,
):
    loop = tqdm(loader, leave=True)

    disc_losses_real = []
    disc_losses_fake = []

    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            fake = gen(low_res)
            critic_real = disc(high_res)
            critic_fake = disc(fake.detach())
            gp = gradient_penalty(disc, high_res, fake, device=config.DEVICE)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + config.LAMBDA_GP * gp
            )

        opt_disc.zero_grad()
        d_scaler.scale(loss_critic).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        with torch.cuda.amp.autocast():
            l1_loss = 1e-2 * l1(fake, high_res)
            adversarial_loss = 5e-3 * -torch.mean(disc(fake))
            loss_for_vgg = vgg_loss(fake, high_res)
            gen_loss = l1_loss + loss_for_vgg + adversarial_loss

        opt_gen.zero_grad()
        g_scaler.scale(gen_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        writer.add_scalar("Critic loss", loss_critic.item(), global_step=tb_step)
        tb_step += 1

        ## Log discriminator losses for histogram
        disc_losses_real = np.append(disc_losses_real, critic_real.detach().cpu().numpy().flatten())
        disc_losses_fake = np.append(disc_losses_fake, critic_fake.detach().cpu().numpy().flatten())

        if idx % 100 == 0 and idx > 0:
            saved_im, original = plot_examples(repo_path + "/data/processed/test_crops/", gen, False)

            ## log wandb loss
            wandb.log({"disc_loss":loss_critic, "adv_loss":adversarial_loss, "vgg_loss":loss_for_vgg, "mse_loss":l1_loss, "gen_loss":gen_loss, "gradient penalty":gp})

            if idx % 1000 == 0:
                orig, svd = np.asarray(original), np.squeeze(np.transpose(saved_im.detach().cpu().numpy(), (2,3,1,0)))
                psnr =  PSNR(orig/255, svd)
                struc_sim = ssim(orig/255, svd, multichannel=True, channel_axis=2, data_range=1)
                #print(psnr)
                #print(struc_sim)
                wandb.log({"example":wandb.Image(saved_im, caption=f'PSNR: {psnr}, SSIM: {struc_sim}')})

        loop.set_postfix(
            gp=gp.item(),
            critic=loss_critic.item(),
            l1=l1_loss.item(),
            vgg=loss_for_vgg.item(),
            adversarial=adversarial_loss.item(),
        )
    
    ## log histogram
    real_hist = wandb.Histogram(disc_losses_real)
    fake_hist = wandb.Histogram(disc_losses_fake)

    wandb.log({'real_hist':real_hist, 'fake_hist':fake_hist})

    return tb_step


def main():
    wandb.init(
        project="thesis_esrgan", entity='s164397',

        config={
            "learning_rate": config.LEARNING_RATE,
            "architecture": "ESRGAN",
            "epochs": config.NUM_EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "imsize_hr": config.HIGH_RES
        }
    )
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
    initialize_weights(gen)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
    writer = SummaryWriter("logs")
    tb_step = 0
    l1 = nn.L1Loss()
    gen.train()
    disc.train()
    vgg_loss = VGGLoss()

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    if config.PRETRAINED:
        gen_path = '/work3/s164397/Thesis/Oblique2019/saved_models/ESRGAN/pretrained/ESRGAN_generator.pth'
        load_checkpoint(
            gen_path,
            gen,
            opt_gen,
            config.LEARNING_RATE
        )

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN,
            gen,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC,
            disc,
            opt_disc,
            config.LEARNING_RATE,
        )

    for epoch in range(config.NUM_EPOCHS):
        tb_step = train_fn(
            loader,
            disc,
            gen,
            opt_gen,
            opt_disc,
            l1,
            vgg_loss,
            g_scaler,
            d_scaler,
            writer,
            tb_step,
        )

        if config.SAVE_MODEL:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)


if __name__ == "__main__":
    try_model = True
    gen_path = '/work3/s164397/Thesis/Oblique2019/saved_models/ESRGAN/3/gen.pth'
    #gen_path = repo_path + '/../gen.pth'

    if try_model:
        # Will just use pretrained weights and run on images
        # in test_images/ and save the ones to SR in saved/
        gen = Generator(in_channels=3).to(config.DEVICE)
        opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
        load_checkpoint(
            gen_path,
            gen,
            opt_gen,
            config.LEARNING_RATE,
        )
        plot_examples(repo_path + "/data/processed/test_crops3/", gen, False)
    else:
        # This will train from scratch
        main()
