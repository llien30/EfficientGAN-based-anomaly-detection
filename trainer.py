import torch
import torch.nn as nn
from torchvision.utils import save_image

from libs.meter import AverageMeter

import time
from PIL import Image

import wandb


def train(G, D, E, z_dim, dataloader, CONFIG, no_wandb):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    lr_ge = 0.0001
    lr_d = 0.0001 / 4
    beta1, beta2 = 0.5, 0.999

    g_optimizer = torch.optim.Adam(G.parameters(), lr_ge, [beta1, beta2])
    d_optimizer = torch.optim.Adam(D.parameters(), lr_d, [beta1, beta2])
    e_optimizer = torch.optim.Adam(E.parameters(), lr_ge, [beta1, beta2])

    # Binary Cross Entropy
    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    # the default mini batch size
    mini_batch_size = 64
    fixed_z = torch.randn(CONFIG.num_fakeimg, z_dim, 1, 1).to(device)

    G.to(device)
    D.to(device)
    E.to(device)

    G.train()
    D.train()
    E.train()

    torch.backends.cudnn.benchmark = True

    # num_train_imgs = len(dataloader.dataset)
    # batch_size = dataloader.batch_size

    iteration = 1

    num_epochs = CONFIG.num_epochs

    for epoch in range(num_epochs):
        t_epoch_start = time.time()

        print("----------------------(train)----------------------")
        print("Epoch {}/{}".format(epoch, num_epochs))
        print("---------------------------------------------------")

        d_loss_meter = AverageMeter("D_loss", ":.4e")
        g_loss_meter = AverageMeter("G_loss", ":.4e")
        e_loss_meter = AverageMeter("E_loss", ":.4e")

        for samples in dataloader:
            imges = samples["img"]

            imges = imges.to(device)
            mini_batch_size = imges.size()[0]

            label_real = torch.full((mini_batch_size,), 1).to(device)
            label_fake = torch.full((mini_batch_size,), 0).to(device)

            """
            learning Discriminator
            """
            z_out_real = E(imges, CONFIG)
            d_out_real, _ = D(imges, z_out_real, CONFIG)

            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            fake_imges = G(input_z, CONFIG)
            d_out_fake, _ = D(fake_imges, input_z, CONFIG)

            d_loss_real = criterion(d_out_real.view(-1), label_real)
            d_loss_fake = criterion(d_out_fake.view(-1), label_fake)

            d_loss = d_loss_real + d_loss_fake
            d_loss_meter.update(d_loss.item())

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            """
            learning Generator
            """
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            fake_imges = G(input_z, CONFIG)
            d_out_fake, _ = D(fake_imges, input_z, CONFIG)

            g_loss = criterion(d_out_fake.view(-1), label_real)
            g_loss_meter.update(g_loss.item())

            g_optimizer.zero_grad()
            g_loss.backward()
            g_loss.step()

            """
            learning Encoder
            """
            z_out_real = E(imges, CONFIG)
            d_out_real, _ = D(imges, z_out_real, CONFIG)
            # use label_fake to caliculate log(1-D)
            e_loss = criterion(d_out_real.view(-1), label_fake)
            e_loss_meter.update(e_loss.item())

            e_optimizer.zero_grad()
            e_loss.backward()
            e_optimizer.step()

            iteration += 1

        t_epoch_finish = time.time()
        print("---------------------------------------------------")
        print(
            "Epoch{}|| D_Loss :{:.4f} || G_Loss :{:.4f} || D_Loss :{:.4f}".format(
                epoch, d_loss_meter.avg, g_loss_meter.avg, e_loss_meter.avg,
            )
        )
        print("timer:  {:.4f} sec.".format(t_epoch_finish - t_epoch_start))
        fake_imges = G(fixed_z)
        save_image(fake_imges, "fake_imges.png")

        if not no_wandb:
            wandb.log(
                {
                    "train_time": t_epoch_finish - t_epoch_start,
                    "d_loss": d_loss_meter.avg,
                    "g_loss": g_loss_meter.avg,
                    "e_loss": e_loss_meter.avg,
                },
                step=epoch,
            )

            img = Image.open("fake_imges.png")
            wandb.log({"image": [wandb.Image(img)]}, step=epoch)

            t_epoch_start = time.time()
    return G, D, E
