import torch
from torch.utils.data import DataLoader

import yaml
from addict import Dict
import os
import argparse

from model import NetD, NetG, NetE

from libs.dataloader import Dataset
from libs.transform import ImageTransform
from libs.weights import weights_init
from trainer import train

import wandb


def get_parameters():
    """
    make parser to get parameters
    """

    parser = argparse.ArgumentParser(description="take parameters like num_epochs...")

    parser.add_argument("config", type=str, help="path of a config file for training")

    parser.add_argument("--no_wandb", action="store_true", help="Add --no_wandb option")

    return parser.parse_args()


args = get_parameters()

CONFIG = Dict(yaml.safe_load(open(args.config)))

if not args.no_wandb:
    wandb.init(
        config=CONFIG,
        name=CONFIG.name,
        # have to change project when you want to change project
        project="mnist78",
        jobtype="training",
    )

mean = (0.5,)
std = (0.5,)

train_dataset = Dataset(
    csv_file=CONFIG.train_csv_file, transform=ImageTransform(mean, std)
)

train_dataloader = DataLoader(train_dataset, batch_size=CONFIG.batch_size, shuffle=True)

G = NetG(CONFIG)
D = NetD(CONFIG)
E = NetE(CONFIG)

G.apply(weights_init)
D.apply(weights_init)
E.apply(weights_init)

if not args.no_wandb:
    # Magic
    wandb.watch(G, log="all")
    wandb.watch(D, log="all")
    wandb.watch(E, log="all")

G_update, D_update, E_update = train(
    G,
    D,
    E,
    z_dim=CONFIG.z_dim,
    dataloader=train_dataloader,
    CONFIG=CONFIG,
    no_wandb=args.no_wandb,
)

if not os.path.exists(CONFIG.save_dir):
    os.makedirs(CONFIG.save_dir)

torch.save = (G_update.state_dict(), os.path.join(CONFIG.save_dir, "G.prm"))
torch.save = (D_update.state_dict(), os.path.join(CONFIG.save_dir, "D.prm"))
torch.save = (E_update.state_dict(), os.path.join(CONFIG.save_dir, "E.prm"))

print("Done")

