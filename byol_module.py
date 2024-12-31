#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 18:45:40 2024

@author: J. Seaward
"""


from pathlib import Path
import os
import argparse

from typing import Union, Optional, OrderedDict

import torch
from byol_pytorch import BYOL
from torchvision import models, datasets
from torchvision.transforms import v2

from data_img import Img_VAE_Dataset
from tqdm import trange


# %% args

parser = argparse.ArgumentParser()

## Dataloader arguments
parser.add_argument("--datapath", "-p", help="Path to stored dataset.")
parser.add_argument(
    "--image_size",
    default=256,
    type=int,
    help="Length in pixels to scale the max dimension of the images to.",
)
parser.add_argument(
    "--batch_size", default=24, type=int, help="Batch size for the dataloader."
)
parser.add_argument(
    "--shuffle",
    "-s",
    default=False,
    type=bool,
    help="If the dataloader will shuffle the samples.",
)

## Learner arguments
parser.add_argument(
    "--state_dict",
    "-w",
    default=None,
    type=Union[Path, str],
    help="Path to pretrained weights state dictionary",
)
parser.add_argument(
    "--device",
    "-d",
    default=0,
    type=Union[int, str],
    help="Cuda device to use. Pass -1 for cpu.",
)
parser.add_argument(
    "--lr", default=3e-4, type=float, help="Learning rate for training."
)

## Train loop arguments
parser.add_argument(
    "--epochs", "-e", default=50, type=int, help="Numper of training epochs."
)
parser.add_argument(
    "--outpath",
    "-o",
    default="./byol_resnet.pt",
    type=Union[Path, str],
    help="Where to save the weights after training.",
)

args = parser.parse_args()

# %% functions


def _collect_dataloader(
    path: Union[str, Path], im_sz: int = 256, batch_sz: int = 24, shuffle: bool = False
) -> torch.utils.data.DataLoader:
    """
    Returns a torch dataloader of the data at PATH. Can be in yolo form with images/labels
    subdirectories or the Caltech256 dataset. Point at an empty or non-existant directory to
    put Caltech256 dataset there.

    Parameters
    ----------
    path : Union[str,Path]
        Path to dataset.
    im_sz : int, optional
        Length in pixels to scale the max dimension of the images to. The default is 256.
    batch_sz : int, optional
        Batch size for the dataloader. The default is 24.
    shuffle : bool, optional
        If the dataloader will shuffle the samples. The default is False.

    Returns
    -------
    dl : pytorch Dataloader
        Dataloader for training.

    """
    if os.path.exists(path) and len(os.listdir(path)) > 0:
        if "tech256" in path:
            ds = datasets.Caltech256(path)
        else:
            ds = Img_VAE_Dataset(path, max_dim=im_sz, numpy=False)
    else:
        Path(path).mkdir(exist_ok=True, parents=True)
        ds = datasets.Caltech256(path, download=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_sz, shuffle=shuffle)

    return dl


def _collect_learner(
    model: Optional[models.resnet.ResNet] = None,
    state_dict: Optional[Union[Path, str, OrderedDict]] = None,
    device: Union[int, str, torch.device] = torch.device("cuda"),
    lr: float = 3e-4,
    im_sz: int = 256,
    weights=models.ResNet50_Weights.DEFAULT,
):
    if model is None:
        model = models.resnet50(weights=weights)
    if not isinstance(device, torch.device):
        device = torch.device(device)
    model.to(device)
    if state_dict is not None:
        if isinstance(state_dict, (str, Path)):
            state_dict = torch.load(state_dict, weights_only=True)
        model.load_state_dict(state_dict)
    learner = BYOL(model, image_size=im_sz, hidden_layer="avgpool")
    opt = torch.optim.Adam(learner.parameters(), lr=lr)

    return learner, opt


# %%
def train(
    learner,
    opt,
    dataloader,
    epochs=50,
    outpath="./improved-net1.pt",
    device=torch.device("cuda"),
):
    for _ in trange(epochs):
        for images, _ in dataloader:
            images = images.to(device)
            loss = learner(images)
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average()  # update moving average of target encoder
    torch.cuda.empty_cache()
    # save your improved network
    if Path(outpath).exists():
        raise FileExistsError
    Path(os.path.split(outpath)[0]).mkdir(exist_ok=True, parents=True)
    torch.save(learner.net.state_dict(), outpath)


def _old_test():
    ds = Img_VAE_Dataset("data/hardhat/test", max_dim=256, numpy=False)
    resnet = models.resnet50().cuda(0)
    resnet.load_state_dict(torch.load("./improved-net_cuda.pt", weights_only=True))
    learner = BYOL(resnet, image_size=256, hidden_layer="avgpool")
    imgs, _ = ds[[5, 10, 15]]
    imgs = torch.Tensor(imgs)
    t = v2.RandomHorizontalFlip(p=1)
    imgs2 = t(imgs)
    projection, embedding = learner(imgs.cuda(0), return_embedding=True)
    projection2, embedding2 = learner(imgs2.cuda(0), return_embedding=True)
    print(torch.cdist(embedding, embedding))
    print(torch.cdist(embedding, embedding2))


def main(args=args):
    # should think about a way to call main with suplimental args and whether the args should be parsed every import

    dataloader = _collect_dataloader(
        path=args.datapath,
        im_sz=args.image_size,
        batch_sz=args.batch_size,
        shuffle=args.shuffle,
    )

    learner, opt = _collect_learner(
        model=None,
        state_dict=args.state_dict,
        device=args.device,
        lr=args.lr,
        im_sz=args.image_size,
        weights=models.ResNet50_Weights.DEFAULT,
    )

    train(
        learner=learner,
        opt=opt,
        dataloader=dataloader,
        epochs=args.epochs,
        outpath=args.outpath,
        device=args.device,
    )


if __name__ == "__main__":
    if args.device < 0:
        args.device = torch.device("cpu")
    elif torch.cuda.device_count() == 1:
        args.device = torch.device("cuda")
    else:
        args.device = torch.device(f"cuda:{args.device}")

    main(args)
