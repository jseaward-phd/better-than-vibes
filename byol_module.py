#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 18:45:40 2024

@author: J. Seaward

Script for constructing and training Bootstrap Your Own Latent (BYOL) learners.
More explaining....
"""


from pathlib import Path
import os
import argparse

from typing import Union, Optional, OrderedDict, Tuple

import torch
import numpy as np
from byol_pytorch import BYOL
from torchvision import models, datasets
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm

from data_img import Img_VAE_Dataset, torch_load_fn


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
            transforms = torch_load_fn(dims=im_sz).transforms
            ds = datasets.Caltech256(path, transform=transforms)
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
) -> Tuple[BYOL, torch.optim.Adam]:
    """
    Construct a BYOL learner from a model. If none is provided, a torchvision resnet50
    with the most up-to-date weights provided by your torchvision install is used.
    Also returns an Adam optimizer with learning rate lr.

    Parameters
    ----------
    model : Optional[models.resnet.ResNet], optional
        Model to be BOYL trained by the learner. The default is None, returning a ResNet50.
    state_dict : Optional[Union[Path, str, OrderedDict]], optional
        Weights or path to weights to be loaded into the model before training.
        The default is None, using the provided or pretrained weights.
    device : Union[int, str, torch.device], optional
        Device on which to do the training.
        Pass 'cpu' for cpu or an int or a string like for a 'cuda:1' for a specific gpu.
        The default is torch.device("cuda"), allowing torch to auto-select a gpu.
    lr : float, optional
        Learning rate to pass to the optimizer. The default is 3e-4.
    im_sz : int, optional
        Maximum dimension to which to scale the images in the dataset. The default is 256.

    Returns
    -------
    learner : BYOL learner
        Trained learner that provides the embedding. Uses hidden_layer="avgpool".
        The model is stored as 'learner.net'.
    opt : pytorch Adam optimizer
        Adam optimizer with the provided learning rate.

    """
    if model is None:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
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
    learner: BYOL,
    opt,
    dataloader: torch.utils.data.DataLoader,
    epochs: int = 50,
    outpath: Optional[Union[Path, str]] = None,
    device: Union[str, int, torch.device] = torch.device("cuda"),
) -> None:
    """
    Train the provided BYOL learner, using the provided optimizer and dataloader.
    Saves the weights of learner.net at the outpath if one is provided.
    Learner is trained "in place" and nothing is returned.

    Parameters
    ----------
    learner : BYOL
        Learner to be trained (in place).
    opt : pytorch optimizer
        Optimizer to be used in training.
    dataloader : torch.utils.data.DataLoader
        Dataloader to provide the data for training.
    epochs : int, optional
        Number of epochs to train for. The default is 50.
    outpath : Union[Path, str], optional
        Path at which to save the weights of learner.net.
        Will make partent folers if they do not exist. Will not overwrite an existing file.
        The default is None, and no weights will be saved.
    device : Union[str, int, torch.device], optional
        Device on which to do the training.
        Pass 'cpu' for cpu or an int or a string like for a 'cuda:1' for a specific gpu.
        The default is torch.device("cuda"), allowing torch to auto-select a gpu.

    Raises
    ------
    FileExistsError
        Will not overwrite an exist ing file. Will have a clobber flag in the future.

    Returns
    -------
    None
        Trainer trained in place.

    """
    if outpath is not None:
        if Path(outpath).exists():
            raise FileExistsError("A file already exists at that location.")
    writer = SummaryWriter(
        log_dir=Path(outpath).parent.joinpath("tb_logs", Path(outpath).stem)
    )
    bestloss = 1e5
    try:
        for epoch in trange(epochs, unit="Epoch"):
            losses = []
            for images, _ in tqdm(dataloader, unit="Batch", leave=False):
                images = images.to(device)
                loss = learner(images)
                opt.zero_grad()
                loss.backward()
                opt.step()
                learner.update_moving_average()  # update moving average of target encoder
                losses.append(loss.detach().item())
            meanloss = np.mean(losses)
            writer.add_scalar("Loss/train", meanloss, epoch)
            if meanloss < bestloss:
                bestloss = meanloss
                if outpath is not None:
                    Path(os.path.split(outpath)[0]).mkdir(exist_ok=True, parents=True)
                    torch.save(learner.net.state_dict(), outpath)

    except KeyboardInterrupt:
        pass
    torch.cuda.empty_cache()
    writer.flush()
    writer.close()
    if outpath is not None:
        Path(os.path.split(outpath)[0]).mkdir(exist_ok=True, parents=True)
        torch.save(learner.net.state_dict(), outpath)
        print(f"Model weights saved at {outpath}, with loss {bestloss:.4f}")


def _old_test() -> Tuple[Img_VAE_Dataset, BYOL]:
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

    return ds, learner


def main(args: Union[OrderedDict, argparse.Namespace]) -> BYOL:
    """
    Takes a dictionary or a namespace. Constructs, trains and returns a BYOL learner.

    Parameters
    ----------
    args : Union[OrderedDict, argparse.Namespace]
        Arguments to be used. See argparse block in byol_module.py for more detail.

    Returns
    -------
    BYOL
        Trained BYOL learner.

    """
    if isinstance(args, OrderedDict):
        args = argparse.Namespace(**args)
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
    )

    train(
        learner=learner,
        opt=opt,
        dataloader=dataloader,
        epochs=args.epochs,
        outpath=args.outpath,
        device=args.device,
    )

    return learner


if __name__ == "__main__":
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
        action="store_true",
        help="If the dataloader will shuffle the samples.",
    )

    ## Learner arguments
    parser.add_argument(
        "--state_dict",
        "-w",
        default=None,
        help="Path to pretrained weights state dictionary",
    )
    parser.add_argument(
        "--device",
        "-d",
        default=0,
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
        help="Where to save the weights after training.",
    )
    args = parser.parse_args()

    if args.device < 0:
        args.device = torch.device("cpu")
    elif torch.cuda.device_count() == 1:
        args.device = torch.device("cuda")
    else:
        args.device = torch.device(f"cuda:{args.device}")

    main(args)
