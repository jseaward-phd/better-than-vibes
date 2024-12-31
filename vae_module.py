#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:00:24 2024

@author: J. Seaward
"""

from typing import List, Optional, Sequence, Union
from pathlib import Path
import os, yaml
import numpy as np

from data_img import Img_VAE_Dataset, VAEImageCollectionSet, LetterBox

from PyTorchVAE.models.vanilla_vae import VanillaVAE
from PyTorchVAE.experiment import VAEXperiment
from torchvision import transforms
from torch.utils.data import DataLoader
from skimage.io import ImageCollection
from scipy.spatial.distance import euclidean

from torch.cuda import is_available, empty_cache
from pytorch_lightning import LightningDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch import load, float32


# from pytorch_lightning.plugins import DDPPlugin


# %% defs
class vae_dataset(LightningDataModule):
    # can feed a dataset to this and it will do the train/test split
    def __init__(
        self,
        data_set: Union[Img_VAE_Dataset, Path, str],
        val_idx: Optional[Sequence[int]] = None,
        val_data_set: Optional[Img_VAE_Dataset] = None,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        # img_sz: Union[int, Sequence[int]] = 640,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        if val_idx is None and val_data_set is None:
            raise Exception(
                "Must provide a val set, either by index or a sepeate dataset."
            )
        if not isinstance(data_set, Img_VAE_Dataset):
            data_set = Img_VAE_Dataset(data_set, max_dim=patch_size)
            
        if val_data_set is not None:
            self.train_data_set = data_set.imc
            self.val_data_set = val_data_set.imc
        else:
            train_idx = np.setdiff1d(np.arange(len(data_set)), val_idx)
            train_flist = [data_set.imc.files[i] for i in train_idx]
            val_flist = [data_set.imc.files[i] for i in val_idx]

            self.train_data_set = ImageCollection(train_flist)
            self.val_data_set = ImageCollection(val_flist)

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        # self.img_sz = img_sz

        self.transformsv2 = transforms.v2.Compose(
           [
               LetterBox(new_shape=self.patch_size, scaleup=True),
               transforms.v2.ToImage(),
               transforms.v2.RandomHorizontalFlip(),
               transforms.v2.ToDtype(float32, scale=True),
           ])
        

    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = self.transformsv2
        # transforms.Compose(
        #     [
        #         transforms.ToPILImage(),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomCrop(148),
        #         transforms.Resize(self.patch_size),
        #         transforms.ToTensor(),
        #     ]
        # )

        val_transforms = self.transformsv2
        # transforms.Compose(
        #     [
        #         transforms.ToPILImage(),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomCrop(148),
        #         transforms.Resize(self.patch_size),
        #         transforms.ToTensor(),
        #     ]
        # )

        self.train_dataset = VAEImageCollectionSet(  # need to make a child of from torchvision.datasets.vision import VisionDataset
            self.train_data_set,
            split="train",
            transform=train_transforms,
        )

        # Replace CelebA with your dataset
        self.val_dataset = VAEImageCollectionSet(
            self.val_data_set,
            split="test",
            transform=val_transforms,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )


def format_lightning_state_dict(indict: dict) -> dict:
    outdict = {".".join(k.split(".")[1:]): v for k, v in indict["state_dict"].items()}
    return outdict


def convert_embedding_list(latents_in: list) -> np.array:
    out = np.vstack(
        [
            np.hstack([x.detach().numpy(), y.exp().detach().numpy()])
            for x, y in latents_in
        ]
    ) if latents_in[0].shape[0] > 2 else  np.vstack(
        [
            np.hstack([latents_in[0].detach().numpy(), latents_in[1].exp().detach().numpy()])
        ]
    )
    
    return out


def dist_weight_passthru(dist): #
    return dist


def mean_and_var_dist(a, b):  # takes 2 1D vectors, need to think about 1- and 1/
    a = a.squeeze()
    b = b.squeeze()
    break_pt = len(a) // 2
    assert (
        break_pt == len(b) / 2
    ), "vectors must be of the same length and be such that the first half is the mean and the second half is the lag-variance"  #
    # Excpt the variance of the test pt doesn't matter...
    # raw_dist = euclidean(np.vstack([a[:break_pt],b[:break_pt]]))
    return np.exp((-0.5 * (a[:break_pt] - (b[:break_pt])) ** 2 / a[break_pt:]).sum())


# %% routine phase 1

config_path = "PyTorchVAE/configs/btv_vae.yaml"

config = yaml.safe_load(open(config_path, "r"))
model = VanillaVAE(**config["model_params"])
data = vae_dataset(
    val_idx=np.load("scratch.npy"),  # split test off differently
    **config["data_params"],
    pin_memory=is_available(),
)
data.setup()
# %% Load model


weights = load("runs/VanillaVAE/version_5/checkpoints/last.ckpt", weights_only=True)
model.load_state_dict(state_dict=format_lightning_state_dict(weights))

# %%
tb_logger = TensorBoardLogger(
    save_dir=config["logging_params"]["save_dir"],
    name=config["model_params"]["name"],
)
experiment = VAEXperiment(model, config["exp_params"])

runner = Trainer(
    logger=tb_logger,
    callbacks=[
        LearningRateMonitor(),
        ModelCheckpoint(
            save_top_k=2,
            dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
            monitor="val_loss",
            save_last=True,
        ),
    ],
    # strategy="ddp_notebook", #DDPPlugin(find_unused_parameters=False),
    **config["trainer_params"],
)
# %%

Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")
empty_cache()
runner.fit(experiment, datamodule=data)
print('Done!')