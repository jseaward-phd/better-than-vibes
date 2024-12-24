#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:00:24 2024

@author: J. Seaward
"""

from pytorch_lightning import LightningDataModule
from typing import List, Optional, Sequence, Union, Any, Callable
from pathlib import Path
import yaml 
import numpy as np

from data_img import Img_VAE_Dataset, VAEImageCollectionSet

from PyTorchVAE.models.vanilla_vae import VanillaVAE
from PyTorchVAE.dataset import VAEDataset
from pytorch_lightning import Trainer
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from skimage.io import ImageCollection



#%% defs
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
        img_sz: Union[int, Sequence[int]] = 640,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()
        
        if val_idx is None and val_data_set is None:
            raise Exception("Must provide a val set, either by index or a sepeate dataset.")
        if not isinstance(data_set, Img_VAE_Dataset):
            data_set = Img_VAE_Dataset(data_set,max_dim=img_sz)
        
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
        
            
    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(148),
                transforms.Resize(self.patch_size),
                transforms.ToTensor(),
            ]
        )
        
        val_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(148),
                transforms.Resize(self.patch_size),
                transforms.ToTensor(),
            ]
        )
        
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


#%% routine

config_path = 'PyTorchVAE/configs/btv_vae.yaml'

config = yaml.safe_load(open(config_path,'r'))
model = VanillaVAE(**config['model_params'])
data = vae_dataset(val_idx=np.load('scratch.npy'),
    **config["data_params"], pin_memory=len(config["trainer_params"]["gpus"]) != 0
)
