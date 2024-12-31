#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 18:45:40 2024

@author: J. Seaward
"""

import torch
from byol_pytorch import BYOL
from torchvision import models

from data_img import Img_VAE_Dataset

#%%

resnet = models.resnet50(pretrained=True)
ds = Img_VAE_Dataset("data/hardhat/test")

learner = BYOL(
    resnet,
    image_size = 256,
    hidden_layer = 'avgpool'
)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

