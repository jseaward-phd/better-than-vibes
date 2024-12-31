#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 18:45:40 2024

@author: J. Seaward
"""

import torch
from byol_pytorch import BYOL
from torchvision import models
from torchvision.transforms import v2

from data_img import Img_VAE_Dataset
from tqdm import trange

# %%

resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).cuda(0)
ds = Img_VAE_Dataset("data/hardhat/test", max_dim=256, numpy=False)

learner = BYOL(resnet, image_size=256, hidden_layer="avgpool")

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)
dl = torch.utils.data.DataLoader(ds, batch_size=24, shuffle=True)

# %%
EPOCHS = 25
for _ in trange(EPOCHS):
    for images, lbls in dl:
        images = images.cuda(0)
        loss = learner(images)
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average()  # update moving average of target encoder

# save your improved network
torch.save(resnet.state_dict(), "./improved-net_cuda.pt")

# %%

imgs, lbls = ds[[5, 10, 15]]
imgs = torch.Tensor(imgs)
t = v2.RandomHorizontalFlip(p=1)
imgs2 = t(imgs)
projection, embedding = learner(imgs.cuda(0), return_embedding=True)
projection2, embedding2 = learner(imgs2.cuda(0), return_embedding=True)
print(torch.cdist(embedding, embedding))
print(torch.cdist(embedding, embedding2))
