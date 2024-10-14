#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 17:48:54 2024

@author: rakshat
"""
import argparse

from sklearn.neighbors import (
    KNeighborsClassifier,
)  # pass weights = "distance" # coerce into scklearn-style set,

from data import Img_Vec_Dataset
