#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 17:48:54 2024

@author: JSeaward
"""
import argparse

from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    train_test_split,  # can just train/test split np.arange(len(ds)) to get back indeces for large sets
    StratifiedKFold,  # use X = np.zeros(n_samples) in .split
)  # if you have indeces and want knns from large set, can make an idx array and NearestVectorCaller._call_vec_set
from data import Img_Obj_Dataset

ds = Img_Obj_Dataset("data/hardhat/test", max_dim=256)
# can pass metric == "cosine", get from NearestVectorCaller.metric on large sets
knn = KNeighborsClassifier(weights="distance")
X, y = ds.whole_img_vec_set[:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


knn.fit(X_train, y_train)
# knn.predict(X_test)
accuracy_score(y_test, knn.predict(X_test))
## try standard sklearn scoring pipline w & w/o k-fold validation

def mean_gen(data):
    n = 0
    mean = 0.0
 
    for x in data:
        n += 1
        mean += (x - mean)/n

    if n < 1:
        return float('nan')
    else:
        return mean