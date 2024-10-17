#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 17:48:54 2024

@author: JSeaward
"""
import argparse

from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import accuracy_scorepdb
from sklearn.model_selection import (
    train_test_split, # can just train/test split np.arange(len(ds)) to get back indeces for large sets
    StratifiedKFold,  # use X = np.zeros(n_samples) in .split
) #if you have indeces and want knns from large set, can make an idx array and NearestVectorCaller._call_vec_set
from data import Img_Obj_Dataset

ds = Img_Obj_Dataset("data/hardhat/test", max_dim=256)
knn = KNeighborsClassifier(weights="distance")
X, y = ds.whole_img_vec_set[:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# can pass metric == "cosine", get from NearestVectorCaller.metric on large sets
knn.fit(X_train, y_train)
# knn.predict(X_test)
accuracy_score(y_test, knn.predict(X_test))
## try standard sklearn scoring pipline w & w/o k-fold validation
