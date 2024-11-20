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
import numpy as np

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

# only include p if y==1 or, reeally, want p(right answer) so take elemnt of predicted prob indicated by binary y.
def prediction_info(y_true, y_predicted):
    # replace log(0) with n instead of -Inf
    if isinstance(y_predicted, list):
        y_predicted = np.stack(y_predicted, axis=1)
    class_num = 1 if len(y_true.shape) == 1 else y_true.shape[1]
    y_true = y_true.reshape([-1,class_num])
    y_predicted = y_predicted.reshape([-1,class_num,2])
    A = np.zeros_like(y_true)
    with np.errstate(divide="ignore"):
        for i, (pred, true) in enumerate(zip(y_predicted, y_true)):
            for j, (p, t) in enumerate(zip(pred, true)):
                I = -np.log(p[t]) / np.log(class_num*2)
                if np.isinf(I):
                    I = y_true.size
                A[i, j] = I
    return A

def prediction_entropy(y_true, y_predicted):
    A = prediction_info(y_true, y_predicted)
    return np.mean(A, axis=0)


def prediction_info_generator(y_true, y_predicted): # sinlge label?? def  a generator
    # replace log(0) with n instead of -Inf
    if isinstance(y_predicted, list):
        y_predicted = np.stack(y_predicted, axis=1)
    with np.errstate(divide="ignore"):
        for pred, true in zip(y_predicted, y_true):
            for p, t in zip(pred, true):
                I = -np.log(p[t])
                if np.isinf(I):
                    I = y_true.size
                yield I
                
def order_folds_by_entropy(X, y, clf, fold_idxs:list, reverse=True): # reverse = True sorts highest to lowest 
    Hs = []
    for idxs in fold_idxs:
        y_predicted = clf.predict_proba(X.iloc[idxs])
        Hs.append(np.sum(prediction_entropy(y[idxs], y_predicted)))
    fold_idxs = [list(x) for x in fold_idxs]
    _, sorted_fold_idxs = zip(*sorted(zip(Hs, fold_idxs), reverse=reverse)) # sorts on first in inner zip
    return [np.array(x) for x in sorted_fold_idxs]
        
def add_stratified_folds_test(X,y,clf,n_splits=10,verbose = True):  # should do some without stratification to show the difference.
    kfold_idx_gen = StratifiedKFold(n_splits=n_splits).split(X, y)
    train_idx = np.array([], int)
    running_train, entropies, train_idx, scores, samples = [], [], [], [], []
    for k, (_, test_idx) in enumerate(kfold_idx_gen):
        running_train.append(test_idx)
    running_test = running_train.pop()
    for k, old_test_idx in enumerate(running_train):
        train_idx = np.append(train_idx, old_test_idx).astype(int)
        clf.fit(X.iloc[train_idx], y[train_idx])
        y_predicted = clf.predict_proba(X.iloc[running_test])
        y_true = y[running_test]
        H = prediction_entropy(y_true, y_predicted)
        score = clf.score(X.iloc[test_idx], y_true)
        entropies.append(H)
        scores.append(score)
        samples.append(len(train_idx))
        if verbose:
            print(
                f"Fold : {k+1}, " f"Test set entropy : {np.mean(H)}",
                f"Train samples : {len(train_idx)}",
                f"Score : {score}",
            )
    return samples, entropies, scores

def add_best_fold_first_test(X,y,clf,n_splits=10,verbose = True):  # should do some without stratification to show the difference.
    kfold_idx_gen = StratifiedKFold(n_splits=n_splits).split(X, y)
    train_idx = np.array([], int)
    running_train, entropies, train_idx, scores, samples = [], [], [], [], []
    for k, (_, test_idx) in enumerate(kfold_idx_gen):
        running_train.append(test_idx)
    running_train = [list(x) for x in running_train]
    running_test = running_train.pop()
    iters = len(running_train)
      
    for k in range(iters):
        best_fold_idxs = order_folds_by_entropy(X, y, clf, running_train)[0]
        train_idx = np.append(train_idx, best_fold_idxs).astype(int)
        running_train.remove(list(best_fold_idxs))
        
        clf.fit(X.iloc[train_idx], y[train_idx])
        y_predicted = clf.predict_proba(X.iloc[running_test])
        y_true = y[running_test]
        H = prediction_entropy(y_true, y_predicted)
        score = clf.score(X.iloc[test_idx], y_true)
        entropies.append(H)
        scores.append(score)
        samples.append(len(train_idx))
        if verbose:
            print(
                f"Fold : {k+1}, " f"Test set entropy : {np.mean(H)}",
                f"Train samples : {len(train_idx)}",
                f"Score : {score}",
            )
    return samples, entropies, scores

# training routine
