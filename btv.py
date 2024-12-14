#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 17:48:54 2024

@author: JSeaward
"""
import argparse

from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    train_test_split,  # can just train/test split np.arange(len(ds)) to get back indeces for large sets
    StratifiedKFold,  # use X = np.zeros(n_samples) in .split
)  # if you have indeces and want knns from large set, can make an idx array and NearestVectorCaller._call_vec_set
import numpy as np
from pandas import DataFrame
from copy import deepcopy

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

# only include p if y==1 or, really, want p(right answer) so take elemnt of predicted prob indicated by binary y.
def prediction_info_multilabel(y_true, y_predicted):  #for multilabel classification NOT WORKINNG
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
                I = -np.log(p[t]) / np.log(2) # bits
                if np.isinf(I):
                    I = y_true.size
                A[i, j] = I
    return A

def prediction_info(y_true, y_predicted):  #for classification with a single, binary label 
    # replace log(0) with n instead of -Inf
    if isinstance(y_predicted, list):
        y_predicted = np.stack(y_predicted, axis=1)
    A = []
    with np.errstate(divide="ignore"):
        for pred, true in zip(y_predicted, y_true):
            I = -np.log(pred[true]/2+1/2) / np.log(2) # bits
            A.append(I)
    return np.array(A)

def prediction_entropy(y_true, y_predicted):
    A = prediction_info(y_true, y_predicted)
    return np.mean(A, axis=0)


# def prediction_info_generator(y_true, y_predicted): # sinlge label?? def  a generator
#     # replace log(0) with n instead of -Inf
#     if isinstance(y_predicted, list):
#         y_predicted = np.stack(y_predicted, axis=1)
#     with np.errstate(divide="ignore"):
#         for pred, true in zip(y_predicted, y_true):
#             for p, t in zip(pred, true):
#                 I = -np.log(p[t])/np.log(2) # bits
#                 if np.isinf(I):
#                     I = y_true.size
#                 yield I

def cal_info_about(X_train,y_train, X_test, y_test, clf): #make idx version
    clf2= deepcopy(clf)
    baseline_info = np.sum(prediction_info(y_test, clf2.predict_proba(X_test)))
    clf2.fit(X_train, y_train)
    rel_info =  baseline_info - np.sum(prediction_info(y_test, clf2.predict_proba(X_test)))
    return rel_info

def order_folds_by_entropy(X, y, clf, fold_idxs:list, reverse=True, info=False): # reverse = True sorts highest to lowest, so prioratize the training data that the model, as provided, knows the least about.
    Hs = []
    for idxs in fold_idxs:
        y_predicted = clf.predict_proba(X.iloc[idxs]) if isinstance(X, DataFrame) else  clf.predict_proba(X[idxs]) 
        score = np.sum(prediction_entropy(y[idxs], y_predicted)) if not info else np.sum(prediction_info(y[idxs], y_predicted))
        Hs.append(score)
    fold_idxs = [list(x) for x in fold_idxs]
    _, sorted_fold_idxs = zip(*sorted(zip(Hs, fold_idxs), reverse=reverse)) # sorts on first in inner zip
    return [np.array(x) for x in sorted_fold_idxs]

def order_samples_by_info(X, y, clf, reverse=True): # reverse = True sorts highest to lowest, so prioratize the training data that the model, as provided, knows the least about.

    y_predicted = clf.predict_proba(X) 
    info = prediction_info(y, y_predicted)

    if not isinstance(X, DataFrame) :
        _, sorted_X = zip(*sorted(zip(info, X), reverse=reverse)) 
    else:
        ascending = not reverse
        sorted_X = X.copy()
        sorted_X['info'] = info
        sorted_X = sorted_X.sort_values('info', ascending=ascending).drop('info', axis=1)
        
    _, sorted_y = zip(*sorted(zip(info, y), reverse=reverse))
    
    return sorted_X, np.array(sorted_y), info
    
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
        score = clf.score(X.iloc[running_test], y_true)
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

def add_best_fold_first_test(X,y,clf,n_splits=10, X_test=None, verbose = True):  # should do some without stratification to show the difference.
    kfold_idx_gen = StratifiedKFold(n_splits=n_splits).split(X, y)
    train_idx = np.array([], int)
    running_train, entropies, train_idx, scores, samples = [], [], [], [], []
    for k, (_, test_idx) in enumerate(kfold_idx_gen):
        running_train.append(test_idx)
    running_train = [list(x) for x in running_train]
    if X_test is None:
        running_test = running_train.pop()
    elif isinstance(X_test, DataFrame):
        running_test = X_test.index
    else: 
        running_test = X_test
        
    try:
        check_is_fitted(clf)
    except NotFittedError as exc:
        print("Unfiitted estimator provided. Fitting on first fold.")
        idx0 = running_train.pop(0)
        clf.fit(X.iloc[idx0], y[idx0])
    iters = len(running_train)
    
    for k in range(iters):
        best_fold_idxs = order_folds_by_entropy(X, y, clf, running_train)[0]
        train_idx = np.append(train_idx, best_fold_idxs).astype(int)
        running_train.remove(list(best_fold_idxs))
        
        clf.fit(X.iloc[train_idx], y[train_idx])
        y_predicted = clf.predict_proba(X.iloc[running_test])
        y_true = y[running_test]
        H = prediction_entropy(y_true, y_predicted)
        score = clf.score(X.iloc[running_test], y_true)
        entropies.append(np.sum(H))
        scores.append(score)
        samples.append(len(train_idx))
        if verbose:
            print(
                f"Fold : {k+1}, " f"Test set entropy : {np.mean(H)}",
                f"Train samples : {len(train_idx)}",
                f"Score : {score}",
            )
    return samples, entropies, scores

def extraction_rate(clf,X_train,y_train, n = 1, entropy = True, mean = True):
    #how much info is left in the training set after fitting. may not want it to be zero. THIS IS WRONG. WANT INFO BEFORE/AFTER FITTING
    A = []
    for i in range(n):
        clf.fit(X_train,y_train)
        a = prediction_entropy(y_train,clf.predict_proba(X_train)) if entropy else np.sum(prediction_info(y_train,clf.predict_proba(X_train)))
        A.append(a)

    out = np.mean(A) if mean else A
    return out

# Do a training set selection routine with a dknn or provided clf, picking a training set with sufficient info (as calculated at the outset) about the test set.

# training routine, 
def train_best_fold_first_test(X,y,clf,n_splits=100,verbose = False, tol=10, X_test=None):  # should do some without stratification to show the difference. Should try selecting training set with info (as justged by inital clf) equal to ignorance in test set (ditto)
    kfold_idx_gen = StratifiedKFold(n_splits=n_splits).split(X, y)
    train_idx = np.array([], int)
    running_train, entropies, train_idx, scores, samples = [], [], [], [], []
    for k, (_, test_idx) in enumerate(kfold_idx_gen):
        running_train.append(test_idx)
    running_train = [list(x) for x in running_train]
    if X_test is None:
        running_test = running_train.pop()
    elif isinstance(X_test, DataFrame):
        running_test = X_test.index
    else: 
        running_test = X_test
    y_true = y[running_test]
    
    try:
        check_is_fitted(clf)
    except NotFittedError as exc:
        print("Unfiitted estimator provided. Fitting on first fold.")
        idx0 = running_train.pop(0)
        clf.fit(X.iloc[idx0], y[idx0])
    iters = len(running_train)
    best_idxs = []
    last_best_k = 0
    best_H = np.inf
    
    for k in range(iters):
        best_fold_idxs = order_folds_by_entropy(X, y, clf, running_train)[0]
        train_idx = np.append(train_idx, best_fold_idxs).astype(int)
        running_train.remove(list(best_fold_idxs))
        
        clf.fit(X.iloc[train_idx], y[train_idx])
        y_predicted = clf.predict_proba(X.iloc[running_test])
        H = prediction_entropy(y_true, y_predicted)
        score = clf.score(X.iloc[running_test], y_true)
        if H<best_H: 
            best_idxs = train_idx
            last_best_k = k
            best_H = H
        entropies.append(H)
        scores.append(score)
        samples.append(len(train_idx))
        if verbose:
            print(
                f"Fold : {k+1}, " f"Test set entropy : {np.mean(H)}",
                f"Train samples : {len(train_idx)}",
                f"Score : {score}",
            )
        if k - last_best_k >= tol: 
            print(f"No improvement found after adding {k - last_best_k} folds ({len(train_idx) - len(best_idxs)} samples). Fitting on {last_best_k} folds, {len(best_idxs)} smaples.")
            break
    clf.fit(X.iloc[best_idxs], y[best_idxs])
    return samples, entropies, scores, best_idxs