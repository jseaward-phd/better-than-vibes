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
import pandas as pd
from copy import deepcopy
from tqdm import tqdm, trange


def mean_gen(data):
    n = 0
    mean = 0.0

    for x in data:
        n += 1
        mean += (x - mean) / n

    if n < 1:
        return float("nan")
    else:
        return mean


def dist_weight_ignore_self(
    dist,
):  # custom weighting function for a dknn that ignores points already in the fit set.
    if dist.dtype is np.dtype(object):
        for point_dist_i, point_dist in enumerate(dist):
            # check if point_dist is iterable
            # (ex: RadiusNeighborClassifier.predict may set an element of
            # dist to 1e-6 to represent an 'outlier')
            if hasattr(point_dist, "__contains__") and 0.0 in point_dist:
                dist[point_dist_i] = point_dist == 0.0
            else:
                dist[point_dist_i] = 1.0 / point_dist
    else:
        with np.errstate(divide="ignore"):
            dist = 1.0 / dist
        inf_mask = np.isinf(dist)
        dist[inf_mask] = 0
    return dist


# only include p if y==1 or, really, want p(right answer) so take elemnt of predicted prob indicated by binary y.
def prediction_info_multilabel(
    y_true, y_predicted
):  # for multilabel classification NOT WORKINNG
    if isinstance(y_predicted, list):
        y_predicted = np.stack(y_predicted, axis=1)
    class_num = 1 if len(y_true.shape) == 1 else y_true.shape[1]
    y_true = y_true.reshape([-1, class_num])
    y_predicted = y_predicted.reshape([-1, class_num, 2])
    A = np.zeros_like(y_true)
    with np.errstate(divide="ignore"):
        for i, (pred, true) in enumerate(zip(y_predicted, y_true)):
            for j, (p, t) in enumerate(zip(pred, true)):
                I = -np.log(p[t] / 2 + 1 / 2) / np.log(2)  # bits
                if np.isinf(I):
                    I = y_true.size
                A[i, j] = I
    return A


def prediction_info(y_true, y_predicted, discount_chance=True):
    # for classification with a single, binary label, otherwize need to selecet with the true label differently and replace 2 with 2^{# of labels} or do a sum over labels
    A = []
    if isinstance(y_predicted, list):
        y_predicted = np.stack(y_predicted, axis=1)
        classes = y_predicted.shape[2] - 1
        for pred, true in zip(y_predicted, y_true):
            if discount_chance:
                I = [
                    -np.log(p[t] + 1 / 2**classes) / np.log(2)
                    for p, t in zip(pred, true)
                ]
            else:
                I = [
                    -np.log(p[t] / (2**classes) + 1 / (2**classes)) / np.log(2)
                    for p, t in zip(pred, true)
                ]
            A.append(I)
    else:
        classes = y_predicted.shape[1] - 1
        for pred, true in zip(y_predicted, y_true):
            I = (
                -np.log(pred[true] / (2**classes) + 1 / (2**classes)) / np.log(2)
                if not discount_chance
                else -np.log(pred[true] + 1 / 2**classes) / np.log(2)
            )  # bits
            A.append(I)
    A = np.array(A)
    A = np.where(A > 0, A, 0)
    return A


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


def cal_info_about_test_set_in_train_set(
    X_train, y_train, X_test, y_test, clf
):  # This checks how much information a finetuning set X_train/y_train contains about test set X_test/y_test that clf does not have already. #TODO:make idx version
    clf2 = deepcopy(clf)
    baseline_info = np.sum(prediction_info(y_test, clf2.predict_proba(X_test)))
    clf2.fit(X_train, y_train)
    rel_info = baseline_info - np.sum(
        prediction_info(y_test, clf2.predict_proba(X_test))
    )
    return rel_info


def pick_nearest2test(X_train, y_train, X_test, y_test):
    clf_knn = fit_dknn_toXy(X_train, y_train)
    full_train_test_info_residual = np.sum(
        prediction_info(y_test, clf_knn.predict_proba(X_test))
    )

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(X_train)
    _, train0_idxs = nn.kneighbors(X_test)
    train0_idxs = train0_idxs.squeeze()
    clf_knn.fit(X_train[train0_idxs], y_train[train0_idxs])
    selected_train_test_info_residual = np.sum(
        prediction_info(y_test, clf_knn.predict_proba(X_test))
    )
    print(
        "Info left in test set after training on full train set: ",
        full_train_test_info_residual,
        "Full training set size: ",
        len(y_train),
    )
    print(
        f"Info left in test set after training on {len(train0_idxs)} train set examples nearest to test set: ",
        selected_train_test_info_residual,
    )
    return train0_idxs


def prune_training_set(
    X, y, test_idx=None, k=10, return_smaller_sets=True, return_mask=True, thresh=0
):
    # maybe do this in rounds to avoid dropping all the examples in isolated clusters. Will need to do rounds anyway for big sets...
    if test_idx is not None:
        allidx = np.arange(X.shape[0])
        trainidx = np.setdiff1d(allidx, test_idx)
        X_train, y_train, X_test, y_test = (
            X[train_idx],
            y[train_idx],
            X[test_idx],
            y[test_idx],
        )

        tiny_idx, tiny_y = collect_min_set(y_train, k)
        tiny_x = X_train[tiny_idx]
        clf_knn = fit_dknn_toXy(
            tiny_x, tiny_y, k=k
        )  # fit an initial knn for whole train info estimation

        min_train_test_info_residual = np.sum(
            prediction_info(y_test, clf_knn.predict_proba(X_test))
        )
        full_train_test_info = cal_info_about_test_set_in_train_set(
            X_train, y_train, X_test, y_test, clf_knn
        )
        clf_knn.fit(X_train, y_train)
        full_train_test_info_residual = np.sum(
            prediction_info(y_test, clf_knn.predict_proba(X_test))
        )
        # TODO: More user-friendly names
        print("full train set size: ", y_train.size)
        print("full_train_test_info: ", full_train_test_info)
        print("min_train_test_info_residual: ", min_train_test_info_residual)
        print("full_train_test_info_residual: ", full_train_test_info_residual)
        print("full train knn score: ", clf_knn.score(X_test, y_test))
    else:
        X_train, y_train = X, y

    clf_knn_selfdrop = KNeighborsClassifier(
        n_neighbors=100, metric="euclidean", weights=dist_weight_ignore_self
    )  # TODO: optimize n_neighbors
    clf_knn_selfdrop.fit(X_train, y_train)
    info = prediction_info(y_train, clf_knn_selfdrop.predict_proba(X_train))

    ##not sure this is meaningful:
    train_self_info = info.sum(0)
    print("Train set self-info:", train_self_info)
    if info.ndim > 1:
        info = info.sum(1)

    if test_idx is not None:
        # sorted_X, sorted_y, info = order_samples_by_info(
        #     X_train, y_train, clf_knn_selfdrop, sort_info=False
        # )
        tiny_idx, tiny_y = collect_min_set(y_train, k)
        tiny_x = X_train[tiny_idx]
        clf_knn.fit(
            tiny_x, tiny_y
        )  # need a real function for this. One that is of minimum size but includes all classes.
        self_pruned_train_test_info = cal_info_about_test_set_in_train_set(
            X_train[info > thresh], y_train[info > thresh], X_test, y_test, clf_knn
        )
        print("Self-pruned train set size: ", y_train[info > thresh].size)
        print("self_pruned_train_test_info: ", self_pruned_train_test_info)
        clf_knn.fit(X_train[info > thresh, :], y_train[info > thresh])
        self_pruned_train_test_info_residual = np.sum(
            prediction_info(y_test, clf_knn.predict_proba(X_test))
        )
        print(
            "self_pruned_train_test_info_residual: ",
            self_pruned_train_test_info_residual,
        )
        print("self_pruned_score: ", clf_knn.score(X_test, y_test))

    selected_training_mask = (
        info > thresh if return_mask else np.arange(len(y_train))[info > thresh]
    )
    if return_smaller_sets:
        X_train_selected, y_train_selected = (
            X_train[selected_training_mask, :],
            y_train[selected_training_mask, :],
        )
        return X_train_selected, y_train_selected, selected_training_mask
    else:
        return selected_training_mask


def order_folds_by_entropy(
    X, y, clf, fold_idxs: list, reverse=True, info=False
):  # reverse = True sorts highest to lowest, so prioratize the training data that the model, as provided, knows the least about.
    Hs = []
    for idxs in fold_idxs:
        y_predicted = (
            clf.predict_proba(X.iloc[idxs])
            if isinstance(X, pd.DataFrame)
            else clf.predict_proba(X[idxs])
        )
        score = (
            np.sum(prediction_entropy(y[idxs], y_predicted))
            if not info
            else np.sum(prediction_info(y[idxs], y_predicted))
        )
        Hs.append(score)
    fold_idxs = [list(x) for x in fold_idxs]
    _, sorted_fold_idxs = zip(
        *sorted(zip(Hs, fold_idxs), reverse=reverse)
    )  # sorts on first in inner zip
    return [np.array(x) for x in sorted_fold_idxs]


def order_samples_by_info(
    X, y, clf, reverse=True, sort_info=False
):  # reverse = True sorts highest to lowest, so prioratize the training data that the model, as provided, knows the least about.

    y_predicted = clf.predict_proba(X)
    info = prediction_info(y, y_predicted)
    if info.ndim > 1:
        info = info.sum(
            1
        )  # need to do something more sophisticated to order by info in different class labels
        sorted_y = y[np.argsort(info), :]
    else:
        sorted_y = zip(*sorted(zip(info, y), reverse=reverse))

    if isinstance(X, list):
        _, sorted_X = zip(*sorted(zip(info, X), reverse=reverse))
    elif isinstance(X[0], np.ndarray):
        sorted_X = X[np.argsort(info)[::-1], :] if reverse else X[np.argsort(info), :]
    elif isinstance(X, pd.DataFrame):
        ascending = not reverse
        sorted_X = X.copy()
        sorted_X["info"] = info
        sorted_X = sorted_X.sort_values("info", ascending=ascending).drop(
            "info", axis=1
        )
    else:
        raise Exception(f"Variable X of unhandled type {type(X)}.")

    if sort_info:
        info.sort()
    if reverse:
        info = info[::-1]
    return sorted_X, np.array(sorted_y), info


# %%  Models  ###
def fit_dknn_toXy(X, y, k=10, metric="euclidean", self_exlude=False):
    weights = dist_weight_ignore_self if self_exlude else "distance"
    clf = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights)
    if isinstance(X, pd.DataFrame):
        clf.fit(X.values, y)
    else:
        clf.fit(X, y)
    return clf


def fit_dknn_toUMAP_reducer(reducer, y_train, k=10, metric="euclidean"):
    clf = KNeighborsClassifier(n_neighbors=k, metric=metric, weights="distance")
    clf.fit(reducer.embedding_, y_train)
    return clf


def collect_min_set(y, min_sz=0):
    # for collecting a minimum set for initial clf fitting which contains all classes
    idxs_out = []

    for y_val in np.unique(y, axis=0):
        idxs_out.append(np.where(np.all(y == y_val, axis=1))[0][0])

    while len(idxs_out) < min_sz:
        idxs_out = np.union1d(
            idxs_out, np.random.choice(np.arange(len(y)), len(idxs_out) - min_sz)
        )
    return list(idxs_out), y[idxs_out]


# %%  Routines  ###
def add_stratified_folds_test(
    X, y, clf, n_splits=10, verbose=True
):  # should do some without stratification to show the difference.
    kfold_idx_gen = StratifiedKFold(n_splits=n_splits).split(X, y)
    train_idx = np.array([], int)
    running_train, entropies, train_idx, scores, samples = [], [], [], [], []
    for k, (_, test_idx) in enumerate(kfold_idx_gen):
        running_train.append(test_idx)
    running_test = running_train.pop()
    if isinstance(X, pd.DataFrame):
        X = X.values
    for k, old_test_idx in enumerate(running_train):
        train_idx = np.append(train_idx, old_test_idx).astype(int)
        clf.fit(X[train_idx], y[train_idx])
        y_predicted = clf.predict_proba(X[running_test])
        y_true = y[running_test]
        H = prediction_entropy(y_true, y_predicted)
        score = clf.score(X[running_test], y_true)
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


def add_best_fold_first_test(
    X, y, clf, n_splits=10, X_test=None, verbose=True
):  # should do some without stratification to show the difference.
    kfold_idx_gen = StratifiedKFold(n_splits=n_splits).split(X, y)
    train_idx = np.array([], int)
    running_train, entropies, train_idx, scores, samples = [], [], [], [], []
    for k, (_, test_idx) in enumerate(kfold_idx_gen):
        running_train.append(test_idx)
    running_train = [list(x) for x in running_train]
    if isinstance(X, pd.DataFrame):
        X = X.values
    if X_test is None:
        running_test = running_train.pop()
    elif isinstance(X_test, pd.DataFrame):
        running_test = X_test.index
    else:
        running_test = X_test

    try:
        check_is_fitted(clf)
    except NotFittedError as exc:
        print("Unfiitted estimator provided. Fitting on first fold.")
        idx0 = running_train.pop(0)
        clf.fit(X[idx0], y[idx0])
    iters = len(running_train)

    for k in trange(iters):
        best_fold_idxs = order_folds_by_entropy(X, y, clf, running_train)[0]
        train_idx = np.append(train_idx, best_fold_idxs).astype(int)
        running_train.remove(list(best_fold_idxs))

        clf.fit(X[train_idx], y[train_idx])
        y_predicted = clf.predict_proba(X[running_test])
        y_true = y[running_test]
        H = prediction_entropy(y_true, y_predicted)
        score = clf.score(X[running_test], y_true)
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


def extraction_rate(clf, X_train, y_train, n=1, entropy=True, mean=True):
    # how much info is left in the training set after fitting. may not want it to be zero. THIS IS WRONG. WANT INFO BEFORE/AFTER FITTING
    A = []
    for i in range(n):
        clf.fit(X_train, y_train)
        a = (
            prediction_entropy(y_train, clf.predict_proba(X_train))
            if entropy
            else np.sum(prediction_info(y_train, clf.predict_proba(X_train)))
        )
        A.append(a)

    out = np.mean(A) if mean else A
    return out


# Do a training set selection routine with a dknn or provided clf, picking a training set with sufficient info (as calculated at the outset) about the test set.


# training routine,
def train_best_fold_first_test(
    X, y, clf, n_splits=100, verbose=False, tol=10, X_test=None
):  # should do some without stratification to show the difference. Should try selecting training set with info (as justged by inital clf) equal to ignorance in test set (ditto)
    kfold_idx_gen = StratifiedKFold(n_splits=n_splits).split(X, y)
    train_idx = np.array([], int)
    running_train, entropies, train_idx, scores, samples = [], [], [], [], []
    for k, (_, test_idx) in enumerate(kfold_idx_gen):
        running_train.append(test_idx)
    running_train = [list(x) for x in running_train]
    if isinstance(X, pd.DataFrame):
        X = X.values
    if X_test is None:
        running_test = running_train.pop()
    elif isinstance(X_test, pd.DataFrame):
        running_test = X_test.index
    else:
        running_test = X_test
    y_true = y[running_test]

    try:
        check_is_fitted(clf)
    except NotFittedError as exc:
        print("Unfiitted estimator provided. Fitting on first fold.")
        idx0 = running_train.pop(0)
        clf.fit(X[idx0], y[idx0])
    iters = len(running_train)
    best_idxs = []
    last_best_k = 0
    best_H = np.inf

    for k in trange(iters):
        best_fold_idxs = order_folds_by_entropy(X, y, clf, running_train)[0]
        train_idx = np.append(train_idx, best_fold_idxs).astype(int)
        running_train.remove(list(best_fold_idxs))

        clf.fit(X[train_idx], y[train_idx])
        y_predicted = clf.predict_proba(X[running_test])
        H = prediction_entropy(y_true, y_predicted)
        score = clf.score(X[running_test], y_true)
        if H < best_H:
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
            print(
                f"No improvement found after adding {k - last_best_k} folds ({len(train_idx) - len(best_idxs)} samples). Fitting on {last_best_k} folds, {len(best_idxs)} smaples."
            )
            break

    clf.fit(X[best_idxs], y[best_idxs])
    return samples, entropies, scores, best_idxs
