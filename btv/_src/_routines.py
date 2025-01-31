#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some big, messy routines packaged as functions. See demo notebook for clearer understanding.

Created on Sun Jan 26 11:28:47 2025

@author: rakshat
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy
from tqdm import trange, tqdm
from matplotlib import pyplot as plt

from ._src import (
    prediction_info,
    fit_dknn_toXy,
    estimate_rateVSchance,
    order_folds,
    chance_info,
)
from ._selection import prune_by_info

from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from typing import Sequence, Union, Tuple
from pandas import DataFrame
from .custom_types import Data_Features, Label_Set


# %%
def pick_nearest2test(
    X_train: Data_Features,
    y_train: Label_Set,
    X_test: Data_Features,
    y_test: Label_Set,
) -> Sequence[int]:
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


def scan_info_thresh(
    X_train: Data_Features,
    y_train: Label_Set,
    X_test: Data_Features,
    y_test: Label_Set,
    clf,
    max_info: Union[int, float] = 1,
    info_steps: int = 100,
    plot=True,
    score_metric="Accuracy",
    keep_hull=False,
) -> Tuple[list[float], list[int], float]:
    scores, lengths = [], []
    thresholds = np.arange(0, max_info, max_info / info_steps)
    for th in tqdm(thresholds, desc="Scanning information threshold..."):
        idx_pruned = prune_by_info(X_train, y_train, thresh=th, keep_hull=keep_hull)
        if len(idx_pruned) == 0:
            while len(lengths) < info_steps:
                lengths.append(0)
                scores.append(0)
            print(f"INFO: No points remaining with info>={th}.")
            break
        clf.fit(X_train[idx_pruned], y_train[idx_pruned])
        lengths.append(len(idx_pruned) / len(y_train))
        scores.append(clf.score(X_test, y_test))

    if plot:
        plt.xlabel("Information threshold [bits]")
        plt.title("Classification score with decresing training set information")
        score_line = plt.plot(thresholds, scores, label="Score")
        ax1 = plt.gca()
        ax1.set_ylabel(f"Score ({score_metric})")

        ax2 = ax1.twinx()
        length_line = ax2.plot(thresholds, lengths, color="g", label="Train set size")
        ax2.set_ylabel("Fraction of training set used")

        lines = score_line + length_line
        lbls = [x.get_label() for x in lines]
        plt.legend(lines, lbls)

        plt.show()
        print(
            f"Max score of {max(scores):0.5f} at threshold of {thresholds[np.argmax(scores)]}."
        )
    return scores, lengths, thresholds[np.argmax(scores)]


def cal_info_about_test_set_in_finetune_set(
    X_train: Data_Features,
    y_train: Label_Set,
    X_ft: Data_Features,
    y_ft: Label_Set,
    X_test: Data_Features,
    y_test: Label_Set,
    clf=None,
    discount_chance: bool = True,
) -> float:
    """
    Calculate the infornation about a test set (X_test,y_test) a fine-tuning set (X_ft,y_ft) provides,
    given training has already been performed on (X_train,y_train)

    Parameters
    ----------
    X_train : np.ndarray
        Data features for training.
    y_train : Sequence[int]
        Data labels for training.
    X_ft : np.ndarray
        Data features for fine-tuning.
    y_ft : Sequence[int]
        Data labels for fine-tuning.
    X_test : np.ndarray
        Data features for testing.
    y_test : Sequence[int]
        Data labels for testing.
    clf : sklearn style classifier, optional
        Classifier to be tested. Requires .fit(X,y) and .predict_proba(X) methods
        The default is None, where a dknn will be used.
    discount_chance : bool, optional
        Whether or not to discount the effects of chance when calculating the information.
        See docstring of `prediction_info` for more detail.
        The default is True.

    Returns
    -------
    rel_info : float
        Difference in information left in the Test set after training with the fine-tuning set.
        Lower is better. Measured in bits.

    """
    if clf is None:
        _clf = fit_dknn_toXy(X_train, y_train)
    else:
        _clf = deepcopy(clf)
    baseline_info = np.sum(
        prediction_info(
            y_test, _clf.predict_proba(X_test), discount_chance=discount_chance
        )
    )
    _clf.fit(np.vstack([X_train, X_ft]), np.append(y_train, y_ft))
    rel_info = baseline_info - np.sum(
        prediction_info(
            y_test, _clf.predict_proba(X_test), discount_chance=discount_chance
        )
    )
    return rel_info


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
    elif isinstance(X, DataFrame):
        ascending = not reverse
        sorted_X = X.copy()
        sorted_X["info"] = info
        sorted_X = sorted_X.sort_values("info", ascending=ascending).drop(
            "info", axis=1
        )
    else:
        raise TypeError(f"Variable X of unhandled type {type(X)}.")

    if sort_info:
        info.sort()
    if reverse:
        info = info[::-1]
    return sorted_X, np.array(sorted_y), info


def add_stratified_folds_test(
    X, y, clf, n_splits=10, verbose=True
):  # should do some without stratification to show the difference.
    _clf = deepcopy(clf)
    # y_fold = y.sum(1) if y.ndim > 1 else y
    kfold_idx_gen = StratifiedKFold(n_splits=n_splits).split(X, y)
    train_idx = np.array([], int)
    running_train, entropies, train_idx, scores, samples = [], [], [], [], []
    for k, (_, test_idx) in enumerate(kfold_idx_gen):
        running_train.append(test_idx)
    running_test = running_train.pop()
    if isinstance(X, DataFrame):
        X = X.values
    for k, old_test_idx in enumerate(running_train):
        train_idx = np.append(train_idx, old_test_idx).astype(int)
        _clf.fit(X[train_idx], y[train_idx])
        y_true = y[running_test]
        H = estimate_rateVSchance(X[running_test], y[running_test], clf=_clf)
        score = _clf.score(X[running_test], y_true)
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
    if isinstance(X, DataFrame):
        X = X.values
    if X_test is None:
        running_test = running_train.pop()
    elif isinstance(X_test, DataFrame):
        running_test = X_test.index
    else:
        running_test = X_test

    try:
        check_is_fitted(clf)
    except NotFittedError:
        print("Unfiitted estimator provided. Fitting on first fold.")
        idx0 = running_train.pop(0)
        clf.fit(X[idx0], y[idx0])
    iters = len(running_train)

    for k in trange(iters):
        best_fold_idxs = order_folds(X, y, clf, running_train)[0]
        train_idx = np.append(train_idx, best_fold_idxs).astype(int)
        running_train.remove(list(best_fold_idxs))

        clf.fit(X[train_idx], y[train_idx])
        y_true = y[running_test]
        H = estimate_rateVSchance(X[running_test], y_true, clf=clf)
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


def train_best_fold_first_test(
    X,
    y,
    clf_in,
    n_splits=100,
    verbose=False,
    tol=10,
    test_idx: Union[Sequence[int], DataFrame] = None,
):  # should do some without stratification to show the difference. Should try selecting training set with info (as justged by inital clf) equal to ignorance in test set (ditto)
    clf = deepcopy(clf_in)
    kfold_idx_gen = StratifiedKFold(n_splits=n_splits).split(X, y)
    train_idx = np.array([], int)
    running_train, rates, train_idx, scores, samples = [], [], [], [], []
    for k, (_, test_idx) in enumerate(kfold_idx_gen):
        running_train.append(test_idx)
    running_train = [list(x) for x in running_train]
    if isinstance(X, DataFrame):
        X = X.values
    if test_idx is None:
        running_test = running_train.pop()
    elif isinstance(test_idx, DataFrame):
        running_test = test_idx.index
    else:
        running_test = test_idx
    y_true = y[running_test]

    try:
        check_is_fitted(clf)
    except NotFittedError:
        print("Unfiitted estimator provided. Fitting on first fold.")
        idx0 = running_train.pop(0)
        clf.fit(X[idx0], y[idx0])
    iters = len(running_train)
    best_idxs = []
    last_best_k = 0
    best_r = 0

    for k in trange(iters):
        best_fold_idxs = order_folds(X, y, clf, running_train)[0]
        train_idx = np.append(train_idx, best_fold_idxs).astype(int)
        running_train.remove(list(best_fold_idxs))

        clf.fit(X[train_idx], y[train_idx])
        r = estimate_rateVSchance(X[running_test], y_true, clf=clf)
        score = clf.score(X[running_test], y_true)
        if r > best_r:
            best_idxs = train_idx
            last_best_k = k
            best_r = r
        rates.append(r)
        scores.append(score)
        samples.append(len(train_idx))
        if verbose:
            print(
                f"Fold : {k+1}, " f"Test set info rate : {r}",
                f"Train samples : {len(train_idx)}",
                f"Score : {score}",
            )
        if k - last_best_k >= tol:
            print(
                f"No improvement in information rate found after adding {k - last_best_k} folds ({len(train_idx) - len(best_idxs)} samples). Fitting on {last_best_k} folds, {len(best_idxs)} smaples."
            )
            break

    clf_in.fit(X[best_idxs], y[best_idxs])
    print(
        f"Best score: {clf_in.score(X[best_idxs], y[best_idxs])} on {len(train_idx) - len(best_idxs)} samples."
    )
    # samples, rates, and scores are for plotting. Samples is for the x axis
    return samples, rates, scores, best_idxs
