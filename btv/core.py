#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 17:48:54 2024

@author: JSeaward
"""

from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import (
    StratifiedKFold,
)  # use X = np.zeros(n_samples) in .split

# if you have indeces and want knns from large set, can make an idx array and NearestVectorCaller._call_vec_set
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import trange

from typing import Optional, Sequence, Union, Callable
label_set = Union[Sequence[int],np.ndarray[Sequence[int]]]  ## start here
# from sklearn.base import BaseEstimator

# %%


def dist_weight_ignore_self(dist: np.array) -> np.array[float]:
    """
    Custom weighting function for a dknn that ignores points already in the fit set.
    That means that if the distance to a point in the fit set is 0, that point is ignored
    and the estimation is made on the distance to the remaining points in training set.
    Adapted from 'distance' weighting function at https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/neighbors/_base.py#L81

    Parameters
    ----------
    dist : np.array
        Array of distances to use a weights.

    Returns
    -------
    dist : np.array[float]
        Weights for classifier.

    """
    if dist.dtype is np.dtype(object):
        for point_dist_i, point_dist in enumerate(dist):
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


def prediction_info(
    y_true: Sequence[int], y_predicted: np.ndarray[float], discount_chance: bool = False
) -> np.ndarray[float]:
    """
    The information remaining in a label given a predicted label.
    If the predicted label is 100% certain and correct, the true label contains no information,
    Function assumes that the classification is a single label, which may be multi-class.

    Parameters
    ----------
    y_true : Sequence[int]
        Corrent labels.
    y_predicted : np.ndarray[float]
        Predicted Labels.
    discount_chance : bool, optional
        Whether or not to discount the effects of chance when calculating the information provided by a label.
        Useful when trying to estimate how much a model has learned.
        Apriori, a classifier has a chance of 1/class_number of guessing the right answer.
        Should that floor be counted as information provided by the label?
        The default is False.

    Returns
    -------
    A : np.ndarray
        The information remaining in each label in y_true given y_predicted.

    """
    #
    A = []
    classes = y_predicted.shape[1] - 1
    for pred, true in zip(y_predicted, y_true):
        # the +1 and the denominator cause a sure, wrong anser to provide 1 bit instead of infinite information
        I = (
            -np.log2(pred[true] / (2**classes) + 1 / (2**classes))
            if not discount_chance
            else -np.log2(pred[true] + 1 / 2**classes)
        )  # bits
        A.append(I)
    A = np.array(A)
    A = np.where(A >= 0, A, 0)
    return A


def prediction_info_multilabel(
    y_true: Union[Sequence[np.ndarray[int]], np.ndarray[int]],
    y_predicted: Union[Sequence[np.ndarray[float]], np.ndarray[float]],
    discount_chance: bool = False,
) -> np.ndarray[float]:
    """
    The information remaining in a label given a predicted label.
    For multi-label models.

    Parameters
    ----------
    y_true : Union[Sequence[np.ndarray[int]], np.ndarray[int]]
        Correct labels.
    y_predicted : Union[Sequence[np.ndarray[float]], np.ndarray[float]]
        Predicted Labels.
    discount_chance : bool, optional
        Whether or not to discount the effects of chance when calculating the information provided by a label.
        Useful when trying to estimate how much a model has learned.
        Apriori, a classifier has a chance of 1/class_number of guessing the right answer.
        Should that floor be counted as information provided by the label?
        The default is False.

    Returns
    -------
    np.array
        The information remaining in each label in y_true given y_predicted.

    """
    if isinstance(y_predicted, np.ndarray):
        y_predicted = [x.squeeze() for x in np.split(y_predicted, y_predicted.shape[0])]
    A = []
    for y_t, y_p in zip(y_true, y_predicted):
        A.append(prediction_info(y_t, y_p))
    return np.array(A)


def _chance_info(
    y, class_num: Optional[int] = None, use_freq: bool = True
) -> Union[float, int]:
    if use_freq:
        # assumes y is a label set, not a prediction set
        classes, counts = np.unique(y, return_counts=True)
        classes = list(classes)
        p = np.array([counts[classes.index(lbl)] / len(y) for lbl in y])
        info = -np.log2(p).sum()
    else:
        if class_num is None:
            class_num = len(np.unique(y))
        p = 1 / class_num
        info = -np.log2(p) * len(y)
    return info


def _chance_info_multilabel(
    y, class_num: Optional[int] = None, use_freq: bool = True
) -> np.ndarray:
    # expects y to be N x class_num with each row having the form [p(c1), p(c2), ...]
    if use_freq:
        # assumes y is a label set, not a prediction set
        freq = np.sum(y, axis=0) / len(y)
        info = -np.log2(freq) * len(y)
    else:
        if class_num is None:
            class_num = np.array(y).shape[1]
        p = 1 / class_num
        info = -np.log2(p) * len(y)
        info = np.array([info] * class_num)
    return info


def chance_info(
    y: Union[Sequence[Union[int, float]], np.ndarray],
    class_num: Optional[int] = None,
    use_freq: bool = True,
) -> Union[np.ndarray, float, int]:
    """
    Calculate the information remaining in a set of labels, given simple statistical assumptions.
    For example, there is 25% chance of classifying something correctly out of four classes.
    Therefore, a 4-class label provides 2 bits of information.

    Parameters
    ----------
    y : Union[Sequence[Union[int,float]],np.ndarray]
        A set of labels.
    class_num : Optional[int], optional
        Force an number of classes. The default is None, letting the function infer it from y.
    use_freq : bool, optional
        Calculte the probability of each value of y to be its frequency in y.
        The default is True, requiring y to be a label set containing integer labels.
        Passing 'False' will use only the shape of y and assume even distribution of label values.

    Returns
    -------
    Union[np.ndarray,float,int]
        Information remaining in the set given simple s tatistics.

    """
    args = [y, class_num, use_freq]
    return _chance_info_multilabel(*args) if np.ndim(y) > 1 else _chance_info(*args)


def _extraction_rateVSchance(X, y, _clf, use_freq: bool = True) -> float:
    info_baseline = chance_info(y, use_freq=use_freq)
    info = prediction_info(y, _clf.predict_proba(X), discount_chance=False).sum()
    info_rate = (info_baseline - info) / info_baseline
    assert info_rate >= 0, "The model is WORSE than guessing?"
    return info_rate


def estimate_rateVSchance(
    X: np.ndarray,
    y: Sequence[int],
    clf=None,
    use_freq: bool = True,
    metric: Union[str, Callable] = "euclidean",
) -> float:
    """
    Estimate the information extraction rate of a model on a dataset.

    Parameters
    ----------
    X : M x N np.ndarray
        Data points, with M samples and N features.
    y : Sequence[int]
        Data labels.
    clf : sklearn-style Classifier, optional
        Classifier to use. Needs only a clf.predict_proba method.
        The default is None, where a self-excluing dknn will be used.
        See documentation of dist_weight_ignore_self for more details.
    use_freq : bool, optional
        Use the frequency of label values to calculate chance info rate.
        The default is True.
    metric : Union[str,Callable], optional
        scipy distance metric to use if building the d-knn classifier.
        The default is "euclidean".

    Returns
    -------
    float
        Percentage of the inforamtion in the dataset (X,y) which clf has learned.

    """
    _clf = fit_dknn_toXy(X, y, metric=metric, self_exlude=True) if clf is None else clf
    return _extraction_rateVSchance(X, y, _clf, use_freq)


def cal_info_about_test_set_in_finetune_set(
    X_train: np.ndarray,
    y_train: Sequence[int],
    X_ft: np.ndarray,
    y_ft: Sequence[int],
    X_test: np.ndarray,
    y_test: Sequence[int],
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


def pick_nearest2test(
    X_train: np.ndarray,
    y_train: Sequence[int],
    X_test: np.ndarray,
    y_test: Sequence[int],
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


def order_folds_by_entropy(
    X:np.ndarray, y, clf, fold_idxs: list, reverse=True, info=False
):  # reverse = True sorts highest to lowest, so prioratize the training data that the model, as provided, knows the least about.
    Hs = []
    for idxs in fold_idxs:
        y_predicted = (
            clf.predict_proba(X.iloc[idxs])
            if isinstance(X, pd.DataFrame)
            else clf.predict_proba(X[idxs])
        )
        score = (
            np.sum(estimate_rateVSchance(X[idxs], y[idxs], clf=clf))
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
def fit_dknn_toXy(
    X: np.ndarray,
    y: Sequence[int],
    k: Optional[int] = None,
    metric: Union[str, Callable] = "euclidean",
    self_exlude: bool = False,
):
    if k is None:
        k = X.shape[1] * 2  # 2 neighbors per feature
    weights = dist_weight_ignore_self if self_exlude else "distance"
    clf = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights)
    if isinstance(X, pd.DataFrame):
        clf.fit(X.values, y)
    else:
        clf.fit(X, y)
    return clf


def collect_min_set(y, min_sz=0):
    # for collecting a minimum set for initial clf fitting which contains all classes
    idxs_out = []

    for y_val in np.unique(y, axis=0):
        idxs_out.append(np.where(np.all(y == y_val, axis=1))[0][0])

    while len(idxs_out) < min_sz:
        idxs_out = np.union1d(
            idxs_out, np.random.choice(np.arange(len(y)), min_sz - len(idxs_out))
        )
    return list(idxs_out), y[idxs_out]


# %%  Routines  ###
def add_stratified_folds_test(
    X, y, clf, n_splits=10, verbose=True
):  # should do some without stratification to show the difference.
    y_fold = y.sum(1) if y.ndim > 1 else y
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
        y_true = y[running_test]
        H = estimate_rateVSchance(X[running_test], y[running_test], clf=clf)
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
    except NotFittedError:
        print("Unfiitted estimator provided. Fitting on first fold.")
        idx0 = running_train.pop(0)
        clf.fit(X[idx0], y[idx0])
    iters = len(running_train)

    for k in trange(iters):
        best_fold_idxs = order_folds_by_entropy(X, y, clf, running_train)[0]
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


def extraction_rate(clf, X_train, y_train, n=1, entropy=True, mean=True):
    # how much info is left in the training set after fitting. may not want it to be zero. THIS IS WRONG. WANT INFO BEFORE/AFTER FITTING
    A = []
    for i in range(n):
        clf.fit(X_train, y_train)
        a = (
            estimate_rateVSchance(X_train, y_train, clf=clf)
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
    except NotFittedError:
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
        H = estimate_rateVSchance(X[running_test], y_true, clf=clf)
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
