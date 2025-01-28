#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core functions of the Better Than Vibes module. Include the information theoretic
functions, custom types for checking, and self-excluding dknn clasifier setup.

Created on Tue Oct  1 17:48:54 2024

@author: JSeaward
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from copy import deepcopy

#### Typing stuff
from pandas import DataFrame
from typing import Optional, Sequence, Union, Callable, Tuple
from .custom_types import (
    Single_Label_Set,
    Multi_Label_Set,
    Label_Set,
    Single_Prediction_Set,
    Mulilabel_Prediction_Set,
    Data_Features,
    Prediction_Set,
)


# from sklearn.base import BaseEstimator

# %%
def _prediction_info_singlelabel(
    y_true: Single_Label_Set,
    y_predicted: Single_Prediction_Set,
    discount_chance: bool = False,
) -> np.ndarray:
    """
    The information remaining in a label given a predicted label.
    If the predicted label is 100% certain and correct, the true label contains no information,
    Function assumes that the classification is a single label, which may be multi-class.

    Parameters
    ----------
    y_true : Sequence[int]
        Corrent labels.
    y_predicted : Simgle_Prediction_Set
        Predicted Labels.
    discount_chance : bool, optional
        Whether or not to discount the effects of chance when calculating the information provided by a label.
        Useful when trying to estimate how much a model has learned.
        Apriori, a classifier has a chance of 1/class_number of guessing the right answer.
        Should that floor be counted as information provided by the label?
        The default is False.

    Returns
    -------
    A : np.ndarray, dtype = float
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


def _prediction_info_multilabel(
    y_true: Multi_Label_Set,
    y_predicted: Mulilabel_Prediction_Set,
    discount_chance: bool = False,
) -> np.ndarray:
    """
    The information remaining in a label given a predicted label.
    For multi-label models.

    Parameters
    ----------
    y_true : Union[Sequence[np.ndarray,dtype=int], np.ndarray[int]]
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
    np.array dtype=float
        The information remaining in each label in y_true given y_predicted.

    """
    if isinstance(y_predicted, np.ndarray):
        y_predicted = [x.squeeze() for x in np.split(y_predicted, y_predicted.shape[0])]
    A = []
    for y_t, y_p in zip(y_true, y_predicted):
        A.append(
            _prediction_info_singlelabel(y_t, y_p, discount_chance=discount_chance)
        )
    return np.array(A)


def prediction_info(
    y_true: Label_Set,
    y_predicted: Prediction_Set,
    discount_chance: bool = False,
) -> np.ndarray:
    """
    Unified caller for information content of single and multi-label
    true/predicted label sets.-

    Parameters
    ----------
    y_true : Label_Set
        Correct labels.
    y_predicted : Prediction_Set
        Predicted Labels.
    discount_chance : bool, optional
        Whether or not to discount the effects of chance when calculating the information provided by a label.
        Useful when trying to estimate how much a model has learned.
        Apriori, a classifier has a chance of 1/class_number of guessing the right answer.
        Should that floor be counted as information provided by the label?
        The default is False.

    Returns
    -------
    np.ndarray, dtype=float.
        The information remaining in each label in y_true given y_predicted.

    """
    if np.array(y_true).ndim > 1:
        return _prediction_info_multilabel(
            y_true, y_predicted, discount_chance=discount_chance
        )

    return _prediction_info_singlelabel(
        y_true, y_predicted, discount_chance=discount_chance
    )


def _chance_info(
    y: Union[Single_Label_Set, Single_Prediction_Set],
    class_num: Optional[int] = None,
    use_freq: bool = True,
) -> Union[float, int]:
    if use_freq:
        # assumes y is a label set, not a prediction set
        assert y[0] == int(
            y[0]
        ), "Please pass a label set (integer valuesd) to use frequnecy statistics."
        classes, counts = np.unique(y, return_counts=True)
        classes = list(classes.astype(int))
        p = np.array([counts[classes.index(lbl)] / len(y) for lbl in y])
        info = -np.log2(p).sum()
    else:
        if class_num is None:
            class_num = len(np.unique(y))
        p = 1 / class_num
        info = -np.log2(p) * len(y)
    return info


def _chance_info_multilabel(
    y: Union[Multi_Label_Set, Mulilabel_Prediction_Set],
    class_num: Optional[int] = None,
    use_freq: bool = True,
) -> np.ndarray:
    # expects y to be N x class_num with each row having the form [p(c1), p(c2), ...]
    if use_freq:
        # assumes y is a label set, not a prediction set
        assert y.ravel()[0] == int(
            y.ravel()[0]
        ), "Please pass a label set (integer valuesd) to use frequnecy statistics."
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
    y: Union[Label_Set, Prediction_Set],
    class_num: Optional[int] = None,
    use_freq: bool = True,
) -> Union[np.ndarray, float, int]:
    """
    Calculate the information remaining in a set of labels, given simple statistical assumptions.
    For example, there is 25% chance of classifying something correctly out of four classes.
    Therefore, a 4-class label provides 2 bits of information.

    Parameters
    ----------
    y : Union[Label_Set,Prediction_Set]
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
    if info_rate < 0:
        print("INFO: The model is WORSE than guessing.")
    return info_rate


def estimate_rateVSchance(
    X: Data_Features,
    y: Label_Set,
    clf=None,
    use_freq: bool = True,
    metric: Union[str, Callable] = "euclidean",
) -> float:
    """
    Estimate the information extraction rate of a model on a dataset.

    Parameters
    ----------
    X : M x N Data_Features
        Data points, with M samples and N features.
    y : Label_Set
        Data labels.
    clf : sklearn-style Classifier, optional
        Classifier to use. Needs only a clf.predict_proba method.
        The default is None, where a self-excluing dknn will be used to estimate
        the self-information of the datraset.
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
    _clf = fit_dknn_toXy(X, y, metric=metric, self_exclude=True) if clf is None else clf
    return _extraction_rateVSchance(X, y, _clf, use_freq)


def extraction_rate(
    clf, X_train: Data_Features, y_train: Label_Set, n: int = 1, rate: bool = True, refit:bool=False
) -> Union[Sequence[float],float]:
    """
    Fit classifier to a training set n times and get the information extraction rate each time.

    Parameters
    ----------
    clf : sklearn-style Classifier
        Classifier to use. Needs .fit(X,y) and .predict_proba(X) methods.
    X_train : M x N Data_Features
        Data points, with M samples and N features.
    y_train : Label_Set
        M data labels for X_train.                     
    n : int, optional
        Number of times to re-fit the classifier.
        The default is 1, suitable for deterministic classifiers.
    rate : bool, optional
        Whether or not to use the percent of the info present in the train set
        with simple statistics or the total info left after training.
        The default is True, using the rate against chance information.
    refit : bool, optional
        Whether or not to refit the passed classifier. The default is False,
        making a deepcopy and leaving the passed clasifier unchanged.
    Returns
    -------
    Sequence[float] or float
        Inforamtion rates/totals for each of the n trials.

    """
    assert n > 0
    A = []
    _clf = clf if refit else deepcopy(clf)

    for _ in range(n):
        _clf.fit(X_train, y_train)
        a = (
            estimate_rateVSchance(X_train, y_train, clf=_clf)
            if rate
            else np.sum(prediction_info(y_train, _clf.predict_proba(X_train)))
        )
        A.append(a)

    return np.array(A) if n>1 else A[0]


def order_folds(
    X: Data_Features,
    y: Label_Set,
    clf,
    fold_idxs: list[Sequence[int]],
    rate: bool = True,
) -> list[np.ndarray]:
    """
    Order folds from a k-folds strategy by thier information value to trained classifier.
    Sorts folds high to low and giving the folds the classifier is worst at first.
    Does not retrain classifier.

    Parameters
    ----------
    X : Data_Features
        Data features over which the folds are defined.
    y : Label_Set
        labels for X.
    clf : sklearn-style classifier
        Trained/fitted classifier to test. Needs only the .predict_proba(X) method.
    fold_idxs : list[Sequence[int]]
        Fold indices as provided by methods such as from sklearn.model_selection.StratifiedKFold.
        A list K folds long, each a sequence of inddecies in (X,y) which constitute a fold.
    rate : bool, optional
        Whether to sort on the information rate (as opposed to total info).
        The default is True. Passing false will use the total information in the fold.
        Using total information will be sensitive to fold size, with bigger folds
        having more total information

    Returns
    -------
    list[np.ndarray]
        List of index sequences that will sort the folds. Constructed to have th same form as
        fold_idxs from sklearn.

    """
    # reverse = True sorts highest to lowest, so prioratize the training data that the model, as provided, knows the least about.
    Is = []
    for idxs in fold_idxs:
        y_predicted = (
            clf.predict_proba(X.iloc[idxs])
            if isinstance(X, DataFrame)
            else clf.predict_proba(X[idxs])
        )
        score = (
            np.sum(estimate_rateVSchance(X[idxs], y[idxs], clf=clf))
            if rate
            else np.sum(prediction_info(y[idxs], y_predicted))
        )
        Is.append(score)
    fold_idxs = [list(x) for x in fold_idxs]
    _, sorted_fold_idxs = zip(
        *sorted(zip(Is, fold_idxs), reverse=True)
    )  # sorts on first in inner zip
    return [np.array(x) for x in sorted_fold_idxs]


##### Model and data setup functions  ####


def _dist_weight_ignore_self(dist: np.array) -> np.array:
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
    dist : np.array, dtype = float
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


def fit_dknn_toXy(
    X: Data_Features,
    y: Label_Set,
    k: Optional[int] = None,
    metric: Union[str, Callable] = "euclidean",
    self_exclude: bool = False,
) -> KNeighborsClassifier:
    """
    Fit a distance-weighted k nearest-nieghbors classifier to the data (x,y)

    Parameters
    ----------
    X : Data_Features
        Data features M x N samples.
    y : Label_Set
        M labels for X. .
    k : Optional[int], optional
        Number of labels for the classifier to consider.
        The default is None, correspondin to 2 neighbors per feature
        (1 added if self_exclude=True).
    metric : Union[str, Callable], optional
        scipy distance metric to use if building the d-knn classifier.
        The default is "euclidean".
    self_exclude : bool, optional
        Whether to use the custom weight function '_dist_weight_ignore_self'
        which will ignore data points with distance=0 at infereence time.
        The default is False.

    Returns
    -------
    KNeighborsClassifier
        d-knn classifier fit to (x,y).

    """
    if k is None:
        k = X.shape[1] * 2
        if self_exclude:
            k += 1
    weights = _dist_weight_ignore_self if self_exclude else "distance"
    clf = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights)
    if isinstance(X, DataFrame):
        clf.fit(X.values, y)
    else:
        clf.fit(X, y)
    return clf


def collect_min_set(y: Label_Set, min_sz: int = 0) -> Tuple[list[int], Label_Set]:
    """
    Collect a minimum set for training, including one of each class label in
    the passed label set, y. Useful when you need a functionally untrained classifier
    but sklearn-style ones must be fit to something.

    Parameters
    ----------
    y : Label_Set
        Data labels.
    min_sz : int, optional
        Minimum size of the returned training set.
        The default is 0, returning a set as large as the number of unique labels in y.

    Returns
    -------
    idxs : list[int]
        list of the indices in y that form the minimal set.
    labels : Label_Set
        The labels of the minimal set.

    """
    # for collecting a minimum set for initial clf fitting which contains all classes
    idxs_out = []

    if np.ndim(y) > 1:
        for y_val in np.unique(y, axis=0):
            idxs_out.append(np.where(np.all(y == y_val, axis=1))[0][0])
    else:
        for y_val in np.unique(y):
            idxs_out.append(np.where(y == y_val)[0][0])

    while len(idxs_out) < min_sz:
        idxs_out = np.union1d(
            idxs_out, np.random.choice(np.arange(len(y)), min_sz - len(idxs_out))
        )
    idxs, labels = list(idxs_out), y[idxs_out]
    return idxs, labels
