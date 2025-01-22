#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:08:39 2025

@author: rakshat
"""

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

from typing import Optional, Sequence, Union, Tuple
from sklearn.base import BaseEstimator

from btv import prediction_info, fit_dknn_toXy, chance_info


# %%


########## Information theoretic selection #########
def make_low_info_mask(
    X: np.ndarray,
    y: Sequence[int],
    metric: str = "euclidean",
    thresh: Union[int, float] = 0,
) -> Sequence[bool]:
    clf_knn_selfdrop = fit_dknn_toXy(
        X, y, k=X.shape[1] * 2, metric=metric, self_exlude=True
    )
    info = prediction_info(
        y,
        clf_knn_selfdrop.predict_proba(X),
        discount_chance=False,
    )
    return info <= thresh


def get_exterior_pnts(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    chull = ConvexHull(X)
    return X[chull.vertices], chull.vertices


def prune_by_knn_info(
    X: np.ndarray,
    y: Sequence[int],
    clf: Optional[BaseEstimator] = None,
    metric: str = "euclidean",
    thresh: Union[int, float] = 0,
) -> np.ndarray:
    if clf is None:
        info_mask = make_low_info_mask(X, y)
    else:
        info = prediction_info(
            y,
            clf.predict_proba(X),
            discount_chance=False,
        )
        info_mask = info <= thresh
    return np.where(~info_mask)[0].astype(int)


def prune_by_info(
    X,
    y,
    clf: Optional[BaseEstimator] = None,
    metric: str = "euclidean",
    thresh: Union[int, float] = 0,
    keep_hull=False,
) -> np.ndarray:
    out_idxs = prune_by_knn_info(X, y, clf, metric, thresh)
    if keep_hull:
        _, hull_idxs = get_exterior_pnts(X)
        out_idxs = np.intersect1d(hull_idxs, out_idxs)
    return out_idxs.astype(int)


########### Geometric functions + selection ##########
def _estimate_overflow(X1: np.ndarray, X2: np.ndarray) -> float:
    return ConvexHull(np.vstack([X1, X2])).volume - ConvexHull(X2).volume


def estimate_overflow(
    X1: np.ndarray, X2: np.ndarray, normalize_by_X1: bool = False
) -> float:
    overflow = _estimate_overflow(X1, X2)
    return overflow / ConvexHull(X1).volume if normalize_by_X1 else overflow


def estimate_overlap(
    X1: np.ndarray, X2: np.ndarray, normalize_by_X1: bool = False
) -> float:  # want big, if negative, ALL the dev set is outside
    V1 = ConvexHull(X1).volume
    overlap = V1 - _estimate_overflow(X1, X2)
    return overlap / V1 if normalize_by_X1 else overlap


def select_via_test_overlap(
    X_train: np.ndarray, X_test: np.ndarray, margin: Optional[float] = None
) -> Sequence[int]:
    # margin is a percent of the distance from the center of the test set to the furthest pt
    # That doesn't work when the test and train basically overlap
    # test_hull_pts, _ = get_exterior_pnts(X_test)
    test_cntr = np.mean(X_test, axis=0).reshape(1, -1)  # format for cdist
    test_distances = cdist(test_cntr, X_test).squeeze()
    train_distances = cdist(test_cntr, X_train).squeeze()
    R = (
        test_distances.mean() + test_distances.std() * 3
        if margin is None
        else test_distances.max() * (1 + margin)
    )
    return np.arange(len(X_train))[train_distances <= R]


######### Fine-Tuning Set Rankings #############
