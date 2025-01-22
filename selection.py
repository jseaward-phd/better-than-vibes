#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:08:39 2025

@author: rakshat
"""

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist, pdist
from prettytable import PrettyTable


from typing import Optional, Sequence, Union, Tuple
from sklearn.base import BaseEstimator

from btv import prediction_info, fit_dknn_toXy, estimate_rateVSchance


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
) -> np.ndarray[int]:
    if clf is None:
        info_mask = make_low_info_mask(X, y, metric=metric)
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
) -> np.ndarray[int]:
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
) -> float:
    # want big, if negative, ALL the dev set is outside
    V1 = ConvexHull(X1).volume
    overlap = V1 - _estimate_overflow(X1, X2)
    return overlap / V1 if normalize_by_X1 else overlap


def select_pts_nearest2test(X_train, X_test, metric="euclidean") -> np.ndarray[int]:
    nn = NearestNeighbors(n_neighbors=1, metric=metric).fit(X_train)
    _, train0_idxs = nn.kneighbors(X_test)
    return train0_idxs.squeeze().astype(int)


def select_clusters_nearest2test(
    X_train,
    X_test,
    metric: str = "euclidean",
    min_cluster_size: Optional[int] = None,
    drop_outliers: bool = True,
) -> list[int]:
    if min_cluster_size is None:
        min_cluster_size = X_train.shape[1] * 2 + 1  # twice the number of features + 1

    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric=metric)
    cluster_idx = clusterer.fit_predict(np.vstack([X_train, X_test]))
    good_clusters = np.unique(
        cluster_idx[len(X_train) :]
    )  # clusters that include test pts
    if len(np.unique(cluster_idx)) == len(np.unique(good_clusters)):
        if drop_outliers:
            print(
                "Warning: Test points located in all clusters. Only unclustered outliers will be dropped."
            )
        else:
            print("Warning: Test points located in all clusters. Retuning all indices.")
            return list(range(len(X_train)))
    if drop_outliers:
        good_clusters = np.delete(good_clusters, np.where(good_clusters == -1))
    return [i for i in range(len(X_train)) if cluster_idx[i] in good_clusters]


def select_via_test_overlap(
    X_train: np.ndarray, X_test: np.ndarray, margin: Optional[float] = 0, verbose=True
) -> np.ndarray[int]:
    # margin is a percent of the distance from the center of the test set to the furthest pt
    # That doesn't work when the test and train basically overlap
    test_cntr = np.mean(X_test, axis=0).reshape(1, -1)  # format for cdist
    test_distances = cdist(test_cntr, X_test).squeeze()
    train_distances = cdist(test_cntr, X_train).squeeze()
    R = (
        test_distances.mean() + test_distances.std() * margin
        if margin > 1
        else test_distances.max() * (1 + margin)
    )
    if verbose:
        train_cntr = np.mean(X_train, axis=0).reshape(1, -1)
        center_sep = pdist([test_cntr.squeeze(), train_cntr.squeeze()]).item()
        print(f"Distnace between test and train centers: {center_sep:0.4f}")
        train_internal_dist = cdist(train_cntr, X_train)
        print(f"Train max radius: {train_internal_dist.max():0.4f}")
        print(f"Train mean radius:{train_internal_dist.mean():0.4f}")
        print(f"Test radius STD: {train_internal_dist.std():0.4f}")
        print(f"Test max radius: {test_distances.max():0.4f}")
        print(f"Test mean radius:{test_distances.mean():0.4f}")
        print(f"Test radius STD: {test_distances.std():0.4f}")
        print(f"cutoff radius: {R:0.4f}")
    return np.arange(len(X_train))[train_distances <= R]


######### Tests and Fine-Tuning Set Rankings #############


def _cut_by_info_rate(
    X_ft_list,
    y_ft_list,
    info_thresh,
    verbose=False,
    to_keep: Optional[list[int]] = None,
) -> Tuple[list[int], np.ndarray]:
    info_rates = [estimate_rateVSchance(X, y) for X, y in zip(X_ft_list, y_ft_list)]
    if to_keep is None:
        to_keep = list(range(len(X_ft_list)))
    for i, X in enumerate(X_ft_list):
        if i not in to_keep:
            continue
        if info_rates[i] < info_thresh:
            to_keep.remove(i)
            if verbose:
                print(
                    f"Dropping fine tuning set at index {i}. It has estimated info rate of {info_rates[i] :0.4f}"
                )
    return to_keep, np.array(info_rates)


def _cut_by_overlap(
    X_ft_list,
    X_test,
    overlap_thresh,
    verbose=False,
    to_keep: Optional[list[int]] = None,
) -> Tuple[list[int], np.ndarray]:
    if to_keep is None:
        to_keep = list(range(len(X_ft_list)))
    overlaps = []
    for i, X in enumerate(X_ft_list):
        if i not in to_keep:
            continue
        overlap_percent = estimate_overlap(X_test, X, normalize_by_X1=True)
        overlaps.append(overlap_percent)
        if overlap_percent < overlap_thresh:
            to_keep.remove(i)
            if verbose:
                print(
                    f"Dropping fine tuning set at index {i}. Estimate that it overlaps with {overlap_percent :0.3f}% of the test set."
                )
    return to_keep, np.array(overlaps)


def _overflow_from_train(
    X_train: Sequence[np.ndarray],
    X_ft_list: Sequence[np.ndarray],
    to_keep: Optional[list[int]] = None,
) -> np.ndarray:
    if to_keep is None:
        to_keep = list(range(len(X_ft_list)))
    train_overflows = np.ones(len(X_ft_list))
    for i, X in enumerate(X_ft_list):
        if i not in to_keep:
            continue
        train_overflows[i] = estimate_overflow(X, X_train, normalize_by_X1=True)
    return train_overflows


def select_ft_sets(
    X_train: Sequence[np.ndarray],
    X_ft_list: Sequence[np.ndarray],
    y_ft_list: Sequence[np.ndarray[int]],
    X_test: Sequence[np.ndarray],
    y_test: Sequence[np.ndarray[int]],
    verbose=True,
    return_sorted=True,
    info_rate_tol: float = 0,
    overlap_thresh=0.05,
):
    test_info_rate = estimate_rateVSchance(X_test, y_test)
    info_thresh = test_info_rate - test_info_rate * info_rate_tol
    if verbose:
        print(f"Information rate of test set estimated at {test_info_rate:.5f}.")
        print(
            f"Fine-tune sets must have an information rate of {info_thresh:.5f} for inclusion."
        )
    to_keep, info_rates = _cut_by_info_rate(
        X_ft_list, y_ft_list, info_thresh, verbose=verbose
    )
    to_keep, test_overlaps = _cut_by_overlap(
        X_ft_list, X_test, overlap_thresh, verbose=verbose, to_keep=to_keep
    )
    train_overflows = _overflow_from_train(X_train, X_ft_list, to_keep=to_keep)
