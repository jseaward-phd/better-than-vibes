#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:08:39 2025

@author: rakshat
"""

import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist, pdist
from copy import deepcopy
from prettytable import PrettyTable
from tqdm import tqdm


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
    X_train: np.ndarray,
    X_test: np.ndarray,
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
    X_ft_list: Sequence[np.ndarray],
    y_ft_list: Sequence[Sequence[int]],
    info_thresh: float,
    verbose=False,
    to_keep: Optional[Sequence[bool]] = None,
) -> Tuple[Sequence[bool], np.ndarray[float]]:
    if to_keep is None:
        to_keep = [True] * len(X_ft_list)
    info_rates = np.zeros(len(X_ft_list))
    for i, (X, y) in tqdm(
        enumerate(zip(X_ft_list, y_ft_list)),
        desc="Estimating self-information rate...",
        total=len(X_ft_list),
    ):
        if to_keep[i]:
            info_rate = estimate_rateVSchance(X, y)
            if info_rate < info_thresh:
                to_keep[i] = False
                if verbose:
                    print(
                        f"INFO: Dropping fine tuning set at index {i}. It has estimated info rate of {info_rate:0.4f}"
                    )
            else:
                info_rates[i] = info_rate
    return to_keep, np.array(info_rates)


def _cut_by_test_overlap(
    X_ft_list: Sequence[np.ndarray],
    X_test: np.ndarray,
    overlap_thresh: Union[float, int],
    verbose=False,
    to_keep: Optional[Sequence[bool]] = None,
) -> Tuple[Sequence[bool], np.ndarray[float]]:
    if to_keep is None:
        to_keep = [True] * len(X_ft_list)
    overlaps = np.zeros(len(X_ft_list))
    for i, X in tqdm(
        enumerate(X_ft_list),
        desc="Checking overlap with test set...",
        total=len(X_ft_list),
    ):
        if to_keep[i]:
            overlap_percent = estimate_overlap(X_test, X, normalize_by_X1=True)
            if overlap_percent < overlap_thresh:
                to_keep[i] = False
                if verbose:
                    print(
                        f"INFO: Dropping fine tuning set at index {i}. Estimate that it overlaps with {overlap_percent :0.3f}% of the test set."
                    )
            else:
                overlaps[i] = overlap_percent
    return to_keep, np.array(overlaps)


def _overflow_from_train(
    X_train: np.ndarray,
    X_ft_list: Sequence[np.ndarray],
    to_keep: Optional[Sequence[bool]] = None,
) -> np.ndarray[float]:
    if to_keep is None:
        to_keep = [True] * len(X_ft_list)
    train_overflows = np.zeros(len(X_ft_list))
    for i, X in tqdm(
        enumerate(X_ft_list), desc="Estimating portion outside the training set..."
    ):
        if to_keep[i]:
            train_overflows[i] = estimate_overflow(X, X_train, normalize_by_X1=True)
    return train_overflows


# %%
def rank_by_scores(
    *score_lists: Sequence[Sequence[Union[float, int]]],
    to_keep: Optional[Sequence[bool]] = None,
    big_first: Optional[Sequence[bool]] = None,
    sort_order: Optional[Sequence[int]] = None,
    verbose: bool = True,
    rank_tol: Optional[Sequence[Optional[float]]] = None,
) -> np.ndarray[int]:
    if verbose:
        var_names = []
        for i, v in enumerate(score_lists):
            try:
                var_names.append(
                    [name for name, value in globals().items() if value is v][0]
                )
            # make passed scores global tfor titled tables
            except IndexError:
                var_names.append(f"Attribute score {i+1}")
    assert all(
        len(x) == len(score_lists[0]) for x in score_lists
    ), "Should have a score for each finetuning set."
    if to_keep is None:
        to_keep = [True] * len(score_lists[0])
    if big_first is None:
        big_first = np.array([-1] * len(score_lists[0]))
    else:
        big_first = np.where(big_first, -1, 1)
    assert all(len(big_first) == len(x) for x in score_lists)
    # np.argsort sorts smallest to bigest. To sort biggest to smallest, multiply the scores by  -1
    score_lists = [b * x for b, x in zip(big_first, score_lists)]
    # most imortant attribute should be sorted on last. This assumes attributes in scores_list are in order of importance
    if sort_order is None:
        sort_order = np.arange(len(score_lists))[::-1]
    if rank_tol is None:
        rank_tol = [None] * len(score_lists)
    sort_idx = np.arange(len(score_lists[0]))
    for attribute_idx in sort_order:
        # np argsort leaves ties in previous order so the last rankng is the mmost important with previous rankings showing through in ties
        if rank_tol[attribute_idx] is None:
            sort_idx = np.argsort(score_lists[attribute_idx])
        else:
            sort_idx = fuzzy_argrank(
                score_lists[attribute_idx], rank_tol[attribute_idx], big_first=False
            )
        if verbose:
            tab = PrettyTable()
            tab.title = f"{var_names[attribute_idx]}"
            tab.add_column("Input index", np.arange(len(score_lists[0])))
            tab.add_column("Rank", sort_idx.argsort() + 1)
            tab.add_column(
                "Value", score_lists[attribute_idx] * big_first[attribute_idx]
            )
            print(tab)
    if verbose:
        print("Values of 0.0 indicate a dataset excluded during the threshold checks.")
        print("Dropped sets are assigned a 'rank' of -1.")
    sort_idx = np.where(to_keep, sort_idx, -1)
    return sort_idx.astype(int)


def select_ft_sets(
    X_train: Sequence[np.ndarray],
    X_ft_list: Sequence[np.ndarray],
    y_ft_list: Sequence[np.ndarray[int]],
    X_test: Sequence[np.ndarray],
    y_test: Sequence[np.ndarray[int]],
    verbose=True,
    return_sorted=True,
    info_rate_tol: float = 0.03,
    overlap_thresh=0.05,
):
    global test_overlaps, info_rates, train_overflows
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
    to_keep, test_overlaps = _cut_by_test_overlap(
        X_ft_list, X_test, overlap_thresh, verbose=verbose, to_keep=to_keep
    )
    train_overflows = _overflow_from_train(X_train, X_ft_list, to_keep=to_keep)
    ranking = rank_by_scores(
        test_overlaps, info_rates, train_overflows, to_keep=to_keep
    )
    if return_sorted:
        return X_ft_list[ranking], y_ft_list[ranking], ranking
    return ranking


##### Utilies #####


def fuzzy_argrank(
    arr: Sequence[Union[float, int]],
    tol: float = 0.05,
    big_first: bool = False,
):
    # leaves groups of nearby values unordered
    assert tol > 0, "Just use np.argsort, then..."
    arr = np.array(arr)
    clusterer = DBSCAN(eps=tol, min_samples=2)
    cluster_idxs = clusterer.fit_predict(arr.reshape(-1, 1))
    arr2 = deepcopy(arr)
    for i in range(cluster_idxs.max() + 1):
        arr2 = np.where(cluster_idxs == i, arr[cluster_idxs == i].mean(), arr2)
    return np.argsort(arr2)[::-1] if big_first else np.argsort(arr2, kind="stable")
