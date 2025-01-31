#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions and routines for data selection.

Created on Tue Jan 21 17:08:39 2025

@author: rakshat
"""

# typing stuff
from typing import Optional, Sequence, Union, Tuple, Callable
from .custom_types import Data_Features, Label_Set, DataFrame

# imports
import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist, pdist
from copy import deepcopy
from prettytable import PrettyTable
from tqdm import tqdm

# BTV imports
from ._src import (
    prediction_info,
    fit_dknn_toXy,
    estimate_rateVSchance,
    estimate_info,
    class_balance_ratio,
)


# %%
########## Information theoretic selection #########
def make_low_info_mask(
    X: Data_Features,
    y: Label_Set,
    metric: Union[str, Callable] = "euclidean",
    thresh: Union[int, float] = 0,
) -> Sequence[bool]:
    """
    Pick out the data points whose labels provide information below the threshhold, given the other labelled points in the set.

    Parameters
    ----------
    X : M x N Data_Features
        Data points, with M samples and N features.
    y : Label_Set
        Data labels.
    metric : Union[str,Callable], optional
        scipy distance metric to use. The default is "euclidean".
    thresh : Union[int, float], optional
        Maximum information in bits a label can provide to be considered 'low-information'
        and return a 'True' in the output mask.
        The default is 0.

    Returns
    -------
    Sequence[bool]
        Low-info mask. "True" means the sample at that index has info<=thresh.

    """
    clf_knn_selfdrop = fit_dknn_toXy(
        X, y, k=X.shape[1] * 2 + 1, metric=metric, self_exclude=True
    )
    info = prediction_info(
        y,
        clf_knn_selfdrop.predict_proba(X),
        discount_chance=False,
    )
    return info <= thresh


def get_exterior_pnts(X: Data_Features) -> Tuple[np.ndarray, Sequence[int]]:
    """
    Get the exterior points (i.e. those in the convex hull) of a data set.

    Parameters
    ----------
    X : Data_Features
        Data points to find the exterior points of.

    Returns
    -------
    exterior_pts : np.ndarray
        Exterior points in X.
    exterior_idxs : Sequence[int]
       Indices of the exterior points in X.

    """
    if isinstance(X, DataFrame):
        X = X.values
    chull = ConvexHull(X)
    exterior_pts, exterior_idxs = X[chull.vertices], chull.vertices
    return exterior_pts, exterior_idxs


def prune_by_info(
    X: Data_Features,
    y: Label_Set,
    clf=None,
    metric: str = "euclidean",
    thresh: Union[int, float] = 0,
    keep_hull: bool = False,
    shuffle: bool = True,
) -> Sequence[int]:
    """
    Prune dataset (X,y) to only high-information points, and the convex hull (optional).

    Parameters
    ----------
    X : M x N Data_Features
        M data samples with N features to prune.
    y : Label_Set
        M labels of the data to prune.
    clf : sklearn-style classifyer, optional
        Classifier to use to estimate the informtion. Needs only .predict_proba(X) method.
        The default is None, in which case a self-drop d-knn with k = 2*N + 1 will be used.
    metric : str, optional
        scipy distance metric to use. The default is "euclidean".
    thresh : Union[int, float], optional
        Information threshold in bits. Data samples need at least this much information to be included.
        The default is 0.
    keep_hull : bool, optional
        Whether or not the returnd set should include the exterior points of X.
        The default is False.
    shuffle : bool, optional
        Whether to shuffle the returned indices. The default is True.
    Returns
    -------
    Sequence[int]
        Indexies in (x,y) to include.

    """
    if clf is None:
        info_mask = make_low_info_mask(X, y, metric=metric, thresh=thresh)
    else:
        info = prediction_info(
            y,
            clf.predict_proba(X),
            discount_chance=False,
        )
        info_mask = info <= thresh
    out_idxs = np.where(~info_mask)[0].astype(int)
    if keep_hull:
        _, hull_idxs = get_exterior_pnts(X)
        out_idxs = np.intersect1d(hull_idxs, out_idxs)
    if shuffle:
        np.random.shuffle(out_idxs)
    return out_idxs.astype(int)


########### Geometric functions + selection ##########


def _estimate_overflow(X1: np.ndarray, X2: np.ndarray) -> float:
    return ConvexHull(np.vstack([X1, X2])).volume - ConvexHull(X2).volume


def estimate_overflow(
    X1: Data_Features, X2: Data_Features, normalize_by_X1: bool = False
) -> float:
    """
    Estimate the volume of X1 that falls outside of X2, i.e. the "overflow" of X1
    from X2. Uses the convex hull so likely to be an overestimate. When X2 is
    a previous training set and X1 is a fine-tuning set under consideration,
    this should be large, indicating X1 is novel data.

    Parameters
    ----------
    X1 : Data_Features
        Data points to find the overflow of.
    X2 : Data_Features
        Data points to find the overflow from.
    normalize_by_X1 : bool, optional
        Whether to divide by the volume of X1 and return the fraction that lies
        outside of X2. The default is False, returning the estimated overflow in
        units of volume.

    Returns
    -------
    float
        Estimate of the overflow.

    """
    if isinstance(X1, DataFrame):
        X1 = X1.values
    if isinstance(X2, DataFrame):
        X2 = X2.values
    overflow = _estimate_overflow(X1, X2)
    return overflow / ConvexHull(X1).volume if normalize_by_X1 else overflow


def estimate_overlap(
    X1: Data_Features, X2: Data_Features, normalize_by_X1: bool = False
) -> float:
    """
    Estimate the volume of X1 that lies inside of X2. Uses convex hull so an
    answer < 0 suggests that all (or most) of X1 is outside of X2.

    Parameters
    ----------
    X1 : Data_Features
        Data points to esimate the overlap of.
    X2 : Data_Features
        Data points to estimate the overlap with.
    normalize_by_X1 : bool, optional
        Whether to divide by the volume of X1 and return the fraction that lies
        inside of X2. The default is False, returning the estimated overlap in
        units of volume.

    Returns
    -------
    float
        Estimate of the overlap.

    """
    # want big, if negative, ALL the dev set is outside
    if isinstance(X1, DataFrame):
        X1 = X1.values
    if isinstance(X2, DataFrame):
        X2 = X2.values
    V1 = ConvexHull(X1).volume
    overlap = V1 - _estimate_overflow(X1, X2)
    return overlap / V1 if normalize_by_X1 else overlap


def select_pts_nearest2test(
    X_train: Data_Features,
    X_test: Data_Features,
    metric: Union[str, Callable] = "euclidean",
) -> np.ndarray[int]:
    """
    Select the datapoints in X_train that lie closest to X_test.

    Parameters
    ----------
    X_train : Data_Features
        Data points to select from.
    X_test : Data_Features
        Data points to select the neighbors of.
    metric : Union[str,Callable], optional
        scipy distance metric to use. The default is "euclidean".

    Returns
    -------
    neartest_idxs : Sequence[int]
        Indicies in _train that are neighbors (in X_train) to points in X_test.

    """
    nn = NearestNeighbors(n_neighbors=1, metric=metric).fit(X_train)
    _, train0_idxs = nn.kneighbors(X_test)
    neartest_idxs = train0_idxs.squeeze().astype(int)
    return neartest_idxs


def select_clusters_nearest2test(
    X_train: Data_Features,
    X_test: Data_Features,
    metric: str = "euclidean",
    min_cluster_size: Optional[int] = None,
    drop_outliers: bool = True,
) -> Tuple[list[int], list[int]]:
    """
    Find clusters in X_train which contain at least on point in X_test when
    X_test is combined with X_train. Uses HDBSCAN fro clustering.

    Parameters
    ----------
    X_train : Data_Features
        Data points to select from. M samples by N features.
    X_test : Data_Features
        Clusters in the combined set must include points from this set.
        M samples by N features.
    metric : str, optional
        scipy distance metric to use. The default is "euclidean".
    min_cluster_size : Optional[int], optional
        Minimum number of points that must be in a cluster.
        The default is None, corresponding to 2*N + 1.
    drop_outliers : bool, optional
        Whether or not to drop the poins the HDBSCAN does not include in any cluster.
        The default is True, which will exclude outliers from the returned set.

    Returns
    -------
    included_pt_idxs : list[int]
        Indicies of points in X_train which would be clustered with points in X_test.
    good_cluster_idxs : list[int]
        Cluster indices of the points selected.

    """
    if min_cluster_size is None:
        min_cluster_size = X_train.shape[1] * 2 + 1  # twice the number of features + 1
    if isinstance(X_train, DataFrame):
        X_train = X_train.values
    if isinstance(X_test, DataFrame):
        X_test = X_test.values

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
            return list(range(len(X_train))), list(cluster_idx[: len(X_train)])
    if drop_outliers:
        good_clusters = np.delete(good_clusters, np.where(good_clusters == -1))
    included_pt_idxs = [
        i for i in range(len(X_train)) if cluster_idx[i] in good_clusters
    ]
    good_cluster_idxs = [
        cluster_idx[i] for i in range(len(X_train)) if cluster_idx[i] in good_clusters
    ]
    return included_pt_idxs, good_cluster_idxs


def select_via_test_overlap(
    X_train: Data_Features,
    X_test: Data_Features,
    margin: Optional[float] = 0,
    verbose=True,
) -> np.ndarray:
    """
    Attempts to select a set from X_train which overlaps with X_test.
    Useful when the convex hull of X_test only covers a localized region of X_train.

    Parameters
    ----------
    X_train : Data_Features
        Data points to select from.
    X_test : Data_Features
        Data points define the region of interest.
    margin : Optional[float], optional
        If <= 1, defines a percent of the distance from the center of the test
        set to the furthest pt to add to the radius of inclusion centered on
        the center of thest set.
        If > 1, defines a number of standard deviations of the distance of points
        in X_test from the center of X_test to go out from the mean radius of X_test.
        The default is 0, coresponding to aprecise overlap with the convex hull of X_test.
    verbose : TYPE, optional
        Whether or not to print some basic information about the geometry of the data. The default is True.

    Returns
    -------
    pruned_idxs, np.ndarray, dtype=int
        Indicies in X_train that overlap with X_test.

    """
    # margin is a percent of the distance from the center of the test set to the furthest pt
    # That doesn't work when the test and train basically overlap
    if isinstance(X_train, DataFrame):
        X_train = X_train.values
    if isinstance(X_test, DataFrame):
        X_test = X_test.values
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
    pruned_idxs = np.arange(len(X_train))[train_distances <= R]
    return pruned_idxs


######### Tests and Fine-Tuning Set Rankings #############


def _cut_by_class_balance(
    y_ft_list: Sequence[Label_Set],
    test_ratio: float,
    tol: float = 1e-3,
    verbose=False,
    to_keep: Optional[Sequence[bool]] = None,
) -> Tuple[Sequence[bool], np.ndarray[float]]:
    """
    Estimates the class balance for a sequence of fine tuning sets and sets
    a value of False in to_keep if the ratio differs from test_ratio bt tol.
    Skips calculation if there is a False at the index in to_keep.

    Returns
    -------
    to_keep : Sequence[bool]
        Fine-tune sets to keep.
    class_bal_ratios : np.ndarray, dtype=float

    """
    if to_keep is None:
        to_keep = [True] * len(y_ft_list)
    class_bal_ratios = np.zeros(len(y_ft_list))
    for i, y in tqdm(
        enumerate(y_ft_list),
        desc=f"Test class balance ratio is {test_ratio:0.5f}\nEstimating class balance ratio for fine tuning sets..",
        total=len(y_ft_list),
    ):
        if to_keep[i]:
            class_bal_ratio = class_balance_ratio(y)
            if not np.isclose(class_bal_ratio, test_ratio, tol):
                to_keep[i] = False
                if verbose:
                    print(
                        f"INFO: Dropping fine tuning set at index {i}. It has estimated class balance ratio of {class_bal_ratio:0.5f}"
                    )
            else:
                info_amounts[i] = info
    return to_keep, class_bal_ratios


def _cut_by_info_rate(
    X_ft_list: Sequence[Data_Features],
    y_ft_list: Sequence[Label_Set],
    info_thresh: float,
    verbose=False,
    to_keep: Optional[Sequence[bool]] = None,
) -> Tuple[Sequence[bool], np.ndarray[float]]:
    """
    Estimates the information rate of a sequence of fine tuning sets and sets
    a value of False in to_keep if the rate is below info_thresh. Skips calculation
    if there is a False at the index in to_keep.

    Returns
    -------
    to_keep : Sequence[bool]
        Fine-tune sets to keep.
    info_rates : np.ndarray, dtype=float

    """
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
    return to_keep, info_rates


def _cut_by_test_overlap(
    X_ft_list: Sequence[Data_Features],
    X_test: Data_Features,
    overlap_thresh: Union[float, int],
    verbose=False,
    to_keep: Optional[Sequence[bool]] = None,
) -> Tuple[Sequence[bool], np.ndarray[float]]:
    """
    Estimates the overlap with X_test for a sequence of fine tuning sets and sets
    a value of False in to_keep if the overlap is below overlap_thresh. Skips calculation
    if there is a False at the index in to_keep.

    Returns
    -------
    to_keep : Sequence[bool]
        Fine-tune sets to keep.
    overlaps : np.ndarray, dtype=float

    """
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
    return to_keep, overlaps


def _overflow_from_train(
    X_train: Data_Features,
    X_ft_list: Sequence[Data_Features],
    to_keep: Optional[Sequence[bool]] = None,
) -> np.ndarray:
    """
    Estimates the overflow from X_train for a sequence of fine tuning sets.
    Skips calculation if there is a False at the index in to_keep.

    Returns
    -------
    train_overflows : np.ndarray, dtype=float

    """

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
) -> np.ndarray:
    """
    Rank by a series of score sequences.

    Parameters
    ----------
    *score_lists : Sequence[Sequence[Union[float, int]]]
        Sequence of score sequences. Each element of score_lists represents a
        different metric and is a suquence of scores (one for each fintuning set)
        on that metric. Expects most important metric to be passed first.
    to_keep : Optional[Sequence[bool]], optional
        Boolean sequence whose length is the number of scores. A False value will
        ignore scores at that indexand result in a rank of -1.
        The default is None, ignoring no scores.
    big_first : Optional[Sequence[bool]], optional
        Boolean sequence whose length is the number of scores. A True value indicates
        that a high score mertics a high rank. A False indicates that a low score is better.
        The default is None, ranking all scores highest to lowest.
    sort_order : Optional[Sequence[int]], optional
        An index sequence to resort *score_lists and everything else.
        The default is None, indicating no resorting.
    verbose : bool, optional
        If to print ranking tables. The default is True.
    rank_tol : Optional[Sequence[Optional[float]]], optional
        Sequence of tie tolerences for the different scores. When scores are
        within this tolerence are consided a group of ties and are not reordered
        within the group. Groups of ties are ranked by thier average.
        Can be used iteratevely to have scores in one metric influence ties in
        subsiquent metrics and is why *score_lists assumes the scores are passed
        in order of importance.
        PAss None at an index to forego fuzzy ranking at that index.
        The default is None, indicating only using np.argsort and no fuzzy ranking.

    Returns
    -------
    sort_idx: np.ndarray, dtype=int
        index which sorts the sets.

    """
    if verbose:
        var_names = []
        for i, v in enumerate(score_lists):
            try:
                var_names.append(
                    [name for name, value in globals().items() if value is v][0]
                )
            except IndexError:
                var_names.append(f"Metric score {i+1}")
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
    # most imortant metric should be sorted on last. This assumes attributes in scores_list are in order of importance
    if sort_order is None:
        sort_order = np.arange(len(score_lists))[::-1]
    if rank_tol is None:
        rank_tol = [None] * len(score_lists)
    sort_idx = np.arange(len(score_lists[0]))
    for metric_idx in sort_order:
        # np argsort leaves ties in previous order so the last rankng is the mmost important with previous rankings showing through in ties
        if rank_tol[metric_idx] is None:
            sort_idx = np.argsort(score_lists[metric_idx])
        else:
            sort_idx = fuzzy_argrank(
                score_lists[metric_idx], rank_tol[metric_idx], big_first=False
            )
        if verbose:
            tab = PrettyTable()
            tab.title = f"{var_names[metric_idx]}"
            tab.add_column("Input index", np.arange(len(score_lists[0])))
            tab.add_column("Rank", sort_idx.argsort() + 1)
            tab.add_column(
                "Value", score_lists[metric_idx] * -1 * big_first[metric_idx]
            )
            print(tab)
    if verbose:
        print("Values of 0.0 indicate a dataset excluded during the threshold checks.")
        print("Dropped sets are assigned a 'rank' of -1.")
    sort_idx = np.where(to_keep, sort_idx, -1)
    return sort_idx.astype(int)


def select_ft_sets(
    X_train: Data_Features,
    X_ft_list: Sequence[Data_Features],
    y_ft_list: Sequence[Label_Set],
    X_test: Sequence[Data_Features],
    y_test: Sequence[Label_Set],
    verbose: bool = True,
    return_sorted: bool = True,
    info_rate_tol: float = 0.03,
    overlap_thresh: float = 0.05,
) -> Union[
    Tuple[Sequence[Data_Features], Sequence[Label_Set], Sequence[int]], Sequence[int]
]:
    """
    Packaged routine to select fine tuning sets, given a test set, and assuming
    training was alreadyperformed on (X_tran,y_train).

    Parameters
    ----------
    X_train : Data_Features
        Train set features.
    X_ft_list : Sequence[Data_Features]
        Sequence of features from the fine tuning sets to select between.
    y_ft_list : Sequence[Label_Set]
        Sequence of labels from the fine tuning sets to select between.
    X_test : Sequence[Data_Features]
        Test set features.
    y_test : Sequence[Label_Set]
        TEst set labels.
    verbose : bool, optional
        Whether to print notices and warnings. The default is True.
    return_sorted : bool, optional
        Whether to not to return X_ft_list and y_ft_list sorted.
        The default is True. Set to False to return only the sequence
        that will sort X_ft_list and y_ft_list.
    info_rate_tol : float, optional
        Portion of the information rate of the test set which the fine tuning
        sets are permitted to fall below before beong excluded.
        The default is 0.03.
    overlap_thresh : float, optional
        Minimum portion of the fine tuning set that must overlap with the test set
        to be considered. The default is 0.05.

    Returns
    -------
    if return sorted:
        X_ft_list[ranking] : Sequence[Data_Features]
            Sorted fintuning features

        y_ft_list[ranking] : Sequence[Label_Set]
            Sorted fine tuning labels

        ranking : Sequence[int]
             Sequence which will sort the fine tuning sets provided.

    else:
       ranking : Sequence[int]
            Sequence which will sort the fine tuning sets provided.

    """
    global test_overlaps, info_rates, train_overflows  # makes the PrettyTables titles work
    test_info = estimate_info(X_test, y_test)
    test_info_rate = estimate_rateVSchance(X_test, y_test)
    info_thresh = test_info_rate - test_info_rate * info_rate_tol
    test_class_ratio = class_balance_ratio(y_test)
    if verbose:
        print(f"Information of test set estimated at {estimate_info:.5f}.")
        print(f"Information rate of test set estimated at {test_info_rate:.5f}.")
        print(
            f"Fine-tune sets must have an information rate of {info_thresh:.5f} for inclusion."
        )
    to_keep, info_amounts = _cut_by_class_balance(
        y_ft_list, test_class_ratio, verbose=verbose
    )
    to_keep, info_rates = _cut_by_info_rate(
        X_ft_list, y_ft_list, info_thresh, verbose=verbose, to_keep=to_keep
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
) -> np.ndarray:
    """
    Ranks values in arr but leaves groups of nearby values in the order they were passed.
    Ranks groups of nearby values by average. Use in rounds to rank on multiple attributes,
    leaving near-ties to subsiquent rounds.
    Relies in HDBSCAN to find "nearby" values.

    Parameters
    ----------
    arr : Sequence[Union[float, int]]
        Values to sort/rank.
    tol : float, optional
        How close . The default is 0.05.
    big_first : bool, optional
        Whether to sort high to low. The default is False, sorting low to high.

    Returns
    -------
    np.ndarray, dtype=int
        Array which will sort arr.

    """
    assert tol > 0, "Just use np.argsort, then..."
    arr = np.array(arr)
    clusterer = DBSCAN(eps=tol, min_samples=2)
    cluster_idxs = clusterer.fit_predict(arr.reshape(-1, 1))
    arr2 = deepcopy(arr)
    for i in range(cluster_idxs.max() + 1):
        arr2 = np.where(cluster_idxs == i, arr[cluster_idxs == i].mean(), arr2)
    return np.argsort(arr2)[::-1] if big_first else np.argsort(arr2, kind="stable")
