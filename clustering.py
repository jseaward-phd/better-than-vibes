#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:36:51 2025

@author: rakshat
"""

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from scipy.spatial import ConvexHull

from typing import Optional, Sequence

from btv import prediction_info, dist_weight_ignore_self

# %% Prolly want to re-factor this to work in ideces so you can pair all the ys up at the end


def make_low_info_mask(X, y, metric="euclidean", thresh=0):
    clf_knn_selfdrop = KNeighborsClassifier(
        n_neighbors=X.shape[1] * 2, metric=metric, weights=dist_weight_ignore_self
    )
    clf_knn_selfdrop.fit(X, y)
    info = prediction_info(
        y,
        clf_knn_selfdrop.predict_proba(X),
        discount_chance=True,  # option?
    )
    return info <= thresh


def sort_by_label_and_info(X, y, info_mask: Optional[Sequence[bool]] = None):
    dict_label_keys = {l: [] for l in sorted(np.unique(y))}
    if info_mask is None:
        for x, l in zip(X, y):
            dict_label_keys[l].append(x)
    else:
        for x, l in zip(X[info_mask], y[info_mask]):
            dict_label_keys[l].append(x)

    return [np.vstack(dict_label_keys[l]) for l in sorted(np.unique(y))]


def cluster_likes(X_list, metric="euclidean", min_cluster_size: Optional[int] = None):
    if min_cluster_size is None:
        min_cluster_size = X_list[0].shape[1] * 2  # twice the number of features

    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric=metric)
    cluster_idx_list = [clusterer.fit_predict(x) for x in X_list]
    # TODO: consider whether to keep or drop zero-info outliers (cluster index == -1)
    cluster_pt_list = [
        [x[c == i] for i in np.unique(c) if i >= 0]
        for x, c in zip(X_list, cluster_idx_list)
    ]
    # cluster_pt_list is a two-tier list of arrays first by label then by cluster idx
    return cluster_pt_list, cluster_idx_list


def get_center_datapoit(X):
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(X)
    centroid = np.mean(X, axis=0)
    _, nn_index = nn.kneighbors(centroid.reshape(1, -1))
    nearest = X[int(nn_index)]
    return nearest, centroid


def get_exterior_pnts(X):
    chull = ConvexHull(X)
    return X[chull.vertices], chull.vertices


# %% setup test

import data_tab
from scipy.spatial import QhullError
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

ds = data_tab.getdata(44156, verbose=False)
df, X, y = data_tab.dataset2df(ds, class_cols=["class"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = GradientBoostingClassifier(
    n_estimators=100, learning_rate=1.0, max_depth=1, random_state=10
)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_dev1, y_train, y_dev1 = train_test_split(X_train, y_train, test_size=0.2)
X_train, X_dev2, y_train, y_dev2 = train_test_split(X_train, y_train, test_size=0.2)
X_train, X_dev3, y_train, y_dev3 = train_test_split(X_train, y_train, test_size=0.2)
X_train, X_dev4, y_train, y_dev4 = train_test_split(X_train, y_train, test_size=0.2)
X_train, X_dev5, y_train, y_dev5 = train_test_split(X_train, y_train, test_size=0.2)

X_ft_list = [X_dev1, X_dev2, X_dev3, X_dev4, X_dev5]
y_ft_list = [y_dev1, y_dev2, y_dev3, y_dev4, y_dev5]

# %% Select down further

strategy = "hull"

info_mask = make_low_info_mask(X_train, y_train)
like_xs = sort_by_label_and_info(X_train, y_train, info_mask)
cluster_list, _ = cluster_likes(like_xs)

keep_pt_list = []
if strategy == "hull":
    for label, pt_list in enumerate(cluster_list):
        for pts in pt_list:
            try:
                hull_pts, _ = get_exterior_pnts(pts)
            except QhullError:
                continue
            keep_pt_list.append(hull_pts)

else:
    for label, pt_list in enumerate(cluster_list):
        for pts in pt_list:
            nn, _ = get_center_datapoit(pts)
            keep_pt_list.append(nn)
keep_pt_list.append(X_train[~info_mask])
keep_pts = np.vstack(keep_pt_list)
idx = [
    int(np.where(np.all(X_train == x, axis=1))[0]) for x in keep_pts
]  # this step is dumb
y_keep = y[idx]
