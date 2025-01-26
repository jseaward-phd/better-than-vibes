#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:54:53 2024

@author: rakshat
"""

import numpy as np
import pandas as pd
import openml
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from typing import Union, Optional, Sequence

# import umap # save this for later


def dataset2df(
    ds: Union[int, str, openml.datasets.OpenMLDataset],
    class_cols: Optional[Union[str, Sequence[str]]] = None,
    scale: bool = True,
    X_df: bool = False,
    verbose: bool = True,
):
    if not isinstance(ds, openml.datasets.OpenMLDataset):
        ds = openml.datasets.get_dataset(dataset_id=ds)
    if verbose:
        print(ds)
    df, *_ = ds.get_data()

    if scale:
        scaler = StandardScaler()
        df.loc[:, df.dtypes == "float"] = scaler.fit_transform(
            df.loc[:, df.dtypes == "float"]
        )
    if class_cols is None:
        return df
    else:
        if isinstance(class_cols, str):
            class_cols = class_cols.split(",")
        X = df.loc[:, [x not in class_cols for x in df.columns]]
        if scale:
            X = X.loc[:, X.dtypes == "float"]
        if not X_df:
            X = X.values
        y = df.loc[:, class_cols].squeeze()
        le = LabelEncoder()
        y = le.fit_transform(y)
        return df, X, y


def get_core_train_sample(
    y, fold_idx_list
):  # for sets with classes with very low number of samples. Gets a core training set with at least one of every class
    representatives = np.array([])
    fold_idx_list_out = fold_idx_list.copy()
    for y_val in np.unique(y, axis=0):
        representatives = np.append(
            representatives, np.where(np.all(y == y_val, axis=1))[0][0]
        )
    fold_idx_list_out[0] = np.unique(
        np.append(fold_idx_list[0], representatives)
    ).astype(int)

    for i, idxs in enumerate(fold_idx_list_out):
        if i == 0:
            continue
        fold_idx_list_out[i] = np.array(
            [x for x in fold_idx_list_out[i] if x not in fold_idx_list_out[0]]
        )

    return fold_idx_list_out, representatives.astype(int)


# def embedd_with_umap(df, embedding_dim=10, metric="euclidean", scale=True):
#     reducer = umap.UMAP(n_components=embedding_dim, metric=metric)
#     if scale:
#         scaler = StandardScaler()
#         data = scaler.fit_transform(df.loc[:, df.dtypes == "float"])
#     else:
#         data = df.loc[:, df.dtypes == "float"].values

#     reducer.fit(data)
#     return reducer
