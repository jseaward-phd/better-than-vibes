#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tabular data utilities.

Created on Fri Nov 15 13:54:53 2024

@author: rakshat
"""

import numpy as np
import openml
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Union, Optional, Sequence
from .custom_types import Label_Set

# import umap # save this for later

def dataset2df(
    ds: Union[int, str, openml.datasets.OpenMLDataset],
    class_cols: Optional[Union[str, Sequence[str]]] = None,
    scale: bool = True,
    X_df: bool = False,
    verbose: bool = True,
):
    """
    Make a pandas DataFrame and/or numpy array from an openml dataset.

    Parameters
    ----------
    ds : Union[int, str, openml.datasets.OpenMLDataset]
        OpenlML dataset or OpenML id designation.
    class_cols : Optional[Union[str, Sequence[str]]], optional
        Column names in the dataset which indicate classes.
        The default is None, in which case so y set will be split off and the
        whole dataset will be returned as features.
        Pass a string or a list of strings to return those columns as encoded labels.
    scale : bool, optional
        Whether or not to apply standard scaling to features. The default is True.
    X_df : bool, optional
        Whether or not to teturn the features in a DataFrame.
        The default is False, returning features as a numpy array.
    verbose : bool, optional
        Whether or not to print the information attached to the Open ML dataset.
        The default is True.

    Returns
    -------
    if class_cols is None:
        The dataset as features.
    else:
        df : DataFrame
            The entire dataset as a DataFrame
        X : numpay array or DataFrame, depending on X_df
            Data features
        y : numpy array, dtype=int
            Data labels

    """
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
        
def get_core_train_sample(y: Label_Set, fold_idx_list: list[Sequence[int]]):
    """
    For collecting a minimum training set from a k-fold index list to be used
    with additive training set routines.
    Gets an initial training set with at least one of every class to expand upon.

    Parameters
    ----------
    y : Label_Set
        Labels corresponding to the set fed to skklearn KFold routine.
    fold_idx_list : list[Sequence[int]]
        List of index equences which are the folds on the dataset.

    Returns
    -------
    fold_idx_list_out : list[Sequence[int]]
        Fold index list where the fisr fold is guaranteed to include at least
        one of each label.
    representatives : Sequence[int]
        Indices in y that form a minimal set.

    """
    # for sets with classes with very low number of samples.
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


# Can do this later:
# def embedd_with_umap(df, embedding_dim=10, metric="euclidean", scale=True):
#     reducer = umap.UMAP(n_components=embedding_dim, metric=metric)
#     if scale:
#         scaler = StandardScaler()
#         data = scaler.fit_transform(df.loc[:, df.dtypes == "float"])
#     else:
#         data = df.loc[:, df.dtypes == "float"].values

#     reducer.fit(data)
#     return reducer
