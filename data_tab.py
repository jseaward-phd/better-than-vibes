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
import umap

def dataset2df(ds, class_cols = None, scale = True, X_df=False):        
    df, *_ = ds.get_data()
    
    if scale:
        scaler = StandardScaler()
        df.loc[:,df.dtypes == 'float'] = scaler.fit_transform(df.loc[:,df.dtypes == 'float'])
    if class_cols is None:
        return df
    else:
        if isinstance(class_cols,str):
            class_cols = class_cols.split(',')
        X = df.loc[:, [x not in class_cols for x in df.columns]]
        if scale:
            X = X.loc[:,X.dtypes == 'float']
        if not X_df:
            X = X.values
        y = df.loc[:, class_cols].squeeze()
        le = LabelEncoder()
        y = le.fit_transform(y)
        return df, X, y

def getdata(openml_id:int = 44156, verbose = True):
    ds = openml.datasets.get_dataset(dataset_id=openml_id)
    if verbose:
        print(ds)
    return ds

def get_core_train_sample(y, fold_idx_list): # for sets with classes with very low number of samples. Gets a core training set with at least one of every class
    representatives = np.array([])
    fold_idx_list_out = fold_idx_list.copy()
    for y_val in np.unique(y):
        representatives =  np.append(representatives,np.where(y==y_val)[0][0])
    fold_idx_list_out[0] = np.unique(np.append(fold_idx_list[0], representatives)).astype(int)

    for i,idxs in enumerate(fold_idx_list_out):
        if i == 0: continue
        fold_idx_list_out[i] = np.array([x for x in fold_idx_list_out[i] if x not in fold_idx_list_out[0]])
    
    return fold_idx_list_out, representatives

    
def embedd_with_umap(df,embedding_dim = 10, metric='euclidean', scale=True):
    reducer = umap.UMAP(n_components=embedding_dim, metric=metric)
    if scale:
        scaler = StandardScaler()
        data = scaler.fit_transform(df.loc[:,df.dtypes == 'float'])
    else:
        data = df.loc[:,df.dtypes == 'float'].values
        
    reducer.fit(data)
    return reducer

def fit_dknn_toXy(X,y, k = 10, metric = 'euclidean'):
    clf = KNeighborsClassifier(n_neighbors=k, metric=metric, weights='distance')
    if isinstance(X,pd.Dataframe):
        clf.fit(X.values,y)
    else:
        clf.fit(X,y)
    return clf

def fit_dknn_toUMAP_reducer(reducer, y_train, k = 10, metric = 'euclidean'):
    clf = KNeighborsClassifier(n_neighbors=k, metric=metric, weights='distance')
    clf.fit(reducer.embedding_, y_train)
    return clf