#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:41:05 2024

@author: JSeaward
"""

import os, pickle
import numpy as np
import pandas as pd
from math import floor, ceil
from tqdm import tqdm, trange
import skimage as ski

from sklearn.neighbors import (
    NearestNeighbors,
)  # when loading multiple batches, just spit out n closest


# import joblib # for parallel saving and loading of dicts?

# in the end, want two collections of vectorized images and objects, each with one-hot encoded labels
# could even have subclaseses to make different "len()"?
# what sklearn wants are big-ass arrays for X and y, stratified K-fold just gives indicies


# sampling options set later
class Img_Vec_Dataset:  # really only depents on the label convetion.... so shoulld make a general class that goes and gets the right label machinery
    def __init__(
        self, train_dir_path, max_dim=None, label_type="yolo"
    ):  # may want to make a flag to save processed data arrays
        self.trainpath = train_dir_path
        self.img_path = os.path.join(self.trainpath, "images/")  # could make suffix arg
        self.label_path = os.path.join(self.trainpath, "labels/")
        if label_type == "yolo":
            self.label_module = YOLO_Labels(self)

        self.imc = ski.io.ImageCollection(
            self.img_path + "*", load_func=self._load_func
        )

        if max_dim is None:
            max_height, max_width = self._get_max_dims()
        else:
            max_height, max_width = max_dim, max_dim
            self.imc = ski.io.ImageCollection(
                self.img_path + "*", load_func=self._load_func, **{"max_dim": max_dim}
            )
        self.img_vec_channel_length = max_height * max_width
        self.classes, self.max_objs, self.obj_count = self._get_all_classes()

        self.img_vec_set = ImageVectorSet(self)
        self.obj_vec_set = ObjectVectorSet(self)

    def _load_func(self, f, max_dim=None):
        im = ski.io.imread(f)
        if max_dim is not None:
            aspect = im.shape[1] / im.shape[0]
            if aspect > 1:  # short image
                h = int(max_dim / aspect)
                im = ski.transform.resize(im, [h, max_dim])
            else:  # wide image
                w = int(max_dim * aspect)
                im = ski.transform.resize(im, [max_dim, w])
        return ski.util.img_as_ubyte(im)

    # don't do the flattening/concatenating colors here b/c it makes objects hard to get out
    # do any padding wuth the vectorization as well

    def _get_max_dims(self):
        h, w = 0, 0
        for im in tqdm(
            self.imc, desc="Scanning to find max image dimensions...", unit="Image file"
        ):
            H, W = im.shape[:2]
            if H > h:
                h = H
            if W > w:
                w = W
        return H, W

    def _get_all_classes(self):
        labels = set()
        max_labs = 0
        count_labs = 0
        for idx in trange(len(self.imc), desc="Detemining all classes..."):
            df = self.label_module.get_label_df(idx)
            labels = labels.union(df["class"])
            count_labs += len(df)
            if len(df) > max_labs:
                max_labs = len(df)
        return (
            labels,
            max_labs,
            count_labs,
        )  # final one-hot labels are like np.zeroes(len(labels))[label] = 1, but sklearn doesnt need one-hot labels; those are for features

    def get_objects(self, idx, return_df=False):
        im = self.imc[idx]
        obj_boxes, df = self.label_module.get_obj_bounds(idx)
        objs = []
        classes = []
        locs = []
        for i, (label, top, bottom, left, right) in enumerate(obj_boxes):
            obj = im[top:bottom, left:right]
            objs.append(obj)
            if return_df:
                df.loc[i, "top"] = top
                df.loc[i, "bottom"] = bottom
                df.loc[i, "left"] = left
                df.loc[i, "right"] = right
            else:
                classes.append(int(label))
                locs.append((top, left))
        if return_df:
            df["images"] = objs
            return df
        else:
            return zip(objs, classes, locs)


## A swappable lebels module that gets label dfs and px coords of objs


class YOLO_Labels:
    def __init__(self, dataset):
        self.dataset = dataset

    def get_label_df(self, idx):
        im_path = self.dataset.imc.files[idx]
        f_name = os.path.split(im_path)[-1][:-3] + "txt"
        f_path = os.path.join(self.dataset.label_path, f_name)
        df = pd.read_csv(
            f_path,
            delimiter=" ",
            header=None,
            names=["class", "x_center", "y_center", "w", "h"],  # from top left corner
        )
        return df

    def get_obj_bounds(self, idx):
        df = self.get_label_df(idx)
        H, W = self.dataset.imc[idx].shape[:2]
        obj_boxes = []
        for i, row in df.iterrows():
            top = floor((row.y_center - row.h / 2) * H)
            bottom = top + ceil(row.h * H)
            left = floor((row.x_center - row.w / 2) * W)
            right = left + ceil(row.w * W)
            obj_boxes.append((row["class"], top, bottom, left, right))
        return obj_boxes, df


class NearestVectorCaller:  # For datasets of vectors too big to fit in memmory. takes in a vector, returns n nearest in the underlying vec_set
    # damn! dont' actually need layers. Can always find an n that works
    def __init__(
        self,
        vec_set,
        in_path=None,
        out_path="./lg_vec_index.pkl",
        n=None,
        random_seed=0,
        batch_size=1024,
        max_iter=100,
        tol=0.01,
    ):  # set random seed to None for a surprise!
        from sklearn.cluster import MiniBatchKMeans

        self.out_path = out_path
        self.in_path = in_path
        self.f_name
        self.tol = tol
        self.vec_set = vec_set  # something with a getitem function that returns vectors
        if n is None:
            self.n = int(len(self.vec_set) ** 0.5)
        else:
            self.n = n

        if not os.path.isdir(os.path.split(self.out_path)[0]):
            os.makedirs(os.path.split(self.out_path)[0])

        if in_path is not None:
            self.index_kmeans = MiniBatchKMeans(
                n_clusters=self.n,
                random_state=self.random_seed,
                batch_size=batch_size,
                max_iter=max_iter,
            ).fit(vec_set)
            pickle.dump(self.index_kmeans, os.path.join(self.f_path, self.f_name))
        else:
            self.index_kmeans = pickle.load(open(self.in_path))

        self.index_dict = {
            k: np.where(self.index_kmeans.labels_ == k)[0]
            for k in sorted(self.index_kmeans.labels_)
        }

    ### can use kmeans.transform om the vector to see it it is close to multiple centers, since that returns distances
    def get_knn(self, vec, k=None):

        keys_to_load = self.index_kmeans.predict(vec)

        distances = self.index_kmeans.transform(vec).squeeze()
        sorted_distances = np.sort(distances)
        rel_err = np.array([1 - sorted_distances[0] / x for x in sorted_distances[1:]])
        close_sorted_idxs = np.where(rel_err < self.tol)[0] + 1

        if len(close_sorted_idxs > 0):
            close_idxs = np.where(distances == sorted_distances[close_sorted_idxs])[0]
            keys_to_load = np.append(keys_to_load, close_idxs)

        idxs_to_load = np.array([])
        for key in keys_to_load:
            idxs_to_load = np.append(idxs_to_load, self.index_dict[key])

        outX, outY = self._call_vec_set(idxs_to_load)

        if k is None and len(keys_to_load) == 1:
            return outX, outY

        else:
            k = int(len(self.vec_set) / self.n)
            nn = NearestNeighbors(n_neighbors=k).fit(outX)
            _, out_idxs = nn.kneighbors(vec)  ### Could return distance
            return self._call_vec_set(out_idxs.squeeze())

    def _call_vec_set(self, idx_list):
        outX, outY = [], []
        for idx in idx_list:
            x, y = self.vec_set[int(idx)]
            assert x is not None
            outX.append(x)
            outY.append(y)
        return np.stack(outX), np.stack(outY)


class VecGetter:
    def __init__(self, vec_set):
        self.vec_set = vec_set

    def __getitem__(self, idx):
        im, _ = self.vec_set[idx]
        return im

    def __len__(self):
        return len(self.vec_set)


class ImageVectorSet:
    def __init__(self, dataset):
        self.dataset = dataset  # the superior Img Vec Dataset
        self.X = VecGetter(self)

    def image_vector_and_binary_label(self, idx: int):
        im = self.dataset.imc[idx]
        df = self.dataset.label_module.get_label_df(idx)
        im_vec = im2vec(im, self.dataset.img_vec_channel_length)
        labs_binary = [0] * len(self.dataset.classes)
        for label in set(df["class"]):
            labs_binary[label] = 1  # make binary vector for **presence** of object
        return np.array(im_vec), np.array(labs_binary)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.image_vector_and_binary_label(idx)
        elif isinstance(idx, slice):
            holder_vecs, holder_labs = [], []
            step = idx.step if idx.step is not None else 1
            for i in range(idx.start, idx.stop, step):
                vec, labs = self.image_vector_and_binary_label(i)
                holder_vecs.append(vec)
                holder_labs.append(labs)
            return np.stack(holder_vecs), np.stack(holder_labs)

    def __len__(self):
        return len(self.dataset.imc)


class ObjectVectorSet:  # for the bbx objects in the images
    def __init__(self, dataset):
        self.dataset = dataset

    def get_obj_vectors(self, idx: int):
        ims = []
        labs = []
        for im, cl, loc in self.dataset.get_objects(idx):
            im_vec = np.append(
                loc, np.ravel(im, order="F")
            )  # add the top left corner location as the the first 2 dimensions
            ims.append(im_vec)
            labs.append(cl)
        return ims, labs

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.get_obj_vectors(idx)
        elif isinstance(idx, slice):
            holder_vecs, holder_labs = [], []
            step = idx.step if idx.step is not None else 1
            for i in range(idx.start, idx.stop, step):
                vecs, labs = self.get_obj_vectors(i)
                holder_vecs += vecs
                holder_labs += labs
            return holder_vecs, holder_labs

    def __len__(self):
        return self.dataset.obj_count


## utility functions
def im2vec(im, img_vec_channel_length):
    if len(im.shape) > 2:
        holder = []
        for ch in range(im.shape[2]):
            data = list(np.ravel(im[:, :, ch]))
            holder += data + [0] * (img_vec_channel_length - len(data))
    else:
        data = list(np.ravel(im))
        holder = data + [0] * (img_vec_channel_length - len(data))
    return holder


def pad_resize_im(im, size=None):
    amount = int(abs(im.shape[0] - im.shape[1]) / 2)
    pad_tuple = (
        ((0, 0), (amount, amount), (0, 0))
        if im.shape[1] < im.shape[0]
        else ((amount, amount), (0, 0), (0, 0))
    )
    padded_image = np.pad(
        im,
        pad_tuple,
        mode="constant",
        constant_values=0,
    )
    if size is not None:
        padded_image = ski.transform.resize(padded_image, [size, size])
    return padded_image
