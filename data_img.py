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

import torch
from torchvision.transforms import v2
import cv2

from typing import Optional, Sequence, Union, Callable
from pathlib import Path

# when loading multiple batches, just spit out n closest
from sklearn.neighbors import NearestNeighbors



# what sklearn wants are big-ass arrays for X and y, stratified K-fold just gives indicies

# TODO:
#
#       -- add train/test splitting to dataset object. maybe attach a torch DataLoader
#       -- import joblib # for parallel saving and loading?

# sampling options set later
# %% Over-arching task-specific dataset classes. E.g. image sets for object detection, tabular classification sets, etc


# may want to make a flag to save processed data arrays
class Img_VAE_Dataset: # rename since using BYOL approach
    def __init__(
        self,
        train_dir_path: str,
        max_dim: Optional[int] = None,
        label_type: str = "yolo",
    ):
        """
        Dataset for object detection in images using bounding boxes.

        Parameters
        ----------
        train_dir_path : str
            Path to the dataset folder containg two subfolders: "images" and "labels"
        max_dim : int, optional
            Maximum dimension the images can take in eiterdimension. If passed, i
            mages will be padded and rescalled to be a square of this dimesion
            for uniformity.
            The default is 256. Setting to None will pad all images to match the largest in the set.
        label_type : str, optional
            For using differen bounding box labels. The default, and currently only,
            value is "yolo".

        Returns
        -------
        None.

        """
        self.trainpath = train_dir_path
        self.img_path = os.path.join(self.trainpath, "images/")  # could make suffix arg
        self.label_path = os.path.join(self.trainpath, "labels/")
        self.max_dim = max_dim
        ## make different modules for differen label types
        if label_type == "yolo":
            self.label_module = YOLO_Labels(self)
        ## make load functions for different scenarios
        self.means, self.stds = self._get_image_norm_params()
        self._laod_func = (
            torch_load_fn(dims=(max_dim, max_dim), means=self.means, stds=self.stds)
            if max_dim is int
            else torch_load_fn(means=self.means, stds=self.stds)
        )

        self.imc = ski.io.ImageCollection(
            self.img_path + "*", load_func=self._laod_func
        )
        # self.img_vec_channel_length = max_height * max_width will do thhis wth a VAE
        self.classes, self.max_objs, self.obj_count = self._get_all_classes()

        self.vector_dataset = WholeImageSet(self)

        # self.obj_vec_set = ObjectVectorSet(self)

    def __getitem__(self, idx: Union[int, Sequence[int]]):
        return self.vector_dataset[idx]

    def __len__(self):
        return len(self.vector_dataset)

    def _get_image_norm_params(self):
        _laod_func = (
            test_load_fn(dims=(self.max_dim, self.max_dim))
            if self.max_dim is int
            else test_load_fn()
        )

        imc = ski.io.ImageCollection(self.img_path + "*", load_func=_laod_func)
        color = len(imc[0].shape) > 2
        channels = imc[0].shape[2] if color else 1

        px_mean, px_std = [np.zeros(channels)] * 2
        for n, im in tqdm(
            enumerate(imc),
            desc="Scanning to find normalization parameters...",
            unit="Image file",
        ):
            px_mean += (
                (im.mean(0).mean(0) - px_mean) / (n + 1)
                if channels == 2
                else (im.mean() - px_mean) / (n + 1)
            )
            px_std = (
                (im.std(0).std(0) - px_std) / (n + 1)
                if channels == 2
                else (im.std() - px_std) / (n + 1)
            )

        return px_mean, px_std

    def _get_image_dims(self):
        """
        Utility function to scan the image directory and return the largest
        width and height out of all the images. For sizing the images when no
        max_dim is passed when initializing the set.

        Returns
        -------
        h : int
            Max height.
        w : int
            Max width.
        mean : int
            Mean dimension.

        """

        h, w, mean, last_mean = [0] * 4
        for n, im in tqdm(
            enumerate(self.imc),
            desc="Scanning to find max image dimensions...",
            unit="Image file",
        ):
            H, W = im.shape[:2]
            mean = last_mean + (np.mean(H, W) - last_mean) / n

            if H > h:
                h = H
            if W > w:
                w = W
        return h, w, mean

    def _get_all_classes(self):
        """
        Utility function to find all classes in the dataset.

        Returns
        -------
        labels : set
            Set of all class labels.
        max_labs : int
            Maximum number of labels in any one image.
        count_labs : int
            Total number of bounding boxes in the set.

        """
        labels = set()
        max_labs = 0
        count_labs = 0
        for idx in trange(len(self.imc), desc="Detemining all classes..."):
            df = self.label_module.get_label_df(idx)
            labels = labels.union(df["class"])
            count_labs += len(df)
            if len(df) > max_labs:
                max_labs = len(df)
        return labels, max_labs, count_labs

    def get_objects(self, idx: int, return_df: bool = False):
        """
        Get the objects in an image; their image arrays, classes and the
        location of their top left corner in pixel coordinates, with origin at
        the top left.

        Parameters
        ----------
        idx : int
            Index of the image to retrieve the objects from.
        return_df : bool, optional
            Whether or not to return the results as a pandas DataFrame.
            The default is False, which retuns a zip object.

        Returns
        -------
        zip or pd.DataFrame
            Zip object is of form (imges, classes, locations).
            DataFrame has columns "class", "image" "top", "bottom", "left", and
            "right" (in pixel coordinates).

        """
        im = imc_framegetter(self.imc, idx)
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
            df["image"] = objs
            return df
        else:
            return zip(objs, classes, locs)


# %% Load functions
class og_load_fn:
    def __init__(self, max_dim: Union[int, Sequence[int]]):
        if isinstance(self.dims, int):
            max_dim = max_dim
        else:
            max_dim = max(max_dim)

    def __call__(self, f: Union[str, Path]):
        im = ski.io.imread(f)
        if self.max_dim is not None:
            aspect = im.shape[1] / im.shape[0]
            if aspect > 1:  # short image
                h = int(self.max_dim / aspect)
                im = ski.transform.resize(im, [h, self.max_dim])
            else:  # wide image
                w = int(self.max_dim * aspect)
                im = ski.transform.resize(im, [self.max_dim, w])
        return im


class test_load_fn:
    def __init__(self, dims: Optional[Union[int, Sequence[int]]] = None):
        if dims is None:
            dims = (640, 640)  # yolo default is (640,640)
        self.dims = dims
        self.transforms = v2.Compose(
            [
                LetterBox(new_shape=self.dims, scaleup=True), # scaleup=False
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

    def __call__(self, f: Union[str, Path]):
        im = ski.io.imread(f)
        im = self.transforms(im).numpy().transpose([1, 2, 0])
        return im


class torch_load_fn_old:
    def __init__(self, dims):
        self.dims = dims
        self.transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(self.dims),  # (None,max_size), yolo default is (640,640)
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.5479, 0.5197, 0.4716], std=[0.0498, 0.0468, 0.0421]
                ),
            ]
        )

    def __call__(self, f):
        im = ski.io.imread(f)
        im = self.transforms(im).numpy().transpose([1, 2, 0])
        return im


class torch_load_fn:
    def __init__(
        self, dims=None, means=[0.5479, 0.5197, 0.4716], stds=[0.0498, 0.0468, 0.0421]
    ):
        if dims is None:
            dims = (640, 640)  # yolo default is (640,640)
        self.dims = dims
        self.transforms = v2.Compose(
            [
                LetterBox(new_shape=self.dims, scaleup=False),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=means, std=stds),
            ]
        )

    def __call__(self, f):
        im = ski.io.imread(f)
        im = self.transforms(im).numpy().transpose([1, 2, 0])
        return im


# %% Label Modules: Swappable modules that gets label datafrabes and and pixel coordinatess of objects
class YOLO_Labels:
    def __init__(self, dataset):
        """
        Label module for YOLO labels. Expects the standard fomat of a folder with
        one .txt file per image, each .txt file having one boudning box per line
        of the form "class", "x_center", "y_center", "width", "height."
        Meant to be a label module for datasets in this file.

        Parameters
        ----------
        dataset : Img_Obj_Dataset, as in this file.

        """
        self.dataset = dataset

    def get_label_df(self, idx: int):
        """
        Retrieve the labels for an image indexed in the dataset by idx.

        Parameters
        ----------
        idx : int
            Index of data label to retrive.

        Returns
        -------
        df : pd.Dataframe
            Pandas DataFrame of YOLO labels for the objects in the idexed image.

        """
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

    def get_obj_bounds(self, idx: int):
        """
        Retrieve the bounding boxes of the objects in an image in pixel coodinates.

        Parameters
        ----------
        idx : int
            Index of the data item to retireve bounding boxes for.

        Returns
        -------
        obj_boxes : list of tuples
            List containig a tuple for each class bounding box. Takes the form
            [(class, top, bottom, left, right), (class, top, bottom, left, right),...].
        df : pd.Dataframe
            Dataframe of labels in YOLO format.

        """
        df = self.get_label_df(idx)
        H, W = imc_framegetter(self.dataset.imc, idx).shape[:2]
        obj_boxes = []
        for i, row in df.iterrows():
            top = floor((row.y_center - row.h / 2) * H)
            bottom = top + ceil(row.h * H)
            left = floor((row.x_center - row.w / 2) * W)
            right = left + ceil(row.w * W)
            obj_boxes.append((row["class"], top, bottom, left, right))
        return obj_boxes, df


# %% Vector Datasets: Return vectors and object labels. A lot of these can be hugely simplified with the imc framegetter
class VecGetter:
    def __init__(self, vec_set):
        """
        Return only data vectors, no labels.

        Parameters
        ----------
        vec_set : Vector Set, as in this section.
            Soemthing with a __getitem__ that returns (vectros, labels).

        """
        self.vec_set = vec_set

    def __getitem__(self, idx):
        im, _ = self.vec_set[idx]
        return im

    def __len__(self):
        return len(self.vec_set)


### TODO: a yolo v8 preprocess IVS
class ImageVectorDataSet:
    # mostly for the getitem to get around the fact that imc doesn't want to give back from index lists
    def __init__(self, dataset, getY: bool = True):
        """
        Vector Dataset superclass. For individual kinds of vectors, use a subclass
        with a specific getvec_fn

        Parameters
        ----------
        dataset : Dataset with a scikit image collection as .imc
            Dataset to wrap.
        getY : bool, optional
            Construct the full labl set and add as self.y. The defaulst is True.
            Set to false for truly massive datasets.

        Returns
        -------
        data vectors and labels as (array, array)

        """
        self.dataset = dataset  # the superior Img Vec Dataset
        self.X = VecGetter(self)
        if getY:
            self.y = self._get_y()

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx = idx[0]

        if isinstance(idx, int):
            return self.getvec_fn(idx)
        elif isinstance(idx, slice):
            holder_vecs, holder_labs = [], []
            if idx.start is None:
                start = 0
            elif idx.start == -1:
                start = len(self)
            else:
                start = idx.start
            if idx.stop is None or idx.stop == -1:
                stop = len(self)
            else:
                stop = idx.stop
            stop = idx.stop if idx.stop is not None else len(self)
            step = idx.step if idx.step is not None else 1
            for i in range(start, stop, step):
                vec, labs = self.getvec_fn(i)
                holder_vecs.append(vec)
                holder_labs.append(labs)
            return np.stack(holder_vecs), np.stack(holder_labs)
        elif isinstance(idx, (list, np.ndarray)):
            holder_vecs, holder_labs = [], []
            for i in idx:
                vec, labs = self.getvec_fn(i)
                holder_vecs.append(vec)
                holder_labs.append(labs)
            return np.stack(holder_vecs), np.stack(holder_labs)
        else:
            raise Exception("Passed index is of unknown type.")

    def __len__(self):
        return len(self.dataset.imc)

    def _get_y(self):
        all_labels = []
        for idx in range(len(self)):
            df = self.dataset.label_module.get_label_df(idx)
            labs_binary = [0] * len(self.dataset.classes)
            for label in set(df["class"]):
                labs_binary[label] = 1
            all_labels.append(labs_binary)
        y = np.stack(all_labels)
        return y


## These are the things that get custom vectorization methods
class WholeImageSet(ImageVectorDataSet):
    def getvec_fn(self, idx: int):
        im = imc_framegetter(self.dataset.imc, idx)
        df = self.dataset.label_module.get_label_df(idx)
        labs_binary = [0] * len(self.dataset.classes)
        for label in set(df["class"]):
            labs_binary[label] = 1  # make binary vector for **presence** of object
        return np.array(im), np.array(labs_binary)
    
class ImageVectorSet_old(ImageVectorDataSet): # new one will usr some embedder or other
    """
    Vector Dataset for whole images. Meant to be a sub-module for datasets in this file.
    """

    def getvec_fn(self, idx: int):
        """
        Gets the image vector for the image indexed by idx in the attached dataset.
        Makes a binary label for vector of length == number of classes in the dataset.
        A 1 in the lebel vector indicates an object is present. Uset to support
        the__getitem__ function of this class so that that function can accept slices.

        Parameters
        ----------
        idx : int
            Index of image in underlying set

        Returns
        -------
        array
            Single data vector and label as (array, array)

        """
        im = imc_framegetter(self.dataset.imc, idx)
        df = self.dataset.label_module.get_label_df(idx)
        im_vec = im2vec(im, self.dataset.img_vec_channel_length)
        labs_binary = [0] * len(self.dataset.classes)
        for label in set(df["class"]):
            labs_binary[label] = 1  # make binary vector for **presence** of object
        return np.array(im_vec), np.array(labs_binary)


class ObjectVectorSet(ImageVectorDataSet):
    """
    Not fully implimented. getting y doesn't work great since the length is different
    """

    def getvec_fn(self, idx: int):
        ims = []
        labs = []
        for im, cl, loc in self.dataset.get_objects(idx):
            # add the top left corner location as the the first 2 dimensions
            im_vec = np.append(loc, np.ravel(im, order="F"))
            ims.append(im_vec)
            labs.append(cl)
        return ims, labs

    def __len__(self):
        return self.dataset.obj_count


# %% For doing knn on datasets whose vector sets are too big to fit in memmory.
class NearestVectorCaller:
    # damn! dont' actually need layers. Can always find an n that works
    def __init__(
        self,
        vec_set,
        in_path: str = None,
        out_path: str = "./lg_vec_index.pkl",
        n: int = None,
        metric: str = "minkowski",
        random_seed: int = 0,
        batch_size: int = 1024,
        max_iter: int = 100,
        tol: float = 0.01,
        verbose: int = 1,
    ):
        """
        For doing knn on datasets whose vectors will not all fit in memory.
        Uses mini-batch k-means centroids as a set of index vectors. A test vector is
        compared to the index vectors and only the underlying vectors that fall
        near the index vector are loaded. In the case where the test vector is
        close to multiple regions, they are all loaded for nearest neighbor calulation.

        Parameters
        ----------
        vec_set : Vector Dataset, as above
            The large dataset to wrap.
        in_path : str, optional
            Path to pre-computed scikit-learn kmeans object which indexes vec_set.
            The default is None,
            i.e. compute from scratch.
        out_path : str, optional
            Path to save computed kmeans object. The default is "./lg_vec_index.pkl".
        metric : str or callable, optional. The default is "minkowski"
            Distance metric for nearest neghbor calculation. Try "cosine."
            See sklearn.neighbors.NearestNeighbors for more
        n : int, optional
            Number of means to calculate with kmeans. The default is None, i.e.
            the integer closest to sqrt(len(vec_set))
        random_seed : int, optional
            Seed for mini-batch kmeans calculation. The default is 0.
            Set to None to select a random seed.
        batch_size : int, optional
            Size of kmeans mini-batches. The default is 1024.
        max_iter : int, optional
            Maximum iterations for the mini-batch kmeans. The default is 100.
        tol : float, optional
            Tolerance for what index vectors are considered "close". Given in
            terms of the maximum on the ratio of how much further the index
            under consideration is to distance to the closest index vector.
            The default is 0.01.
        verbose : int, optional
            Verbosity of the minibatch kmeans fitter. The default is 1.
            Set to 0 for silent.

        Returns
        -------
        None.

        """
        # set random seed to None for a surprise!
        from sklearn.cluster import MiniBatchKMeans

        self.vec_set = vec_set
        self.in_path = in_path
        self.out_path = out_path
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol
        self.metric = metric

        if n is None:
            self.n = int(len(self.vec_set) ** 0.5)
        else:
            self.n = n

        if not os.path.isdir(os.path.split(self.out_path)[0]):
            os.makedirs(os.path.split(self.out_path)[0])

        if in_path is None:
            self.index_kmeans = MiniBatchKMeans(
                n_clusters=self.n,
                random_state=self.random_seed,
                batch_size=self.batch_size,
                max_iter=self.max_iter,
                verbose=1,
            ).fit(vec_set.X)
            if self.out_path is not None:
                pickle.dump(self.index_kmeans, open(self.out_path, "wb"))
        else:
            self.index_kmeans = pickle.load(open(self.in_path, "rb"))

        self.index_dict = {
            k: np.where(self.index_kmeans.labels_ == k)[0]
            for k in sorted(self.index_kmeans.labels_)
        }

    def get_knn(self, vec, k: int = None, tol: float = None, return_vecs: bool = True):
        """
        Takes in a vector, returns n nearest in the underlying vec_set,
        even if it is near theboarder of multiple kmeans regions

        Parameters
        ----------
        vec : array
            Test vector to find the neighbor of.
        k : int, optional
            Number of nearest neighbors to return. The default is None, which
            results in len(dataset) / # of means in the index
        tol : float, optional
            Tolerance for another index vector to be "close" and thus load its
            associated vectros from the dataset for knn calculation.
            The default is None, which uses the .tol from the caller (default of 1%).
        return_vecs : bool, optional
            Whether or not to return the actural vectors. The defaul is True.
            Setting to False will return indeces only.

        Returns
        -------
        array
            k nearest vectors from the dataset. Suppressed if return_vecs is False.
        array
            Labels of those k nearest vectors. Suppressed if return_vecs is False.
        array
            Indeces in the vector dataset of the k nearest neighbors.
        """
        if len(vec.shape) < 2:
            vec = vec.reshape(1, -1)
        if tol is None:
            tol = self.tol
        keys_to_load = self.index_kmeans.predict(vec)
        keys_to_load_set = set(keys_to_load)

        distances = self.index_kmeans.transform(vec)
        sorted_distances = np.sort(distances)
        rel_err = 1 - sorted_distances[:, 0].reshape([-1, 1]) / sorted_distances[:, 1:]
        a, b = np.where(rel_err < tol)
        keys_to_load_set = keys_to_load_set.union(
            set(
                [
                    np.where(distances[x] == sorted_distances[x, y])[0].item()
                    for x, y in zip(a, b)
                ]
            )
        )

        idxs_to_load = np.array([])
        for key in keys_to_load_set:
            idxs_to_load = np.append(idxs_to_load, self.index_dict[key])

        if k is None and len(keys_to_load_set) == 1:
            if return_vecs:
                outX, outY = self.call_vec_set(idxs_to_load)
                return outX, outY, idxs_to_load
            else:
                return idxs_to_load

        else:
            if k is None:
                k = int(len(self.vec_set) / self.n)

            nn = NearestNeighbors(n_neighbors=k, metric=self.metric).fit(outX)
            _, out_idxs = nn.kneighbors(vec)  ### Could return distance
            if return_vecs:
                outX, outY = self.call_vec_set(out_idxs)
                return outX, outY, out_idxs
            else:
                return out_idxs

    def call_vec_set(self, idx_RA):
        if len(idx_RA.shape) < 2:
            idx_RA = idx_RA.reshape([1, -1])
        outX, outY = [], []
        for idx_list in idx_RA:
            for idx in idx_list:
                x, y = self.vec_set[int(idx)]
                assert x is not None
                outX.append(x)
                outY.append(y)
        outX = np.stack(outX)
        outY = np.stack(outY)
        _, unique_idx = np.unique(outX, axis=0, return_index=True)
        outX = outX[sorted(unique_idx)]
        outY = outY[sorted(unique_idx)]
        return outX, outY

    def get_knn_by_idx(self, idx_RA):
        out_X, out_y = self.call_vec_set(idx_RA)


# %% Utility Functions
from torchvision.datasets.vision import VisionDataset


# class IMCWrapper(ski.io.ImageCollection):
#     def __init__(self, imc):
#         self.imc = imc

#     def __len__(self):
#         return len(self.imc)

#     def __getitem__(self, idx):
#         if isinstance(idx, tuple):
#             idx = idx[0]

#         if isinstance(idx, int):
#             return self.imc[idx]
#         elif isinstance(idx, slice):
#             holder_ims = []
#             if idx.start is None:
#                 start = 0
#             elif idx.start == -1:
#                 start = len(self.imc)
#             else:
#                 start = idx.start
#             if idx.stop is None or idx.stop == -1:
#                 stop = len(self.imc)
#             else:
#                 stop = idx.stop
#             stop = idx.stop if idx.stop is not None else len(self.imc)
#             step = idx.step if idx.step is not None else 1
#             for i in range(start, stop, step):
#                 im = self.imc[i]
#                 holder_ims.append(im)
#             return np.stack(holder_ims)
#         elif isinstance(idx, (list, np.ndarray)):
#             holder_ims = []
#             for i in idx:
#                 im = self.imc[i]
#                 holder_ims.append(im)
#             return np.stack(holder_ims)
#         else:
#             raise Exception("Passed index is of unknown type.")


def imc_framegetter(imc, idx):
    if isinstance(idx, tuple):
        idx = idx[0]

    if isinstance(idx, int):
        return imc[idx]
    elif isinstance(idx, slice):
        holder_ims = []
        if idx.start is None:
            start = 0
        elif idx.start == -1:
            start = len(imc)
        else:
            start = idx.start
        if idx.stop is None or idx.stop == -1:
            stop = len(imc)
        else:
            stop = idx.stop
        stop = idx.stop if idx.stop is not None else len(imc)
        step = idx.step if idx.step is not None else 1
        for i in range(start, stop, step):
            im = imc[i]
            holder_ims.append(im)
        return np.stack(holder_ims)
    elif isinstance(idx, (list, np.ndarray)):
        holder_ims = []
        for i in idx:
            im = imc[i]
            holder_ims.append(im)
        return np.stack(holder_ims)
    else:
        raise Exception("Passed index is of unknown type.")


class VAEImageCollectionSet(VisionDataset):
    def __init__(
        self,
        imc: ski.io.ImageCollection,
        split: str,
        transform: Optional[Callable] = None,
    ):
        self.split = split
        self.imc = imc
        self.transform = transform

    def __len__(self):
        return len(self.imc)

    def __getitem__(self, idx: int):
        im = self.imc[idx]
        if self.transform is not None:
            im = self.transform(im)
        return im, torch.zeros(40, dtype=int)


def im2vec(im, img_vec_channel_length):
    """
    Converts an image array to a vector.

    Parameters
    ----------
    im : array
        Image to convert.
    img_vec_channel_length : int
        Number of pixels in each channel in the image; i.e. width * height

    Returns
    -------
    holder : list
        Vector of the image, padded at the end of each channel with 0.

    """
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
    """
    Pad and resize image to a specified size (height by width)

    Parameters
    ----------
    im : array
        The image to resize.
    size : iterable, optional
        Dimensions (H, W) to output. The default is None, i.e. only pad.

    Returns
    -------
    padded_image : array
        Padded/resized image.

    """
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


# def build_umap_reducer(ivds, embedding_dim=256, metric="cosine", block_frac=1):
#     block_sz = len(ivds) * block_frac
#     reducer = umap.UMAP(n_components=embedding_dim, metric=metric)
#     for ...:
#     return reducer


# ripped off from ultralytics to make image argument in __call__ come first
class LetterBox:
    """
    Resize image and padding for detection, instance segmentation, pose.

    This class resizes and pads images to a specified shape while preserving aspect ratio. It also updates
    corresponding labels and bounding boxes.

    Attributes:
        new_shape (tuple): Target shape (height, width) for resizing.
        auto (bool): Whether to use minimum rectangle.
        scaleFill (bool): Whether to stretch the image to new_shape.
        scaleup (bool): Whether to allow scaling up. If False, only scale down.
        stride (int): Stride for rounding padding.
        center (bool): Whether to center the image or align to top-left.

    Methods:
        __call__: Resize and pad image, update labels and bounding boxes.

    Examples:
        >>> transform = LetterBox(new_shape=(640, 640))
        >>> result = transform(labels)
        >>> resized_img = result["img"]
        >>> updated_instances = result["instances"]
    """

    def __init__(
        self,
        new_shape=(640, 640),
        auto=False,
        scaleFill=False,
        scaleup=True,
        center=True,
        stride=32,
    ):
        """
        Initialize LetterBox object for resizing and padding images.

        This class is designed to resize and pad images for object detection, instance segmentation, and pose estimation
        tasks. It supports various resizing modes including auto-sizing, scale-fill, and letterboxing.

        Args:
            new_shape (Tuple[int, int]): Target size (height, width) for the resized image.
            auto (bool): If True, use minimum rectangle to resize. If False, use new_shape directly.
            scaleFill (bool): If True, stretch the image to new_shape without padding.
            scaleup (bool): If True, allow scaling up. If False, only scale down.
            center (bool): If True, center the placed image. If False, place image in top-left corner.
            stride (int): Stride of the model (e.g., 32 for YOLOv5).

        Attributes:
            new_shape (Tuple[int, int]): Target size for the resized image.
            auto (bool): Flag for using minimum rectangle resizing.
            scaleFill (bool): Flag for stretching image without padding.
            scaleup (bool): Flag for allowing upscaling.
            stride (int): Stride value for ensuring image size is divisible by stride.

        Examples:
            >>> letterbox = LetterBox(new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, stride=32)
            >>> resized_img = letterbox(original_img)
        """
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left

    def __call__(self, image=None, labels=None):
        """
        Resizes and pads an image for object detection, instance segmentation, or pose estimation tasks.

        This method applies letterboxing to the input image, which involves resizing the image while maintaining its
        aspect ratio and adding padding to fit the new shape. It also updates any associated labels accordingly.

        Args:
            labels (Dict | None): A dictionary containing image data and associated labels, or empty dict if None.
            image (np.ndarray | None): The input image as a numpy array. If None, the image is taken from 'labels'.

        Returns:
            (Dict | Tuple): If 'labels' is provided, returns an updated dictionary with the resized and padded image,
                updated labels, and additional metadata. If 'labels' is empty, returns a tuple containing the resized
                and padded image, and a tuple of (ratio, (left_pad, top_pad)).

        Examples:
            >>> letterbox = LetterBox(new_shape=(640, 640))
            >>> result = letterbox(labels={"img": np.zeros((480, 640, 3)), "instances": Instances(...)})
            >>> resized_img = result["img"]
            >>> updated_instances = result["instances"]
        """
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = (
                new_shape[1] / shape[1],
                new_shape[0] / shape[0],
            )  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # for evaluation

        if len(labels):
            labels = self._update_labels(labels, ratio, left, top)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """
        Updates labels after applying letterboxing to an image.

        This method modifies the bounding box coordinates of instances in the labels
        to account for resizing and padding applied during letterboxing.

        Args:
            labels (Dict): A dictionary containing image labels and instances.
            ratio (Tuple[float, float]): Scaling ratios (width, height) applied to the image.
            padw (float): Padding width added to the image.
            padh (float): Padding height added to the image.

        Returns:
            (Dict): Updated labels dictionary with modified instance coordinates.

        Examples:
            >>> letterbox = LetterBox(new_shape=(640, 640))
            >>> labels = {"instances": Instances(...)}
            >>> ratio = (0.5, 0.5)
            >>> padw, padh = 10, 20
            >>> updated_labels = letterbox._update_labels(labels, ratio, padw, padh)
        """
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels

