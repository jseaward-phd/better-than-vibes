#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom types for Better Than Vibes Module

Created on Sun Jan 26 15:56:11 2025

@author: rakshat
"""
from numpy import ndarray
from pandas import DataFrame
from typing import Sequence, Union

Single_Label_Set = Sequence[int]
Multi_Label_Set = Sequence[Single_Label_Set]
Label_Set = Union[Single_Label_Set, Multi_Label_Set]

Single_Prediction_Set = Sequence[Sequence[float]]
Mulilabel_Prediction_Set = Sequence[Single_Prediction_Set]
Prediction_Set = Union[Single_Prediction_Set, Mulilabel_Prediction_Set]

Data_Features = Union[ndarray, DataFrame]
