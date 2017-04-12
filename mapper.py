# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 19:18:14 2017

@author: Isma

Mapper functions

Each function will return a tuple of (X_mapper, filter_columns). The mapper have a fit and a
 transform functions that map a DataFrame into a numpy array
 
filter_columns is the list of columns we want to filter for nan

Easiest way is to just use sklearn-pandas
"""

from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler, Imputer
from labelbinarizer2d import LabelBinarizer2D
import numpy as np
import pandas as pd
from normalimputer import NormalImputer


def filter_cols(X_mapper):
    columns = [c for (c,f) in X_mapper.features if type(f) != Imputer and ('Imputer' not in str(f))]        
    columns = [c[0] if type(c) == list else c for c in columns]
    
    return columns
    
def _get_features(col, imputer):
        if col.dtype == bool:
            return (col.name, None)
        elif col.dtype == np.float64:
            if (imputer == 'normal'):
                return ([col.name], [NormalImputer(), StandardScaler()])
            elif (imputer != None):
                return ([col.name], [Imputer(strategy=imputer), StandardScaler()])
            else:
                return ([col.name], StandardScaler())
        elif col.dtype == np.int64 or col.dtype == 'category':
            return ([col.name], LabelBinarizer2D())
        else:
            raise ValueError('unknown type for col %s'%col.name)
    

def columns_mapper(data, include_cols, imputer_strategy='mean'):
    """
    Mapper that allows to manually select columns
    
    The mapper will select the feature extraction depending on the columns selected
    
    """
    
    cols = list()
    for name, col in data.iteritems():
        if name not in include_cols:
            continue
        cols.append(_get_features(col, imputer_strategy))

    X_mapper = DataFrameMapper(cols, df_out=True)
    
    return X_mapper, filter_cols(X_mapper)        
    

def all_columns_mapper(data, imputer_strategy='mean', add_exclude_cols = []):

    cols = list()
    exclude_cols = [] + add_exclude_cols

    for name, col in data.iteritems():
        if name in exclude_cols:
            continue
        cols.append(_get_features(col, imputer_strategy))
            
    X_mapper = DataFrameMapper(cols, df_out=True)
    
    return X_mapper, filter_cols(X_mapper)
