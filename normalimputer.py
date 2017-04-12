# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 17:35:00 2017

@author: Isma

Very simple imputer implementation that samples from a normal distribution

"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from sklearn.utils import check_array
import numpy as np
import numpy.ma as ma
import numpy.random as rand

def _get_mask(X, value_to_mask):
    """Compute the boolean mask X == missing_values."""
    if value_to_mask == "NaN" or np.isnan(value_to_mask):
        return np.isnan(X)
    else:
        return X == value_to_mask

class NormalImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self, missing_values="NaN"):
        self.missing_values = missing_values
        
    def fit(self, X, y=None):

        X = check_array(X, accept_sparse=None, dtype=np.float64,
                        force_all_finite=False)

        mask = _get_mask(X, self.missing_values)
        masked_X = ma.masked_array(X, mask=mask)
        
        self.mean = ma.mean(masked_X)
        self.std = ma.std(masked_X)
        
        return self
        
    def transform(self, X):
        
        check_is_fitted(self, 'mean')                
        check_is_fitted(self, 'std')                
        
        X = check_array(X, accept_sparse=None, dtype=np.float64,
                        force_all_finite=False, copy=True)
        
        invalid_mask = _get_mask(X, self.missing_values)

        n_missing = np.sum(invalid_mask)
        
        values = self.mean + self.std*rand.randn(n_missing) 
        
        coordinates = np.where(invalid_mask.transpose())[::-1]
        
        X[coordinates] = values
        
        return X