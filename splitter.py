# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 19:18:13 2017

@author: Isma/Leo

Splitter functions

Each function will return a list of (train, test) dataframes for each cross validation split

If no cross validation split, just return a list with one (train, test) tuple

If the full data (no test set), just return a list with a (train, _) one-element tuple
"""

from sklearn.model_selection import KFold

def _filter_min_value(data, colName, minValue):
    
    if (minTempo is not None):
        data = data[~(data[colName] < minValue)]
        
    return data

def random_sample_range(data,constraintRangeName, rangeValues,constraintMinName = None, minValueTrain = None, minValueTest = None, n_fold=3, shuffle=True):
    """
    Filters out data in the *rangeValues* for the *constraintRangeName*  column, as well as any nan
    Do a random split of the data using K-Fold cross validation
    """
    data = data[data[constraintRangeName].isin(rangeValues)]
    
    kf = KFold(n_splits=n_fold, shuffle=shuffle, random_state = 385123)
    
    splits = []
    
    for (train_idx, test_idx) in kf.split(data):
        splits.append((_filter_min_value(data.iloc[train_idx], constraintMinName, minValueTrain), 
                     _filter_min_value(data.iloc[test_idx], constraintMinName, minValueTest)))
        
    return splits
    
    
def split_range(data, constraintRangeName, train_range, test_range,constraintMinName = None, minValueTrain = None, minValueTest = None,):
    """
    Filters the data in train/test sets depending on periodo letivo
    both train_years and test_years need to be lists
    """
    
    train = data[data['periodo_letivo'].isin(train_range)]
    test = data[data['periodo_letivo'].isin(test_range)]
    
    train = _filter_min_value(train, constraintMinName, minValueTrain)
    test = _filter_min_value(test, constraintMinName, minValueTest)
        
    return [(train, test)]

def overSampleData(data, multiplicationFactor = 4):
    idxTrue = (data['y'] == True)
    dataTrue = data[idxTrue]
    oversampledData = data
    for i in range(1,multiplicationFactor):
        oversampledData = oversampledData.append(dataTrue,ignore_index=True)    
        
    return oversampledData