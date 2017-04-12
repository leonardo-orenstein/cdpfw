# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 15:30:46 2017

@author: leo/isma

Basic framework to execute the ML model

It receives a list of trainedModel, a boolean defining if it's running 
crossValidation and configData with the information for

 - splitter:        loads the data and splits into train/test and any cross-validation
 - mapper:          gets the dataframe and maps into columns
 - dataFileName:    the filename with the data to be used

It returns a list of results dataFrame, indexed according to data, with the prediction,
the y value, and the probability.

If it's executing crossValidation, it will pair the test data with the models. 
Otherwise it will run all the test datas in all models

 
"""

import pandas as pd
import numpy as np
import itertools 

import mapper

from time import time

def execute(trainedModels, configData, testData):
    
    mapperFunc      = configData['mapper']
    mapperParams    = configData.get('mapperParams')
    if (mapperParams is None):
        mapperParams = {}

    total_test_size = 0
    global_start = time()

    executionResults = []
    executionPackage = []
    
    for i, (currModel, test) in enumerate(itertools.product(trainedModels, testData)):

        start = time()
        
        (model, X_mapper) = currModel

        _, filter_cols = mapper.__getattribute__(mapperFunc)(test, **mapperParams)
                
        if (len(filter_cols)>0):
            test = test.loc[test[filter_cols].dropna().index]
        
        X_test = X_mapper.transform(test)
        y_test = test['y']
        
        print("Predicting filtered X_test: %d x %d"%(X_test.shape))
        
        results = pd.DataFrame(index=test.index, columns=['pred', 'y', 'prob'])
        results.loc[test.index, 'pred'] = model.predict(X_test)
        results.loc[test.index, 'prob'] = model.predict_proba(X_test)[:,1]
        results.loc[test.index, 'y'] = y_test
        
        results = results.astype({'pred':bool, 'y':bool, 'prob':np.float64})
        
        total_test_size += len(test)
        executionResults.append((model, results))
        
        print("Time: %.2f min\n"% ((time()-start)/60))
            
    return executionResults, total_test_size
        



def vote(trainedModels, configData, testData):
       
    total_test_size = 0
    global_start = time()
    
    
    td = None
    
    #Merge test data        
    for t in testData:
        if(td is not None):
            td = td.append(t)
        else:
            td = t
            
    executionResults, total_test_size = execute(trainedModels, configData, [td])
    
    votingResults = None
    
    for i, (currModel, r) in enumerate(executionResults):

        start = time()
        
        if votingResults is None:
            votingResults = r
            votingResults[ 'prob_mdl_%d' %i] = r.prob
        else:
            votingResults[ 'prob_mdl_%d' %i] = r.prob
            votingResults.prob = (r.prob + votingResults.prob*i)/(i + 1)
            
    return votingResults , total_test_size
        


