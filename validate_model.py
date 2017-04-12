# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 17:03:46 2017

@author: leo/isma

Basic framework to validate the ML model

It receives a list of results, a boolean defining if it's running 
crossValidation and configData 

It returns a series of metrics for the performance of the model 

If it's executing crossValidation, it will pair the test data with the models. 
Otherwise it will run consider all test results

 
"""

import pandas as pd
import numpy as np

import splitter
import mapper
import models
import charts
import execute_model

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, log_loss, roc_auc_score, confusion_matrix
from time import time

def validate(trainedModels, configData, testData):
    
    results = None
    total_test_size = 0
    
    if('crossValidation' in configData):
        crossValidation = configData['crossValidation']
    else:
        crossValidation = True
    
    
    if(crossValidation is True):
        for i, (currModel, test) in enumerate(zip(trainedModels, testData)):
            
            r, executionSize = execute_model.execute([currModel], configData, [test])
            total_test_size += executionSize
            if(results is not None):
                results = results.append(r[0][1])
            else:
                results = r[0][1]
                
        assert(len(results)==total_test_size)    
        results = [results]
        
    else:
        r = execute_model.execute(trainedModels, configData, testData)
        results = [tr[1] for tr in r]

    for currResult in results:
        currResult = currResult.dropna()
        
        confusion = pd.DataFrame(data=confusion_matrix(currResult['y'], currResult['pred']))
        print("Confusion matrix:")
        print(confusion)
        print(classification_report(currResult['y'], currResult['pred']))
        print("Log Loss: %f"%log_loss(currResult['y'], currResult['prob']))
        print("ROC AUC secore: %f"%roc_auc_score(currResult['y'], currResult['prob']))
        
        ### Plot results
        #ordered_results = charts.plot_probabilities(results, .05)

        if configData.get('plotParams') is None:
            configData['plotParams'] = {}
        
        currResult= charts.plot_clustered_results(currResult, 50,    upperBound = .90, **configData.get('plotParams'))
        
        charts.plot_cancel_curve(currResult, **configData.get('plotParams'))
        charts.plot_probabilities_color(currResult, **configData.get('plotParams'))
        
    return results
    