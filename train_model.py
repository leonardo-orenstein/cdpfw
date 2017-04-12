# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 19:13:29 2017

@author: Isma/Leo

Basic framework to train the ML model

It receives a configData with the information for

 - splitter:        loads the data and splits into train/test and any cross-validation
 - mapper:          gets the dataframe and maps into columns
 - modelType:       defines which model will be used, e.g.: rnn, svm, randomForest
 - modelParams:     define the models parameters
 - dataFileName:    the filename with the data to be used
 
It returns the trained model
 
"""

import pandas as pd
import splitter
import mapper
import models 

from time import time

def train(configData, train_splits):
    
    mapperFunc      = configData['mapper']
    mapperParams    = configData.get('mapperParams')
    if (mapperParams is None):
        mapperParams = {}
    modelType       = configData['modelType']
    modelParams     = configData.get('modelParams')
    if (modelParams is None):
        modelParams = {}
    
    ### Define the model
    
    if(configData.get('overSampleFactor') is None):
        overSampleFactor = 1   
    else:
        overSampleFactor = configData['overSampleFactor']
        
        
    trainedModels = []

    for i, train in enumerate(train_splits):

        ## Get new model every time        
        model = models.__getattribute__(modelType)(**modelParams)
    
        start = time()
        
        print("Running CV %d on Train: %s"%(i, train.shape))
    
        X_mapper, filter_cols = mapper.__getattribute__(mapperFunc)(train, **mapperParams)
        
        ### Run the mapper on the train data set
        if (len(filter_cols)>0):
            train = train.loc[train[filter_cols].dropna().index]

        if overSampleFactor == 'auto':            
            ## automatic oversample the training data to the inverse frequency of the data
            train = splitter.overSampleData(train, (int)(1/train['y'].mean()))            
        else:
            train = splitter.overSampleData(train, overSampleFactor)
        X = X_mapper.fit_transform(train)
        y = train['y']
        
        print("Training filtered X_train: %d x %d"%(X.shape))
        
        # Train the model
        model.fit(X, y)
        
        trainedModels.append((model,X_mapper))
        
        print("Time: %.2f min\n"% ((time()-start)/60))

    return trainedModels