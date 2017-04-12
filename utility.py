# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 13:32:14 2017

@author: leo
"""

"""
eleva_main - Main file for the eleva project. Acts as a mediator between all the 
components needed for creating, executing and validating the model, as well as handling 
configuration and save results

"""

import pandas as pd
import numpy as np
import json
import sys

import dill

import splitter
import mapper
import models
import charts

import train_model
import execute_model
    

functions2Srl = ('mapper', 'splitter')

def getData(configurationData):
    
    splitFunction   = configurationData['splitter']
    splitParams = configurationData.get('splitterParams')
    if (splitParams is None):
        splitParams = {}
    dataFileName    = configurationData['dataFileName']    
    
    data = pd.read_pickle(dataFileName)
    splits = splitter.__getattribute__(splitFunction)(data, **splitParams)
    ### return train_splits, test_splits
    return ([train for (train,test) in splits], [test for (train,test) in splits])

    
def loadConfiguration(configurationFileName = None):
    """
    Load the configuration used on the last run or defined by configurationFile.
    It returns a config packet, that is used to define the behaviour of the other
    functions
    """
    
    if(configurationFileName is None):
        configurationFileName = 'lastConfiguration.pkl'
        
    file = open(configurationFileName, 'rb')

    return dill.load(file)
    
def saveConfiguration(configurationData, configurationFileName = None):
    """
    Saves the configuration passed as a input. If no name is defined, uses the 
    default name 'lastConfiguration'
    """
        
    if(configurationFileName  is None):
        configurationFileName  = 'lastConfiguration.pkl'
        
        
    file = open(configurationFileName, 'wb')

    dill.dump(configurationData, file)    
  