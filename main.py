#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Project Details: 
    main.py 
    This is the main file to access all sub-files for severity classification

Created on Fri Oct 13 16:07:59 2017

__author__      = "nnarenraju"
__copyright__   = "Copyright 2017, NameOfProject"
__credits__     = "nnarenraju"
__license__     = "Apache License 2.0"
__version__     = "1.0.1"
__maintainer__  = "nnarenraju"
__email__       = "nnarenraju@gmail.com"
__status__      = "inUsage"

Github Repository: "https://github.com/nnarenraju/sound-classification"

"""

import dataset_loader as load
import MFCC_feature_extract as MFCC
import network_CNN as CNN

if __name__ == "__main__":
    # Obtaining the training and testing dataset
    training_data, testing_data = load.load_data()
    
    # Computing the MFCC of .wav files to extract features
    train_feat = MFCC.MFCC_init(training_data, "training")
    test_feat = MFCC.MFCC_init(testing_data, "testing")
    
    # Splitting into feature input and labels
    # data is anticipated to be a list of lists
    # labels are anticipated to be a list
    train_data, train_labels = train_feat
    eval_data, eval_labels = test_feat
    
    # Running the classification algorithm
    CNN.run_network(train_feat, test_feat)