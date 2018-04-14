#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Project Details: 
    dataset_loader.py 
    Loads the required signals of different severity levels into training and testing sets

Created on Tue Oct  3 18:34:37 2017

__author__      = "nnarenraju"
__copyright__   = "Copyright 2017, Severity Classification"
__credits__     = "nnarenraju"
__license__     = "Apache License 2.0"
__version__     = "1.0.1"
__maintainer__  = "nnarenraju"
__email__       = "nnarenraju@gmail.com"
__status__      = "inUsage"

Github Repository: "https://github.com/nnarenraju/sound-classification"

"""

import os
import glob
import pickle
import numpy as np


""" 
EXPECTS A WELL STRUCTURED DISTRIBUTION OF DATA
    
Structure anticipated is as follows:
------------------------------------

    [1] Main parent directory = "Directory"
    [2] Training directory and Testing directory
    [3] Subdirectory of different classes = "Class_1", "Class_2", ... , "Class_n"
    [4] Each Class subdirectory containing:
        [a] ".wav" files (main input files)
        [b] Input files of other extensions (possibly extracted features)

    *Names and paths can be set as required*

"""

print (__doc__)

# Do not enter class subdirectory information
# eg., /home/documents/parent_directory/
parent_directory = "Enter absolute path of input directory here"

""" Default Values """
# Training and testing set location
training_dir = "Training"
testing_dir = "Testing"

#.wav for audio signals
file_type = "*.wav"

# Class folder location (approximate name)
# Change as required (5 classes presented here)
no_classes = 5
_class = [""] * no_classes

_class[0] = "*mild*"
_class[1] = "*moderate*"
_class[2] = "*moderate2severe*"
_class[3] = "*severe*"
_class[4] = "*normal*"

#create a dictionary of class names and labels
#assign the label names to the corresponding class values when invoking the _get_class method

def _vectorised(label):
    """ Returns a one hot vector of given label """
    one_hot = np.zeros(len(_class))
    one_hot[label]=1
    return one_hot

def _get_class(class_name, label, file_type, set_dir):
    """ Returns a list of required absolute paths of the particular class """
    os.chdir(parent_directory+set_dir+"/")
    class_list=[]
    for file in glob.glob(class_name+"/"+"*"+file_type):
        class_list.append(file)
    # Sanity Check 0
    if not all(map(os.path.exists, class_list)):
        raise NameError("Input not found in given location")
    return (class_list, _vectorised(label))

def _get_dataset(classes, set_dir):
    """ Loads the locations of training and testing set data alongside labels """
    dataset = []
    for label, _class in enumerate(classes):
        dataset.append(_get_class(_class, label, file_type, set_dir))
    # Re-arrange data into dataset zip file
    with open("../../PATH_file/{}.pkl".format(set_dir), "wb") as f:
      pickle.dump((dataset), f, -1)
    return dataset

def load_data():
    """ Main funtion to dataset loader """
    # Training directory data retreaval
    training_data = _get_dataset(_class, training_dir)
    # Testing directory data retreaval
    testing_data = _get_dataset(_class, testing_dir)
    return training_data, testing_data
