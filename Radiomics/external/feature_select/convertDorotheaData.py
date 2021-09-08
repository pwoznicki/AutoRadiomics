#!/usr/bin/env python

'''
Converts the dorothea training/validation data from .data files into .mat
files for speed of processing in Matlab
'''

import scipy.io as sio
import pandas as pd
import os
from sys import platform

if __name__=='__main__':
    if platform == "linux" or platform == "linux2":
        folder = '/home/kiran/ownCloud/PhD/sim_results/feature_select_challenge/dorothea/'
    elif platform == "darwin":
        folder = '/Users/Kiran/ownCloud/PhD/sim_results/feature_select_challenge/dorothea/'
    elif platform == "win32":
        folder = 'C:\\Users\\kiran\\ownCloud\\PhD\\sim_results\\feature_select_challenge\\dorothea'

    X_train = os.path.join(folder,'dorothea_train.data')
    y_train = os.path.join(folder,'dorothea_train.labels')

    X_valid = os.path.join(folder,'dorothea_valid.data')
    y_valid = os.path.join(folder,'dorothea_valid.labels')

    X_train_df = pd.read_csv(X_train,delim_whitespace=True,header=None,error_bad_lines=False)
    y_train_df = pd.read_csv(y_train,delim_whitespace=True,header=None,error_bad_lines=False)
    X_valid_df = pd.read_csv(X_valid,delim_whitespace=True,header=None,error_bad_lines=False)
    y_valid_df = pd.read_csv(y_valid,delim_whitespace=True,header=None,error_bad_lines=False)

    sio.savemat(os.path.join(folder,'data.mat'), mdict={'X_train': X_train_df.values,
                                                        'y_train': y_train_df.values,
                                                        'X_valid': X_valid_df.values,
                                                        'y_valid': y_valid_df.values})