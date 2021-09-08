#!/usr/bin/env python

'''
This file contains functions that enable MRMR based Feature Selection.
See the paper: Feature Selection Based on Mutual Information: Criteria of Max-Dependency, Max-Relevance,
and Min-Redundancy by H. Peng, F. Long, and C. Ding

Author: Kiran Karra [Virginia Tech]
        <kiran.karra@gmail.com, kiran.karra@vt.edu>

Distribution Statement A: Approved for Public Release, Distribution Unlimited
'''

import warnings
warnings.filterwarnings("ignore")

import os
import shutil

import json

from tempfile import mkdtemp

from tqdm import tqdm

from scipy.stats import randint as sp_randint

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve
import joblib
from joblib import Memory
from sklearn.metrics import matthews_corrcoef

import gplearn
import gplearn.genetic
import gplearn.fitness

import numpy as np

from joblib import Parallel, delayed
from joblib import load, dump

import sys
sys.path.append('/Users/p.woznicki/git/Radiomics/src/Radiomics/external/feature_select')

from identity_transformer import IdentityTransformer

import depmeas

def generic_combined_scorer(x1,o1,ii_1,x2,o2,ii_2,y,h):
    s1 = h(x1,y)
    s2 = h(x2,y)
    o1[ii_1] = s1
    o2[ii_2] = s2

def feature_select(X,y,num_features_to_select=None,K_MAX=1000,estimator=depmeas.mi_tau,n_jobs=-1,verbose=True):
    '''
    Implements the MRMR algorithm for feature-selection: http://ieeexplore.ieee.org/document/1453511/

    Inputs:
        X - A feature-matrix, of shape (N,D) where N is the number of samples and D is the number
            of features
        y - A vector of shape (N,1) which represents the output.  Each index in y is assumed to
            correspond to the row with the same index in X.
        num_features_to_select - the number of features to select from the provided X matrix.  If None
                                 are provided, then all the features that are available are ranked/ordered.
                                 (default: None)
        K_MAX - the maximum number of top-scoring features to consider.
        estimator - a function handle to an estimator of association (that theoretically should
                    follow the DPI assumptions)
        n_jobs - the numer of processes to use with parallel processing in the background
        verbose - if True, show progress

    Output:
        a vector of indices sorted in descending order, where each index represents the "importance"
        of the feature, as computed by the MRMR algorithm.
    '''
    num_dim = X.shape[1]

    if(num_features_to_select is not None):
        num_selected_features = min(num_dim,num_features_to_select)
    else:
        num_selected_features = num_dim
    K_MAX_internal = min(num_dim,K_MAX)

    initial_scores = Parallel(n_jobs=n_jobs)(delayed(estimator)(X[:,ii],y) for ii in range(num_dim))
    # rank the scores in descending order
    sorted_scores_idxs = np.flipud(np.argsort(initial_scores))
    
    # subset the data down so that joblib doesn't have to 
    # transport large matrices to its workers
    X_subset = X[:,sorted_scores_idxs[0:K_MAX_internal]]
    # memory map this for parallelization speed
    tmp_folder = mkdtemp()
    # TODO: why is X_subset crashing when we increase K_MAX_in?  Investigate in detail, but
    # for now, do not use memory mapping for X_subset for stability
    # X_subset_fname = os.path.join(tmp_folder, 'X_subset')
    # dump(X_subset, X_subset_fname)
    # X_subset = load(X_subset_fname, mmap_mode='r')

    selected_feature_idxs    = np.zeros(num_selected_features,dtype=int)
    remaining_candidate_idxs = list(range(1,K_MAX_internal))
    
    # mi_matrix = np.empty((K_MAX_internal,num_selected_features-1))
    # mi_matrix[:] = np.nan

    relevance_vec_fname = os.path.join(tmp_folder, 'relevance_vec')
    feature_redundance_vec_fname = os.path.join(tmp_folder, 'feature_redundance_vec')
    mi_matrix_fname = os.path.join(tmp_folder, 'mi_matrix')
    relevance_vec = np.memmap(relevance_vec_fname, dtype=float, 
                                shape=(K_MAX_internal,), mode='w+')
    feature_redundance_vec = np.memmap(feature_redundance_vec_fname, dtype=float, 
                                shape=(K_MAX_internal,), mode='w+')
    mi_matrix = np.memmap(mi_matrix_fname, dtype=float, 
                                shape=(K_MAX_internal,num_selected_features-1), mode='w+')
    mi_matrix[:] = np.nan

    # TODO: investigate whether its worth it to parallelize the nested for-loop?
    with tqdm(total=num_selected_features,desc='Selecting Features ...',disable=(not verbose)) as pbar:
        pbar.update(1)
        for k in range(1,num_selected_features):
            ncand = len(remaining_candidate_idxs)
            last_selected_feature = k-1

            Parallel(n_jobs=n_jobs)(delayed(generic_combined_scorer)(y,relevance_vec,ii,
                                                X_subset[:,selected_feature_idxs[last_selected_feature]],
                                                feature_redundance_vec,ii,X_subset[:,ii],
                                                estimator) 
                          for ii in remaining_candidate_idxs)

            # copy the redundance into the mi_matrix, which accumulates our redundance as we compute
            mi_matrix[remaining_candidate_idxs,last_selected_feature] = feature_redundance_vec[remaining_candidate_idxs]
            redundance_vec = np.nanmean(mi_matrix[remaining_candidate_idxs,:], axis=1)

            tmp_idx = np.argmax(relevance_vec[remaining_candidate_idxs]-redundance_vec)
            selected_feature_idxs[k] = remaining_candidate_idxs[tmp_idx]
            del remaining_candidate_idxs[tmp_idx]
            
            pbar.update(1)
    
    # map the selected features back to the original dimensions
    selected_feature_idxs = sorted_scores_idxs[selected_feature_idxs]

    # clean up
    try:
        shutil.rmtree(tmp_folder)
    except:
        pass

    return selected_feature_idxs
