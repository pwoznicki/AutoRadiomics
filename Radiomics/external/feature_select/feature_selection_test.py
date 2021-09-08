#!/usr/bin/env python

'''
An example file to show how to use the feature-selection code
'''
from tqdm import tqdm

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import depmeas
import feature_select

'''
An example script that shows how to use the feature-selection code in ml_lib
Performs Feature Selection the Leukemia dataset, which has 72 examples, and > 7000
features.  We compute the top 25 features and show a classifier performance as we
add in the features we compute
'''

if __name__=='__main__':
    NUM_CV = 3
    RANDOM_SEED = 123
    MAX_ITER = 1000

    leuk = fetch_mldata('leukemia', transpose_data=True)
    X = leuk['data']
    y = leuk['target']

    # split the data for testing
    (X_train,X_test,y_train,y_test) = train_test_split(X,y,test_size=0.3,random_state=RANDOM_SEED)

    # perform feature selection
    num_features_to_select = 25
    K_MAX = 1000
    estimator = depmeas.mi_tau
    n_jobs = -1
    feature_ranking_idxs = feature_select.feature_select(X_train,y_train,
        num_features_to_select=num_features_to_select,K_MAX=K_MAX,
        estimator=estimator,n_jobs=n_jobs)
    num_selected_features = len(feature_ranking_idxs)
    # for each feature, compute the accuracy on the test data as we add features
    mean_acc = np.empty((num_selected_features,))
    std_acc  = np.empty((num_selected_features,))
    for ii in tqdm(range(num_selected_features),desc='Computing Classifier Performance...'):
        classifier = svm.SVC(random_state=RANDOM_SEED,max_iter=MAX_ITER)
        X_test_in = X_test[:,feature_ranking_idxs[0:ii+1]]
        scores = cross_val_score(classifier, X_test_in, y_test, cv=NUM_CV, n_jobs=-1)

        mu = scores.mean()
        sigma_sq = scores.std()
        
        mean_acc[ii] = mu
        std_acc[ii] = sigma_sq

    x = np.arange(num_selected_features)+1
    y = mean_acc
    yLo = mean_acc-std_acc/2.
    yHi = mean_acc+std_acc/2.
    
    plt.plot(x,y)
    plt.fill_between(x,yLo,yHi,alpha=0.2)
    plt.grid(True)
    plt.title('Leukemia Dataset Feature Selection\n Total # Features=%d' % (X.shape[1]))
    plt.xlabel('# Selected Features')
    plt.ylabel('SVC Classifier Accuracy')
    plt.show()