#!/usr/bin/env python

from sys import platform
import os
import pickle

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import scipy.io as sio

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn import preprocessing

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd

miEstimators = ['cim','knn_1','knn_6','knn_20','ap','h_mi']

NUM_CV = 10
SEED = 123
MAX_NUM_FEATURES = 10
MAX_ITER = 1000

figures_folder = os.path.join(os.environ['HOME'],'ownCloud','PhD','sim_results','synthetic_feature_select_figures')

def getDataFolder():
    return os.path.join(os.environ['HOME'],'ownCloud','PhD','sim_results','synthetic_feature_select')

def readDataset(dataset):
    ds_lower = dataset.lower()
    z = sio.loadmat(os.path.join(getDataFolder(),ds_lower))
    
    XX = z['XX']
    X = XX[:,0:-1]
    y = XX[:,-1]
    
    miFeatureSelections = {}
    for miEstimator in miEstimators:
        try:
            featureVec = sio.loadmat(os.path.join(getDataFolder(),ds_lower+'_fs_'+miEstimator+'.mat'))
            miFeatureSelections[miEstimator] = featureVec['featureVec']
        except:
            miFeatureSelections[miEstimator] = None
    
    return (X,y,miFeatureSelections)

def getClassifierPerformanceWithOptimalfeatures(classifierStr, dataset):
    (X,y,miFeatureSelections) = readDataset(dataset)
    numActualFeaturesStr = os.path.splitext(dataset)[0].split('_')[3]
    numActualFeatures = int(re.findall("[\d]+", numActualFeaturesStr)[0])

    featureVec = 0:numActualFeatures
    # we do it this way to force the random seed to be the same for each iteration
    # to compare results more consistently
    if(classifierStr=='SVC'):
        classifier = svm.SVC(random_state=SEED,max_iter=MAX_ITER)
    elif(classifierStr=='RandomForest'):
        classifier = RandomForestClassifier(random_state=SEED,n_jobs=-2)
    elif(classifierStr=='KNN'):
        classifier = KNeighborsClassifier(n_jobs=-2)
    elif(classifierStr=='AdaBoost'):
        classifier = AdaBoostClassifier(random_state=SEED)

    X_in = X[:,featureVec]
    # normalize
    X_in = preprocessing.scale(X_in)
    scores = cross_val_score(classifier, X_in, y, cv=NUM_CV, n_jobs=-2)

    mu = scores.mean()
    sigma_sq = scores.std()*2
    
    return (mu,sigma_sq)

def evaluateClassificationPerformance(classifierStr, dataset):
    (X,y,miFeatureSelections) = readDataset(dataset)
    
    resultsMean = np.empty((len(miEstimators),MAX_NUM_FEATURES))
    resultsVar = np.empty((len(miEstimators),MAX_NUM_FEATURES))
    resultsMean.fill(np.nan)
    resultsVar.fill(np.nan)

    eIdx = 0
    for estimator in miEstimators:
        featureVecAsList = miFeatureSelections[estimator]
        if(featureVecAsList is not None):
            featureVec = np.squeeze(np.asarray(miFeatureSelections[estimator]))
            K = min(len(featureVec),MAX_NUM_FEATURES)
            for ii in range(1,K+1):
                colSelect = featureVec[0:ii]-1  # minus one to switch from index-by-1 (Matlab) 
                                                # to index-by-0 (Python)
                # we do it this way to force the random seed to be the same for each iteration
                # to compare results more consistently
                if(classifierStr=='SVC'):
                    classifier = svm.SVC(random_state=SEED,max_iter=MAX_ITER)
                elif(classifierStr=='RandomForest'):
                    classifier = RandomForestClassifier(random_state=SEED,n_jobs=-2)
                elif(classifierStr=='KNN'):
                    classifier = KNeighborsClassifier(n_jobs=-2)
                elif(classifierStr=='AdaBoost'):
                    classifier = AdaBoostClassifier(random_state=SEED)

                X_in = X[:,colSelect]
                # normalize
                X_in = preprocessing.scale(X_in)
                scores = cross_val_score(classifier, X_in, y, cv=NUM_CV, n_jobs=-2)

                mu = scores.mean()
                sigma_sq = scores.std()*2
                
                resultsMean[eIdx,ii-1] = mu
                resultsVar[eIdx,ii-1] = sigma_sq

        eIdx = eIdx + 1

    return (resultsMean,resultsVar)

def generate_skew_comparison_plots(algorithm='KNN'):
    datasetsToTest = ['gaussian_none_lo_4d','gaussian_none_med_4d','gaussian_none_hi_4d','gaussian_none_all_4d',
                      'gaussian_left_lo_4d','gaussian_left_med_4d','gaussian_left_hi_4d','gaussian_left_all_4d',
                      'gaussian_right_lo_4d','gaussian_right_med_4d','gaussian_right_hi_4d','gaussian_right_all_4d']

    # run the ML
    for datasetIdx in range(len(datasetsToTest)):
        print('*'*10 + ' ' + datasetsToTest[datasetIdx] + ' ' + '*'*10)        
        datasetToTest = datasetsToTest[datasetIdx]
        resultsDir = os.path.join(getDataFolder(), 'classification_results')
        try:
            os.makedirs(resultsDir)
        except:
            pass
        fname = os.path.join(resultsDir,datasetToTest+'_'+algorithm+'.pkl')
        if os.path.exists(fname):
            with open(fname,'rb') as f:
                dataDict = pickle.load(f)
            resultsMean = dataDict['resultsMean']
            resultsVar  = dataDict['resultsVar']
        else:
            # only run if the results don't already exist
            resultsMean, resultsVar = evaluateClassificationPerformance(algorithm,datasetToTest)
        
        optimalPerformanceMean,optimalPerformanceVar = getClassifierPerformanceWithOptimalfeatures(algorithm,datasetToTest)
        print(resultsMean,resultsVar)
        print(optimalPerformanceMean,optimalPerformanceVar)
        # store these results
        dataDict = {}
        dataDict['resultsMean'] = resultsMean/optimalPerformanceMean
        dataDict['resultsVar']  = resultsVar
        with open(fname,'wb') as f:
            pickle.dump(dataDict,f)

    # plot the stuff & store
    estimatorsLegend = map(lambda x:x.upper(),miEstimators)
    try:
        estimatorsLegend[estimatorsLegend.index('TAUKL')]  = r'$\tau_{KL}$'
    except:
        pass
    if('tau' in miEstimators):
        estimatorsLegend[estimatorsLegend.index('TAU')]  = r'$\tau$'
    # estimatorsLegend[estimatorsLegend.index('VME')]    = 'vME'
    estimatorsLegend[estimatorsLegend.index('KNN_1')]  = r'$KNN_1$'
    estimatorsLegend[estimatorsLegend.index('KNN_6')]  = r'$KNN_6$'
    estimatorsLegend[estimatorsLegend.index('KNN_20')] = r'$KNN_{20}$'

    datasets_to_plot = datasetsToTest
    for dataset in datasets_to_plot:
        resultsDir = os.path.join(getDataFolder(),'classification_results')
        outputFname = os.path.join(figures_folder,dataset+'.png')
        
        fig,ax = plt.subplots(1,4,sharex=True,sharey=True,figsize=(12,3))

        yMinVal = 1.0
        yMaxVal = 0.0
        for cIdx in [0]:
            f = os.path.join(resultsDir,dataset+'_'+algorithm+'.pkl')
            with open(f,'rb') as f:
                z = pickle.load(f)

            # # load the data for the inset plot also!
            # if(skewVal=='full'):
            #     ds_lower = dataset.lower()
            #     z_data = sio.loadmat(os.path.join(getDataFolder(dataset),ds_lower,'data.mat'))
            # else:
            #     ds_lower = dataset.lower()
            #     z_data = sio.loadmat(os.path.join(getDataFolder(dataset),ds_lower,'data_skew_'+'%0.02f'%(skewVal)+'.mat'))
            # y_train = z_data['y_train']
            # y_train = np.squeeze(np.asarray(y_train))
            # unique_elements, counts_elements = np.unique(y_train, return_counts=True)
            # print(counts_elements)
            # num_neg_one = counts_elements[0]
            # num_pos_one = counts_elements[1]

            lineHandlesVec = []
            for estimatorIdx in range(z['resultsMean'].shape[0]):
                resultsMean = z['resultsMean'][estimatorIdx,:]
                results2Var = z['resultsVar'][estimatorIdx,:]
                resultsStd = np.sqrt(results2Var/2.)
                xx = range(1,len(resultsMean)+1)

                y = resultsMean
                h = ax[cIdx].plot(xx, y)

                yLo = resultsMean-results2Var/2.
                yHi = resultsMean+results2Var/2.
                ax[cIdx].fill_between(xx, yLo, yHi, alpha=0.2)
                ax[cIdx].grid(True)
                # ax[cIdx].set_xticks([1,5,10,15])
                lineHandlesVec.append(h[0])
                
                if(min(yLo)<yMinVal):
                    yMinVal = min(yLo)
                if(max(yHi)>yMaxVal):
                    yMaxVal = max(yHi)

            ax[cIdx].set_title(dataset)
            if(cIdx==0):
                ax[cIdx].set_ylabel(dataset.upper()+'\nCV-10 Classification Accuracy')
            if(cIdx==1):
                ax[cIdx].set_xlabel('# Features')

            # # add an inset plot to show the data skew of output class labels via histogram
            # pos = ax[cIdx].get_position().bounds
            # a = plt.axes([pos[0],pos[1],0.05,0.15])
            # sns.barplot([-1,1],counts_elements,saturation=0.75)
            # #plt.xticks([])
            # a.xaxis.set_ticks_position('top')
            # plt.yticks([])
                
        plt.figlegend( lineHandlesVec, estimatorsLegend, loc = 'center right' )
        plt.savefig(outputFname, bbox_inches='tight')

if __name__=='__main__':
    generate_skew_comparison_plots(algorithm='KNN')