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

import itertools
from matplotlib.lines import Line2D

miEstimators = ['cim', 'taukl',
                'knn_1','knn_6','knn_20','vme', 'ap', 'h_mi', 
                'dCor', 'MIC', 'corr', 'RDC']
# miEstimators_toPlot = ['cim','taukl','dCor','MIC','RDC']
miEstimators_toPlot = ['cim','knn_1','knn_6','knn_20','vme', 'ap', 'h_mi']

classifiersToTest = ['SVC','RandomForest','KNN']

# the datasets tha we include in the paper
datasetsToTest = ['Arcene','Dexter','Madelon','Gisette',
                  'arcene_0.10','dexter_0.10','madelon_0.10','gisette_0.10',
                  'arcene_0.25','dexter_0.25','madelon_0.25','gisette_0.25',
                  'arcene_0.50','dexter_0.50','madelon_0.50','gisette_0.50',
                  'arcene_0.75','dexter_0.75','madelon_0.75','gisette_0.75']

NUM_CV = 10
SEED = 123
MAX_NUM_FEATURES = 50
MAX_ITER = 1000

figures_folder = os.path.join(os.environ['HOME'],'ownCloud','PhD','sim_results','feature_selection_results_figures')

def getDataFolder(dataset):
    dsl = dataset.lower()
    if('arcene' in dsl or 
       'dexter' in dsl or 
       'dorothea' in dsl or 
       'madelon' in dsl or
       'gisette' in dsl):
        folder = os.path.join(os.environ['HOME'],'ownCloud','PhD','sim_results','feature_select_challenge')
    elif(dsl=='drivface'):
        folder = os.path.join(os.environ['HOME'],'ownCloud','PhD','sim_results',dsl)
    elif(dsl=='mushrooms' or
         dsl=='phishing'):
        folder = os.path.join(os.environ['HOME'],'ownCloud','PhD','sim_results','libsvm_datasets')
    elif(dsl=='rf_fingerprinting'):
        folder = os.path.join(os.environ['HOME'],'ownCloud','PhD','sim_results',
            'rf_fingerprinting','data','fs_results')
    return folder

def readDataset(dataset):
    dsl = dataset.lower()
    if(dsl=='arcene' or 
       dsl=='dexter' or 
       dsl=='dorothea' or 
       dsl=='madelon' or
       dsl=='gisette'):
        return _readNips2003Data(dataset)
    elif(dsl=='drivface' or
         dsl=='mushrooms' or
         dsl=='phishing'):
        return _readLibsvmDatasets(dataset)
    elif(dsl=='rf_fingerprinting'):
        return _readRfFingerprintingDatasets()
    elif('_' in dsl):
        dsl_split = dsl.split('_')
        dsl = dsl_split[0]
        skew = dsl_split[1]
        return _readNips2003Data_skewed(dsl,skew)

def _readRfFingerprintingDatasets():
    # static configuration ... do we need to parametrize?
    numQ = 3
    numSamplesForQ = 200

    # read the data
    fName = 'data_numQ_%d_numSampsForQ_%d.csv' % (numQ,numSamplesForQ)
    df = pd.read_csv(os.path.join(getDataFolder('rf_fingerprinting'),'..',fName))

    feature_cols = df.columns.tolist()
    feature_cols.remove('transmitter_id')
    X = df[feature_cols].values
    y = df['transmitter_id'].values

    # read  feature selection vectors
    miFeatureSelections = {}
    for miEstimator in miEstimators:
        try:
            matFile = 'data_numQ_%d_numSampsForQ_%d.csv_fs_%s.mat' % (numQ,numSamplesForQ,miEstimator)
            matFileAndPath = os.path.join(getDataFolder('rf_fingerprinting'),matFile)
            featureVec = sio.loadmat(matFileAndPath)
            miFeatureSelections[miEstimator] = featureVec['featureVec']
        except:
            miFeatureSelections[miEstimator] = None

    return (X,y,miFeatureSelections)

def _readLibsvmDatasets(dataset):
    ds_lower = dataset.lower()
    z = sio.loadmat(os.path.join(getDataFolder(dataset),ds_lower+'_data.mat'))
    
    X = z['X']
    y = z['y']
    y = y.flatten()

    miFeatureSelections = {}
    for miEstimator in miEstimators:
        try:
            featureVec = sio.loadmat(os.path.join(getDataFolder(dataset),ds_lower+'_fs_'+miEstimator+'.mat'))
            miFeatureSelections[miEstimator] = featureVec['featureVec']
        except:
            miFeatureSelections[miEstimator] = None
        
    return (X,y,miFeatureSelections)

def _readNips2003Data(dataset):
    ds_lower = dataset.lower()
    z = sio.loadmat(os.path.join(getDataFolder(dataset),ds_lower,'data.mat'))
    
    X_train = z['X_train']
    y_train = z['y_train']
    X_valid = z['X_valid']
    y_valid = z['y_valid']

    y_train = np.squeeze(np.asarray(y_train))
    y_valid = np.squeeze(np.asarray(y_valid))
    X = np.vstack((X_train,X_valid))
    y = np.append(y_train,y_valid)
    
    miFeatureSelections = {}
    for miEstimator in miEstimators:
        try:
            featureVec = sio.loadmat(os.path.join(getDataFolder(dataset),ds_lower,ds_lower+'_fs_'+miEstimator+'.mat'))
            miFeatureSelections[miEstimator] = featureVec['featureVec']
        except:
            miFeatureSelections[miEstimator] = None
    
    return (X,y,miFeatureSelections)

def _readNips2003Data_skewed(dataset,skew=0.1):
    ds_lower = dataset.lower()
    z = sio.loadmat(os.path.join(getDataFolder(dataset),ds_lower,'data_skew_'+str(skew)+'.mat'))
    
    X_train = z['X_train']
    y_train = z['y_train']
    X_valid = z['X_valid']
    y_valid = z['y_valid']

    y_train = np.squeeze(np.asarray(y_train))
    y_valid = np.squeeze(np.asarray(y_valid))
    X = np.vstack((X_train,X_valid))
    y = np.append(y_train,y_valid)
    
    miFeatureSelections = {}
    for miEstimator in miEstimators:
        try:
            featureVec = sio.loadmat(os.path.join(getDataFolder(dataset),ds_lower,ds_lower+'_skew_'+str(skew)+'_fs_'+miEstimator+'.mat'))
            miFeatureSelections[miEstimator] = featureVec['featureVec']
        except:
            miFeatureSelections[miEstimator] = None
    
    return (X,y,miFeatureSelections)

def evaluateClassificationPerformance(classifierStr, dataset):
    (X,y,miFeatureSelections) = readDataset(dataset)

    # resultsMean = np.empty((len(miEstimators),MAX_NUM_FEATURES))
    # resultsVar = np.empty((len(miEstimators),MAX_NUM_FEATURES))
    # resultsMean.fill(np.nan)
    # resultsVar.fill(np.nan)
    resultsDict_mean = {}
    resultsDict_var  = {}

    eIdx = 0
    for estimator in miEstimators:
        resultsDict_mean[estimator] = np.empty(MAX_NUM_FEATURES)
        resultsDict_var[estimator]  = np.empty(MAX_NUM_FEATURES)
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
                
                # resultsMean[eIdx,ii-1] = mu
                # resultsVar[eIdx,ii-1] = sigma_sq
                resultsDict_mean[estimator][ii-1] = mu
                resultsDict_var[estimator][ii-1]  = sigma_sq

        eIdx = eIdx + 1

    return (resultsDict_mean,resultsDict_var)

def generate_skew_comparison_plots(algorithm='RandomForest'):
    datasets_to_plot = ['Arcene','Dexter','Madelon','Gisette',]
                      # 'arcene_0.25','dexter_0.25','madelon_0.25','gisette_0.25',
                      # 'arcene_0.50','dexter_0.50','madelon_0.50','gisette_0.50']
    # markers = []
    # for m in Line2D.markers:
    #     try:
    #         if len(m) == 1 and m != ' ':
    #             markers.append(m)
    #     except TypeError:
    #         pass
    markers = ['o','v','^','*','+','p','h']

    # run the ML
    for datasetIdx in range(len(datasetsToTest)):
        print('*'*10 + ' ' + datasetsToTest[datasetIdx] + ' ' + '*'*10)        
        datasetToTest = datasetsToTest[datasetIdx]
        resultsDir = os.path.join(getDataFolder(datasetToTest), 'classification_results')
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
            
        # store these results
        dataDict = {}
        dataDict['resultsMean'] = resultsMean
        dataDict['resultsVar']  = resultsVar
        with open(fname,'wb') as f:
            pickle.dump(dataDict,f)

    # plot the stuff & store
    estimatorsLegend = map(lambda x:x.upper(),miEstimators_toPlot)
    try:
        estimatorsLegend[estimatorsLegend.index('TAUKL')]  = r'$\tau_{KL}$'
    except:
        pass
    if('tau' in miEstimators):
        estimatorsLegend[estimatorsLegend.index('TAU')]  = r'$\tau$'
    if('VME' in miEstimators):
        estimatorsLegend[estimatorsLegend.index('VME')]    = 'vME'
    if('KNN_1' in miEstimators):
        estimatorsLegend[estimatorsLegend.index('KNN_1')]  = r'$KNN_1$'
    if('KNN_6' in miEstimators):
        estimatorsLegend[estimatorsLegend.index('KNN_6')]  = r'$KNN_6$'
    if('KNN_20' in miEstimators):
        estimatorsLegend[estimatorsLegend.index('KNN_20')] = r'$KNN_{20}$'

    datasets_to_plot = ['Arcene','Dexter','Madelon','Gisette']
    skews = [0.5,0.75,'full']
    for dataset in datasets_to_plot:
        resultsDir = os.path.join(getDataFolder(dataset),'classification_results')
        outputFname = os.path.join(figures_folder,'by_skew',dataset+'.png')
        
        fig,ax = plt.subplots(1,3,sharex=True,sharey=True,figsize=(9,3))

        yMinVal = 1.0
        yMaxVal = 0.0
        for cIdx in range(len(skews)):
            marker = itertools.cycle(markers) 
            skewVal = skews[cIdx]
            if(skewVal=='full'):
                ds_str = dataset
            else:
                ds_str = '%s_%0.02f' % (dataset.lower(),skewVal)
            f = os.path.join(resultsDir,ds_str+'_'+algorithm+'.pkl')
            with open(f,'rb') as f:
                z = pickle.load(f)

            # load the data for the inset plot also!
            if(skewVal=='full'):
                ds_lower = dataset.lower()
                z_data = sio.loadmat(os.path.join(getDataFolder(dataset),ds_lower,'data.mat'))
            else:
                ds_lower = dataset.lower()
                z_data = sio.loadmat(os.path.join(getDataFolder(dataset),ds_lower,'data_skew_'+'%0.02f'%(skewVal)+'.mat'))
            y_train = z_data['y_train']
            y_train = np.squeeze(np.asarray(y_train))
            unique_elements, counts_elements = np.unique(y_train, return_counts=True)
            num_neg_one = counts_elements[0]
            num_pos_one = counts_elements[1]

            lineHandlesVec = []
            # for estimatorIdx in range(z['resultsMean'].shape[0]):
            for estimator in miEstimators_toPlot:
                resultsMean = z['resultsMean'][estimator]
                results2Var = z['resultsVar'][estimator]
                resultsStd = np.sqrt(results2Var/2.)
                xx = range(1,len(resultsMean)+1)

                y = resultsMean
                h = ax[cIdx].plot(xx, y, marker = marker.next(), linestyle='dashed', linewidth=1, markersize=6, fillstyle='none')

                yLo = resultsMean-results2Var/2.
                yHi = resultsMean+results2Var/2.
                ax[cIdx].fill_between(xx, yLo, yHi, alpha=0.2)
                ax[cIdx].grid(True)
                ax[cIdx].set_xticks([10,30,50])
                lineHandlesVec.append(h[0])
                
                if(min(yLo)<yMinVal):
                    yMinVal = min(yLo)
                if(max(yHi)>yMaxVal):
                    yMaxVal = max(yHi)
            if(skewVal=='full'):
                ax[cIdx].set_title('No Skew')
            else:
                ax[cIdx].set_title('Skew=%0.02f' % (skewVal))
            if(cIdx==0):
                ax[cIdx].set_ylabel(dataset.upper()+'\nClassification Accuracy')
            if(cIdx==1):
                ax[cIdx].set_xlabel('# Features')

            # add an inset plot to show the data skew of output class labels via histogram
            pos = ax[cIdx].get_position().bounds
            a = plt.axes([pos[0],pos[1],0.05,0.15])
            sns.barplot([-1,1],counts_elements,saturation=0.75)
            #plt.xticks([])
            a.xaxis.set_ticks_position('top')
            plt.yticks([])

        # because of a wide variance for KTAU w/ the first feature for Dorothea, 
        # we have to manually set yMin and yMax, otherwise plot is uninformative
        if(dataset=='Dorothea'):
            ax[cIdx].set_ylim(0.8,1.0)
                
        plt.figlegend( lineHandlesVec, estimatorsLegend, loc = 'center right' )
        plt.savefig(outputFname, bbox_inches='tight')


def generate_alg_comparsion_plots():
    # NOTE: NEED TO UPDATE!!!!!!!!!!!!!!!!
    # run the ML
    for datasetIdx in range(len(datasetsToTest)):
        print('*'*10 + ' ' + datasetsToTest[datasetIdx] + ' ' + '*'*10)        
        datasetToTest = datasetsToTest[datasetIdx]
        resultsDir = os.path.join(getDataFolder(datasetToTest), 'classification_results')
        try:
            os.makedirs(resultsDir)
        except:
            pass
        for classifierStr in classifiersToTest:
            fname = os.path.join(resultsDir,datasetToTest+'_'+classifierStr+'.pkl')
            if os.path.exists(fname):
                with open(fname,'rb') as f:
                    dataDict = pickle.load(f)
                resultsMean = dataDict['resultsMean']
                resultsVar  = dataDict['resultsVar']
            else:
                # only run if the results don't already exist
                resultsMean, resultsVar = evaluateClassificationPerformance(classifierStr,datasetToTest)
                
            # store these results
            dataDict = {}
            dataDict['resultsMean'] = resultsMean
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
    if('VME' in miEstimators):
        estimatorsLegend[estimatorsLegend.index('VME')]    = 'vME'
    if('KNN_1' in miEstimators):
        estimatorsLegend[estimatorsLegend.index('KNN_1')]  = r'$KNN_1$'
    if('KNN_6' in miEstimators):
        estimatorsLegend[estimatorsLegend.index('KNN_6')]  = r'$KNN_6$'
    if('KNN_20' in miEstimators):
        estimatorsLegend[estimatorsLegend.index('KNN_20')] = r'$KNN_{20}$'

    for dataset in datasetsToTest:
        resultsDir = os.path.join(getDataFolder(dataset),'classification_results')
        outputFname = os.path.join(figures_folder,dataset+'.png')
        
        fig,ax = plt.subplots(1,3,sharex=True,sharey=True,figsize=(9,3))

        yMinVal = 1.0
        yMaxVal = 0.0
        for cIdx in range(len(classifiersToTest)):
            classifier = classifiersToTest[cIdx]
            f = os.path.join(resultsDir,dataset+'_'+classifier+'.pkl')
            with open(f,'rb') as f:
                z = pickle.load(f)

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
                ax[cIdx].set_xticks([10,30,50])
                lineHandlesVec.append(h[0])
                
                if(min(yLo)<yMinVal):
                    yMinVal = min(yLo)
                if(max(yHi)>yMaxVal):
                    yMaxVal = max(yHi)
            ax[cIdx].set_title(classifier)
            if(cIdx==0):
                ax[cIdx].set_ylabel(dataset.upper()+'\nClassification Accuracy')
            if(cIdx==1):
                ax[cIdx].set_xlabel('# Features')
        # because of a wide variance for KTAU w/ the first feature for Dorothea, 
        # we have to manually set yMin and yMax, otherwise plot is uninformative
        if(dataset=='Dorothea'):
            ax[cIdx].set_ylim(0.8,1.0)
                
        plt.figlegend( lineHandlesVec, estimatorsLegend, loc = 'center right' )
        plt.savefig(outputFname, bbox_inches='tight')

if __name__=='__main__':
    generate_skew_comparison_plots(algorithm='KNN')
