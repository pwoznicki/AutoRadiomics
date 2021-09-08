#!/usr/bin/env python

'''
OLD DEPRECATED
'''

def readArrhythmiaData():
    z = sio.loadmat(os.path.join(folder,'arrhythmia','X.mat'))
    X = z['X']
    y = z['y']

    miFeatureSelections = {}
    for miEstimator in miEstimators:
        featureVec = sio.loadmat(os.path.join(folder,'arrhythmia','arrhythmia_fs_'+miEstimator+'.mat'))
        miFeatureSelections[miEstimator] = featureVec['featureVec']
    
    return (X,y,miFeatureSelections)

def readRnaSeqData():
    z = sio.loadmat(os.path.join(folder,'rnaseq','X.mat'))
    X = z['X']
    y = z['y']

    miFeatureSelections = {}
    for miEstimator in miEstimators:
        featureVec = sio.loadmat(os.path.join(folder,'rnasesq','rnaseq_fs_'+miEstimator+'.mat'))
        miFeatureSelections[miEstimator] = featureVec['featureVec']
    
    return (X,y,miFeatureSelections)

'''
if(dataset=='Arrhythmia'):
    (X,y,miFeatureSelections) = readArrhythmiaData()
    y = np.squeeze(np.asarray(y))
    numCV_val = numCV
elif(dataset=='RnaSeq'):
    (X,y,miFeatureSelections) = readRnaSeqData()
    y = np.squeeze(np.asarray(y))
    numCV_val = numCV
'''