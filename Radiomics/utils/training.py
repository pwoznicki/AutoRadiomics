import os
from os.path import join, exists, basename
import numpy as np
import pandas as pd
from radiomics import featureextractor
import collections
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import ADASYN

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import matplotlib.pyplot as plt

import feature_select

TABLE_DIR = '../tables'
RSEED = 9


def tune_random_forest(x_train, y_train):
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf = RandomForestClassifier(n_estimators=1200, 
                                random_state=RSEED,
                                max_features='auto',
                                max_depth=None,
                                min_samples_split=10,
                                min_samples_leaf=2,
                                bootstrap=False,
                                n_jobs=-1,
                                verbose=0)
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                   n_iter=400, cv=4, verbose=2,
                                   random_state=42, n_jobs=-1)
    rf_random.fit(x_train, y_train)
    print(rf_random.best_params_)


def make_random_forest_model():
    model = RandomForestClassifier(n_estimators=1600, 
                                   random_state=RSEED,
                                   max_features='auto',
                                   max_depth=None,
                                   min_samples_split=10,
                                   min_samples_leaf=2,
                                   bootstrap=False,
                                   n_jobs=-1, verbose=0)
    return model


def tune_xgboost(x, y):
    random_grid = {"learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                   "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
                   "min_child_weight": [1, 3, 5, 7],
                   "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
                   "colsample_bytree": [0.3, 0.4, 0.5, 0.7]}

    xgb = XGBClassifier()
    xgb_random = RandomizedSearchCV(estimator=xgb, param_distributions=random_grid, 
                                    n_iter=2000, cv=4, verbose=2, random_state=42, n_jobs=-1)
    xgb_random.fit(x, y)

    print(xgb_random.best_params_)


def make_xgboost_model():
    model = XGBClassifier(min_child_weight=3, 
                          max_depth=6,
                          learning_rate=0.05,
                          gamma=0.1,
                          colsample_bytree=0.3)
    return model


def tune_svm(x, y, nfolds):
    cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    kernels = ['rbf']
    param_grid = {'kernel': kernels, 'C': cs, 'gamma': gammas}
    grid_search = GridSearchCV(SVC(), param_grid, cv=nfolds, verbose=2)
    grid_search.fit(x, y)
    print(grid_search.best_params_)

    return grid_search.best_params_


def tune_logreg(x, y):
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
                  "penalty": ['l1', 'l2', 'elasticnet', 'none']}
    lr = LogisticRegression(penalty='l2')
    lr_random = GridSearchCV(lr, param_grid)
    lr_random.fit(x, y)
    
    print(lr_random.best_params_)


def extract_features(label_column, data_dir, save_dir, sequences=['ADC', 'T2']):
    df = pd.read_csv(join(TABLE_DIR, 'common_df.csv'))
    df = df[df.ADC_img == 1][df.ADC_lesion == 1][df.source == 'mannheim']
    if label_column == 'Gleason_significant':
        df = df[df.PCA == 1]
    print(len(df))
        
    for seq in sequences:   
        if seq == 'T2':
            param_path = '../parameters/parameter_bettina.yaml'
        else:
            param_path = '../parameters/parameter_piotr.yaml'
            
        extractor = featureextractor.RadiomicsFeatureExtractor(param_path)
        feature_vector_list = []
        
        for idx, entry in tqdm(df.iterrows()):
            id_ = str(entry['pat_id'])
            image_path = join(data_dir, seq, id_, 'img.nii')
            mask_path = join(data_dir, seq, id_, 'lesion.nii')
            label = int(entry[label_column])
            feature_vector = collections.OrderedDict(entry)

            if exists(image_path) and exists(mask_path):
                feature_vector['Image'] = basename(image_path)
                feature_vector['Mask'] = basename(mask_path)
                feature_vector['Label'] = label
                feature_vector.update(extractor.execute(image_path, mask_path))
                feature_vector_list.append(feature_vector)

        feature_df = pd.DataFrame(feature_vector_list, columns=feature_vector.keys())
        os.makedirs(save_dir, exist_ok=True)
        save_path = join(save_dir, seq + '_radiomic_features.csv')
        feature_df.to_csv(save_path, index=False)


def get_columns_to_drop_for_radiomics(leave_pat_id=False):
    # for adc/t2 merging leave out pat_id
    cols = ['source', 'PCA', 'Gleason', 'Gleason nach PE', 'Graduierung',
            'PI-RADS Target', 'PSA (ng/ml)', 'TN-Ausdehnung', 'Läsionen (n)',
            'lesion_location (1-tz, 2-pz)', 'Lesion1_Location', 'Lesion1_PIRADS',
            'Lesion2_Location', 'Lesion2_PIRADS', 'TURP', 'Segmentierungen (n)',
            'radikale Prostatektomie (0-nein, 1-ja)', 'T2_img',
            'T2_whole_prostate', 'T2_lesion', 'T2_peripheral_zone',
            'T2_transition_zone', 'ADC_img', 'ADC_whole_prostate', 'ADC_lesion',
            'birth_date', 'exam_date', 'vol_lesion (cm3)',
            'vol_whole_prostate (cm3)', 'vol_peripheral_zone (cm3)',
            'vol_transition_zone (cm3)', 'Gleason_sum', 'Gleason_significant', 'Image', 'Mask', 'volume_pred',
            'pixelnum_pred', 'dice', 'dilated_dice', 'mean_ADC_pred', 'test', 'Gleason_significant_fold']
    if not leave_pat_id:
        cols.append('pat_id')

    return cols


def merge_sequences(feature_df_dir):
    adc_rf = pd.read_csv(join(feature_df_dir, 'ADC_radiomic_features.csv'))
    t2_rf = pd.read_csv(join(feature_df_dir, 'T2_radiomic_features.csv'))

    adc_rf.drop(columns=get_columns_to_drop_for_radiomics(leave_pat_id=True), inplace=True)
    adc_rf.drop(columns=['Label'], inplace=True)
    adc_col_names = {name: 'adc_' + name for name in adc_rf.columns}
    adc_col_names['pat_id'] = 'pat_id'
    adc_rf = adc_rf.rename(columns=adc_col_names)

    rf = pd.merge(t2_rf, adc_rf, on='pat_id')
    rf.to_csv(join(feature_df_dir, 'common_radiomic_features.csv'), index=False)


def preprocess_rf(rf, df):
    results = rf[rf.source == 'mannheim'][['pat_id', 'PI-RADS Target', 'Label', 'fold',
                                           'Gleason_significant', 'test', 'lesion_location (1-tz, 2-pz)']].copy()
    results.rename(columns={"PCA": "Label"}, inplace=True)
    results['train_id'] = -1
    rf_ids = rf['pat_id'].values
    for id_ in rf_ids:
        results.loc[results.pat_id == id_, 'train_id'] = rf[rf.pat_id == id_].index.values[0]
    diag_cols = [col for col in rf.columns if 'diagnostics' in col]
    rf.drop(columns=diag_cols, inplace=True)
    rf.drop(columns=get_columns_to_drop_for_radiomics(), inplace=True)
    drop_cols = rf.columns[rf.dtypes == 'object']
    rf.drop(columns=drop_cols, inplace=True)
    rf.dropna(axis='columns', inplace=True)

    return rf, results


def train_eval_radiomic(rf_path, num_features='15', classifier='RandomForest', oversample=False):
    rf = pd.read_csv(rf_path)
    rf = rf[rf.source == 'mannheim']
    rf.rename(columns={'PCA_fold': 'fold'}, inplace=True)
    df = pd.read_csv(join(TABLE_DIR, 'common_df.csv'))
    # rf = rf[rf['lesion_location (1-tz, 2-pz)'] == 1]
    rf = rf.sample(frac=1, random_state=1).reset_index(drop=True)
    rf_train = rf[rf.test == 0]
    rf_test = rf[rf.test == 1]
    rf_train, results = preprocess_rf(rf_train, df)
    rf_test, _ = preprocess_rf(rf_test, df)

    auc = []
    results['pred'] = -1
    results['prob'] = -1

    for i in range(5):

        x_train = rf_train[rf_train['fold'] != i].drop(['Label', 'fold'], axis=1).values
        y_train = rf_train[rf_train['fold'] != i]['Label'].values

        num_features = int(num_features)
        key_features = feature_select.feature_select(x_train, y_train, num_features)

        short_cols = ['Label', 'fold']
        for j in range(num_features):
            short_cols.append(rf_train.drop(['Label', 'fold'], axis=1).columns[key_features[j]])
        print(short_cols)
        rf_short = rf_train[short_cols]

        x = rf_short.drop(['Label', 'fold'], axis=1).values
        y = rf_short['Label'].values

        x_train = rf_short[rf_short['fold'] != i].drop(['Label', 'fold'], axis=1).values
        y_train = rf_short[rf_short['fold'] != i]['Label'].values

        x_val = rf_short[rf_short['fold'] == i].drop(['Label', 'fold'], axis=1).values
        y_val = rf_short[rf_short['fold'] == i]['Label'].values

        if oversample is True:
            oversample_method = ADASYN()
            x_train, y_train = oversample_method.fit_resample(x_train, y_train)
        
        if classifier == 'RandomForest':
            model = make_random_forest_model()
        elif classifier == 'XGBoost':
            model = make_xgboost_model()
        elif classifier == 'SVM':
            model = SVC(kernel='rbf', gamma=0.000001, probability=True)
        elif classifier == 'LogReg':
            model = LogisticRegression(C=0.001, penalty='l2')
        else:
            raise ValueError('Error - no such classifier as ', classifier)
        model.fit(x_train, y_train)

        rf_predictions = model.predict(x_val)
        rf_probs = model.predict_proba(x_val)[:, 1]
        auc.append(roc_auc_score(y_val, rf_probs))
        results.loc[results.fold == i, 'pred'] = rf_predictions
        results.loc[results.fold == i, 'prob'] = rf_probs

    print(np.mean(auc), np.std(auc))

    rf_test_short = rf_test[short_cols]
    x_test = rf_test_short.drop(['Label', 'fold'], axis=1).values
    y_test = rf_test_short['Label'].values

    test_probs = model.predict_proba(x_test)[:, 1]
    auc_test = roc_auc_score(y_test, test_probs)
    print(auc_test)

    return results


# Plot formatting
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 18


def evaluate_model(df):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
    # baseline = {'roc': 0.5}

    plt.figure(figsize=(10, 8))
    plt.rcParams['font.size'] = 16
    
    for i in range(5):
        probs = df[df.fold == i]['prob']
        print(len(probs))
        test_labels = df[df.fold == i]['Label']

        results = {'roc': roc_auc_score(test_labels, probs)}
        
        model_fpr, model_tpr, _ = roc_curve(test_labels, probs)

        label_txt = 'Radiomic model fold '+str(i+1)+' (AUC=' + str(round(results['roc'], 2)) + ')'
        plt.plot(model_fpr, model_tpr, 'r', label=label_txt, linestyle=':', c=np.random.rand(3,))
        plt.legend()
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC for PCa - Mannheim ADC and T2 (top 20), TZ')
        # plt.show();
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    plt.plot(base_fpr, base_tpr, 'b', label='Baseline')

    flat_probs = df['prob']
    flat_test_labels = df['Label']
    
    results = {'roc': roc_auc_score(flat_test_labels, flat_probs)}
    model_fpr, model_tpr, _ = roc_curve(flat_test_labels, flat_probs)
    plt.plot(model_fpr, model_tpr, 'r', label='Mean ROC (AUC=' + str(round(results['roc'], 2))+')')
    plt.legend()

    plt.savefig('roc_auc_curve.png')
