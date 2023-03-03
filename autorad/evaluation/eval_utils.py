import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve


def get_youden_threshold(y_true, y_score) -> float:
    """
    Returns optimal threshold that maximizes the
    difference between TPR and FPR
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youden_index = [tpr[i] - fpr[i] for i in range(len(tpr))]
    youden_argmax = np.argmax(youden_index)
    you_thr = thresholds[youden_argmax]

    return you_thr


def get_optimal_threshold(y_test, y_score):
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    i = np.arange(len(tpr))
    roc = pd.DataFrame(
        {
            "tf": pd.Series(tpr - (1 - fpr), index=i),
            "threshold": pd.Series(thresholds, index=i),
        }
    )
    roc_t = roc.ix[(roc.tf - 0).abs().argsort()[:1]]
    return list(roc_t["threshold"])


def get_sensitivity_specificity(y_test, y_pred_proba, threshold):
    y_pred = np.where(y_pred_proba > threshold, 1, 0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    sensitivity = np.round(sensitivity, 3)
    specificity = np.round(specificity, 3)
    return sensitivity, specificity


def get_fpr_tpr_auc(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = np.round(auc(fpr, tpr), 3)
    return fpr, tpr, roc_auc
