from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from statsmodels.stats.contingency_tables import mcnemar

log = logging.getLogger(__name__)

def compare_sensitivity_mcnemar(y_pred_proba_1, y_pred_proba_2):
    """
    Compare sensitivity of two models using McNemar's test
    """
    contingency_table = pd.crosstab(
        index=y_pred_proba_1, columns=y_pred_proba_2
    )
    _, p = mcnemar(contingency_table)
    return p


def bootstrap_auc(y_true, y_pred_proba):
    """
    Get AUC and 95% Confidence Interval from bootstrapping.
    """
    sample_statistic, lower, upper = bootstrap_statistic(
        roc_auc_score,
        y_true,
        y_pred_proba,
    )

    return sample_statistic, lower, upper


def bootstrap_statistic(statistic: Callable, x, y, num_folds=1000):
    """
    Bootstrap statistic for comparing two groups.
    Args:
        statistic: function that takes two lists of values and returns a
            statistic.
        x: list of values for group 1
        y: list of values for group 2
        num_folds: number of bootstrap samples to draw
    Returns:
        statistic: sample statistic for the two groups
        lower_bound: lower bound of the 95% confidence interval
        upper_bound: upper bound of the 95% confidence interval
    """
    stats = []
    for i in range(num_folds):
        boot_x, boot_y = resample(
            x, y, replace=True, n_samples=len(x), random_state=i
        )
        stat = statistic(boot_x, boot_y)
        stats.append(stat)
    stats_arr = np.array(stats)
    sample_statistic = statistic(x, y)
    lower_bound = np.percentile(stats_arr, 2.5)
    upper_bound = np.percentile(stats_arr, 97.5)

    return sample_statistic, lower_bound, upper_bound
