import logging
from functools import wraps
from typing import Callable, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from statsmodels.stats.contingency_tables import mcnemar

log = logging.getLogger(__name__)


def round_up_p(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        p = f(*args, **kwargs)
        rounded_p = np.round(p, 3)
        return rounded_p

    return wrapper


@round_up_p
def compare_groups_not_normally_distributed(
    x: list[float], y: list[float], alternative="two-sided"
):
    """
    Mann-Whitney test (= unpaired Wilcoxon test).
    """
    _, p = stats.ranksums(x, y, alternative=alternative)
    return p


@round_up_p
def compare_age_between_groups(x: list[float], y: list[float]) -> float:
    """
    Perform Welsh's t-test (good when cohorts differ in size,
    because doesn't assume equal variance).
    """
    if not x or not y:
        raise ValueError("x and y must be non-empty lists of strings")
    if any(elem < 0 for elem in (x + y)):
        raise ValueError("Age cannot be negative.")
    _, p = stats.ttest_ind(x, y, equal_var=False)
    return p


@round_up_p
def compare_gender_between_groups(
    genders: list[Union[int, str]], groups: list[Union[int, str]]
) -> int:
    """
    Performs Chi square test for independence.
    Tests if observed frequencies are independent of the expected
    frequencies.
    To be used for categorical variables,e.g. the gender distributions.
    """
    contingency_matrix = pd.crosstab(index=genders, columns=groups)
    _, p, _, _ = stats.chi2_contingency(contingency_matrix)
    return p


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
