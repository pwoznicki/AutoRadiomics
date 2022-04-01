import logging
from functools import wraps
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from imblearn.metrics import sensitivity_specificity_support
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
    x: List[float], y: List[float], alternative="two-sided"
):
    """
    Mann-Whitney test (= unpaired Wilcoxon test).
    """
    _, p = stats.ranksums(x, y, alternative=alternative)
    return p


@round_up_p
def compare_age_between_groups(x: List[float], y: List[float]) -> float:
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
    genders: List[Union[int, str]], groups: List[Union[int, str]]
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


def get_sens_spec(y_true: List[int], y_pred: List[int]) -> Tuple[float, float]:
    """
    Args:
        y_true: list of binary ground-truth labels
        y_pred: list of binary predictions
    Returns:
        sensitivity: True Positive Rate = Sensitivity
        specificity: True Negative Rate = Specificity (TNR = 1-FPR)
    """
    stat_table = sensitivity_specificity_support(y_true, y_pred)
    sensitivity = stat_table[1][0]
    specificity = stat_table[1][1]
    return sensitivity, specificity


def describe_auc(y_true, y_pred_proba):
    """
    Get mean AUC and 95% Confidence Interval from bootstrapping.
    """
    AUCs = []
    num_folds = 1000
    for i in range(num_folds):
        boot_y_pred, boot_y_true = resample(
            y_pred_proba,
            y_true,
            replace=True,
            n_samples=len(y_true),
            random_state=i + 10,
        )
        AUC = roc_auc_score(boot_y_true, boot_y_pred)
        AUCs.append(AUC)

    mean_AUC = np.mean(AUCs)
    lower_CI_bound = np.percentile(AUCs, 2.5)
    upper_CI_bound = np.percentile(AUCs, 97.5)
    print(f"Results from {num_folds} folds of bootrapping:")
    print(f"Mean AUC={mean_AUC}, 95% CI [{lower_CI_bound}, {upper_CI_bound}]")

    return mean_AUC, lower_CI_bound, upper_CI_bound


def describe_stat(y_true, y_pred):
    """
    Get sensitivity and specificity with corresponding CIs using bootstrapping.
    Args:
        y_true: list of binary ground-truth labels
        y_pred:
    """
    sens, spec = get_sens_spec(y_true, y_pred)

    sens_all, spec_all = [], []
    num_folds = 1000
    for i in range(num_folds):
        boot_y_pred, boot_y_true = resample(
            y_pred, y_true, replace=True, n_samples=len(y_true), random_state=i
        )
        sens, spec = get_sens_spec(boot_y_true, boot_y_pred)
        sens_all.append(sens)
        spec_all.append(spec)
    sens_all, spec_all = np.array(sens_all), np.array(spec_all)

    mean_sens = np.mean(sens_all)
    lower_CI_bound_sens = np.percentile(sens_all, 2.5)
    upper_CI_bound_sens = np.percentile(sens_all, 97.5)

    mean_spec = np.mean(spec_all)
    lower_CI_bound_spec = np.percentile(spec_all, 2.5)
    upper_CI_bound_spec = np.percentile(spec_all, 97.5)

    print(f"Results from {num_folds} folds of bootrapping:")
    print(
        f"Sensitivity: Mean={mean_sens}, 95% CI [{lower_CI_bound_sens}, \
          {upper_CI_bound_sens}]"
    )
    print(
        f"Specificity: Mean={mean_spec}, 95% CI [{lower_CI_bound_spec}, \
          {upper_CI_bound_spec}]"
    )

    return (mean_sens, lower_CI_bound_sens, upper_CI_bound_sens), (
        mean_spec,
        lower_CI_bound_spec,
        upper_CI_bound_spec,
    )


# Review, not a particularly nice implementation
def roc_auc_ci(y_true, y_pred_proba, positive=1):
    """
    Get 95% Confidence Interval (CI) for AUC, assuming normal distribution.
    Args:
        y_true: list of binary ground-truth labels
        y_pred_proba: list of probabilities of positive class
        positive: class considered as positive
    Returns:
        AUC: AUC score
        lower: lower bound of 95% CI
        upper: upper bound of 95% CI
    """
    AUC = roc_auc_score(y_true, y_pred_proba)
    N1 = np.sum(y_true == positive)
    N2 = np.sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2 * AUC**2 / (1 + AUC)
    SE_AUC = np.sqrt(
        (
            AUC * (1 - AUC)
            + (N1 - 1) * (Q1 - AUC**2)
            + (N2 - 1) * (Q2 - AUC**2)
        )
        / (N1 * N2)
    )
    lower = AUC - 1.96 * SE_AUC
    upper = AUC + 1.96 * SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1

    return AUC, lower, upper
