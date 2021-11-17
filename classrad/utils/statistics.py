import numpy as np
from imblearn.metrics import sensitivity_specificity_support
from scipy import stats
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.utils import resample


def wilcoxon_unpaired(x, y, alternative="two-sided"):
    """
    Test for differences between unpaired samples.
    """
    stat, p = stats.ranksums(x, y, alternative=alternative)
    if p < 0.001:
        p_string = "< 0.001"
    else:
        p_string = str(np.round(p, 3))
    return stat, p_string


def get_sens_spec(y_true, y_pred):
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
    cm = confusion_matrix(y_true, y_pred)

    print(sens, spec)
    print(cm)

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
    Q2 = 2 * AUC ** 2 / (1 + AUC)
    SE_AUC = np.sqrt(
        (AUC * (1 - AUC) + (N1 - 1) * (Q1 - AUC ** 2) + (N2 - 1) * (Q2 - AUC ** 2))
        / (N1 * N2)
    )
    lower = AUC - 1.96 * SE_AUC
    upper = AUC + 1.96 * SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1

    return AUC, lower, upper
