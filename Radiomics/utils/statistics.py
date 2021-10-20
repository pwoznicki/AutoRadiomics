from re import I
import numpy as np
import numpy as np
from scipy import stats
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, roc_auc_score
from imblearn.metrics import sensitivity_specificity_support as support


def wilcoxon_unpaired(x, y, alternative="two-sided"):
    stat, p = stats.ranksums(x, y)
    if p < 0.001:
        p_string = "< 0.001"
    else:
        p_string = str(np.round(p, 3))
    return stat, p_string


def get_stat(label, pred):
    """returns sensitivity (=TPR) and specificity (=TNR = 1-FPR)"""
    stat_table = support(label, pred)
    tpr = stat_table[1][0]
    tnr = stat_table[1][1]

    return tpr, tnr


def describe_auc(labels, preds_proba):
    AUCs = []
    for i in range(2):
        boot_preds, boot_labels = resample(
            preds_proba,
            labels,
            replace=True,
            n_samples=len(labels),
            random_state=i + 10,
        )
        AUC = roc_auc_score(boot_labels, boot_preds)
        AUCs.append(AUC)

    print(np.mean(AUCs), np.percentile(AUCs, 2.5), np.percentile(AUCs, 97.5))


def describe_stat(labels, preds):
    """prints out sensitivity and specificity with corresponding CIs"""
    sens, spec = get_stat(labels, preds)
    cm = confusion_matrix(labels, preds)

    print(sens, spec)
    print(cm)

    sens_all, spec_all = [], []
    for i in range(1000):
        boot_preds, boot_labels = resample(
            preds, labels, replace=True, n_samples=len(labels), random_state=i
        )
        sens, spec = get_stat(boot_labels, boot_preds)
        sens_all.append(sens)
        spec_all.append(spec)
    sens_all, spec_all = np.array(sens_all), np.array(spec_all)

    print(
        np.mean(sens_all), np.percentile(sens_all, 2.5), np.percentile(sens_all, 97.5)
    )
    print(
        np.mean(spec_all), np.percentile(spec_all, 2.5), np.percentile(spec_all, 97.5)
    )


def roc_auc_ci(y_true, y_score, positive=1):
    AUC = roc_auc_score(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
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
    return (lower, upper)
