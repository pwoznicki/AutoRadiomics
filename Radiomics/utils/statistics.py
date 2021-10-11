import numpy as np
from scipy import stats

def wilcoxon_unpaired(x, y, alternative='two-sided'):
    stat, p = stats.ranksums(x, y)
    if p < 0.001:
        p_string = '< 0.001'
    else:
        p_string = str(np.round(p, 3))
    return stat, p_string