#!/usr/bin/env python

import numpy as np
from scipy.stats import kendalltau

def abs_tau(x,y):
    tau_val,p_val = kendalltau(x,y)
    if(np.isnan(tau_val)):
        return 0
    return np.abs(tau_val)

def mi_tau(x,y):
    tau_val,p_val = kendalltau(x,y)
    if(np.isnan(tau_val)):
        return 0
    return dep2mi(tau_val)
    
def dep2mi(x):
    MAX_VAL = 0.999999;
    if(x>MAX_VAL):
        x = MAX_VAL
    elif(x<-MAX_VAL):
        x = -MAX_VAL
    
    y = -0.5*np.log(1-x*x);

    return y