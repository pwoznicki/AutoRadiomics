#!/usr/bin/env python

'''
from: https://medium.com/@literallywords/sklearn-identity-transformer-fcc18bac0e98 
'''

from sklearn.base import BaseEstimator, TransformerMixin

class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, input_array, y=None):
        return self
    
    def transform(self, input_array, y=None):
        return input_array*1
