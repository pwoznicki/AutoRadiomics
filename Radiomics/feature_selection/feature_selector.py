from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, method, k):
        self.method = method
        self.k = k
        
    def fit(self, X, y):
        if self.method == 'Select K Best':
            self.selector = SelectKBest(f_classif, k=self.k)
        else:
            raise ValueError('Invalid method')
        self.selector.fit(X, y)
        return self
    
    def transform(self, X):
        return self.selector.transform(X)
    
    def inverse_transform(self, X):
        return self.selector.inverse_transform(X)