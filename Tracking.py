
from sklearn.base import BaseEstimator
from sklearn.cluster import DBSCAN

import Transform as transform


class HitToTrackAssignmet(BaseEstimator):
    def __init__(self, eps = 0.02):
        self.eps = eps
        self.min_hits = 3
        self.cls = DBSCAN(eps=self.eps, min_samples=self.min_hits)
    
    def fit(self, X, y):
        self.cls.fit(X,y)

    def predict(self, X):
        return self.cls.fit_predict(X)

