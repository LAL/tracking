import numpy as np

from sklearn.base import BaseEstimator
from sklearn.cluster import DBSCAN

import Transform as transform


class ClusterDBSCAN(BaseEstimator):
    def __init__(self, eps=0.02, rscale=0.001):
        self.eps = eps
        self.rscale = rscale
        self.min_hits = 3
        self.cls = DBSCAN(eps=self.eps, min_samples=self.min_hits)
    
    def fit(self, X, y):
        X = transform.polar(X, self.rscale)
        self.cls.fit(X,y)

    def predict(self, X):
        y = np.array([])
        events=np.unique(X[:,0])
        for ievent in events:
            X_event = X[X[:,0] == ievent]
            X_event = transform.polar(X_event, self.rscale)
            y_event = self.cls.fit_predict(X_event)
            y = np.append(y,y_event)
        return y

