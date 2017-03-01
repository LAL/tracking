
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import DBSCAN

class HitToTrackAssignment(BaseEstimator):
    def __init__(self, eps = 0.02):
        self.eps = eps
        self.min_hits = 3
        self.cls = DBSCAN(eps=self.eps, min_samples=self.min_hits)
    
    def fit(self, X, y):
        self.cls.fit(X,y[:,1])

    def predict(self, X):
        y = np.zeros((0,2))
        events=np.unique(X[:,0])
        for ievent in events:
            X_event = X[X[:,0] == ievent]
            y_event = np.zeros((len(X_event),2))
            y_event = self.cls.fit_predict(X_event)
            y = np.append(y,y_event)
        return y

