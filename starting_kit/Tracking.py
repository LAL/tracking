
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import DBSCAN

coleventX = 4

class HitToTrackAssignment(BaseEstimator):
    def __init__(self, eps = 0.02):
        self.eps = eps
        self.min_hits = 3
        self.cls = DBSCAN(eps=self.eps, min_samples=self.min_hits)
    
    def fit(self, X, y):
        # The columns of the numpy array X (input) are
        # 0 : event

        # The columns of the numpy array y (target) are
        # 0 : event


        self.cls.fit(X,y[:,1])

    def predict(self, X):
        y = np.zeros((0,2))
        events=np.unique(X[:,coleventX])
        for ievent in events:
            X_event = X[X[:,coleventX] == ievent]
            y_event = np.zeros((len(X_event),2))
            y_event[:,0] = self.cls.fit_predict(X_event)
            y_event[:,1] = [ievent] * len(X_event)
            y = np.append(y,y_event,axis=0)
        return y

