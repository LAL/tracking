import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import DBSCAN


from Fitting import *





def polar(hits, rscale = 1):
    ptsnew = np.zeros(hits.shape)
    xy = hits[:,0]**2 + hits[:,1]**2
    ptsnew[:,0] = np.sqrt(xy)*rscale
    ptsnew[:,1] = np.arctan2(hits[:,1], hits[:,0])
    return ptsnew


class Clusterer(BaseEstimator):
    def __init__(self, eps=0.01, rscale=0.0001):
        self.eps = eps
        self.rscale = rscale
        self.min_hits = 3
        self.cls = DBSCAN(eps=self.eps, min_samples=self.min_hits)
    
    def fit(self, X, y):
        X = X[:,1:5] # drop event
        #X = X[:,0:2] # use layer, iphi
        X = X[:,2:4] # use x,y
        X = polar(X, self.rscale)
    #        self.cls.fit(X,y)
    
    def predict_single_event(self, X_event):
        #X_event = X_event[:,0:2]
        X_event = X_event[:,2:4]
        X_event = polar(X_event, self.rscale)
        y_event = self.cls.fit_predict(X_event)

# get cluster

# get 3 hits

# fit hits

# go to layer

# calculate predicted positions

# match hits to positions



        return y_event


