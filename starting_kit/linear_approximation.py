__author__ = 'mikhail91'


import numpy as np
from sklearn.base import BaseEstimator


class Clusterer(BaseEstimator):


    def __init__(self, min_hits=4, window_width=0.03):
        """
        Track Pattern Recognition based on the linear approximation.

        Parameters
        ----------
        min_hits : int
            Minimum number of hits of one track.
        window_width : float
            Phi window width.
        """

        self.min_hits = min_hits
        self.window_width = window_width

    def get_phi(self, x, y):
        """
        Calculate hits phi coordinates in polar system.

        Parameters
        ----------
        x : array-like
            X-coordinates of hits.
        y : array-like
            Y-coordinates of hits.

        Returns
        -------
        phi : array-like
            Phi coordinates of hits.
        """

        x = np.array(x)
        y = np.array(y)

        phi = np.arctan(y / x) * (x != 0) + np.pi * (x < 0) + 0.5 * np.pi * (x==0) * (y>0) + 1.5 * np.pi * (x==0) * (y<0)

        return phi

    def fit(self, X, y):
        pass


    def predict_one_event(self, X):
        """
        Track Pattern Recognition for one event.

        Parameters
        ----------
        X : ndarray-like
            Hit features.

        Returns
        -------
        labels : array-like
            Recognized track labels.
        """

        x, y, layer = X[:, 2], X[:, 3], X[:, 0]
        used = np.zeros(len(x))
        labels = -1. * np.ones(len(x))
        track_id = 0

        # calculate phi of the hits
        phis = self.get_phi(x, y)

        # Go through each hit
        for first_id in np.arange(0, len(x)):

            x1 = x[first_id]
            y1 = y[first_id]

            phi1 = self.get_phi(x1, y1)

            track_inds = np.arange(len(x))[(np.abs(phis - phi1) <= self.window_width)*(used == 0)]

            if len(track_inds) >= self.min_hits:
                used[track_inds] = 1
                labels[track_inds] = track_id
                track_id += 1

        return labels

    def predict(self, X):
        """
        Track Pattern Recognition for all event.

        Parameters
        ----------
        X : ndarray-like
            Hit features.

        Returns
        -------
        labels : array-like
            Recognized track labels.
        """


        unique_event_ids = np.unique(X[:, 0])
        labels = np.empty(len(X), dtype='int')
            
        for event_id in unique_event_ids:
            event_indices = (X[:, 0] == event_id)
            # select an event and drop event ids
            X_event = X[event_indices][:, 1:]
            labels[event_indices] = self.predict_one_event(X_event)

        return np.array(labels)
