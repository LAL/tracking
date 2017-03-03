__author__ = 'mikhail91'


import numpy as np
from sklearn.base import BaseEstimator

class Clusterer(BaseEstimator):

    def __init__(self, n_theta_bins=100, n_radius_bins=100, min_radius=1., min_hits=4):
        """
        Track pattern recognition for one event based on Hough Transform.

        Parameters
        ----------
        n_theta_bins : int
            Number of bins track theta parameter is divided into.
        n_radius_bins : int
            Number of bins track 1/r parameter is divided into.
        min_radius : float
            Minimum track radius which is taken into account.
        min_hits : int
            Minimum number of hits per on recognized track.
        """

        self.n_theta_bins = n_theta_bins
        self.n_radius_bins = n_radius_bins
        self.min_radius = min_radius
        self.min_hits = min_hits


    def tranform(self, X):
        """
        Hough Transformation and tracks pattern recognition.

        Parameters
        ----------
        X : array_like
            Hit features.

        Return
        ------
        matrix_hough : ndarray
            Hough Transform matrix of all hits of an event.
        track_inds : ndarray
            List of recognized tracks. Each track is a list of its hit indexes.
        track_params : ndarray
            List of track parameters.
        """

        x, y = X[:, 2], X[:, 3]
        # Transform cartesian coordinates to polar coordinates
        hit_phis = np.arctan(y / x) * (x != 0) + np.pi * (x < 0) + 0.5 * np.pi * (x==0) * (y>0) + 1.5 * np.pi * (x==0) * (y<0)
        hit_rs = np.sqrt(x**2 + y**2)

        # Set ranges of a track theta and 1/r
        track_thetas = np.linspace(0, 2 * np.pi, self.n_theta_bins)
        track_invrs = np.linspace(0, 1. / self.min_radius, self.n_radius_bins)

        # Init arrays for the results
        matrix_hough = np.zeros((len(track_thetas)+1, len(track_invrs)+1))
        track_inds = []
        track_params = []

        for num1, theta in enumerate(track_thetas):

            # Hough Transform for one hit
            invr = 2. * np.cos(hit_phis - theta) / hit_rs

            # Hough Transform digitization
            bin_inds = np.digitize(invr, track_invrs)
            unique, counts = np.unique(bin_inds, return_counts=True)

            # Count number of hits in each bin. Fill the results arrays.
            for num2, one in enumerate(unique):

                matrix_hough[num1, one] = counts[num2]

                if counts[num2] >= self.min_hits and one != 0 and one < len(track_invrs) and num1 !=0 and num1 < len(track_thetas):

                    track_inds.append(np.arange(len(bin_inds))[bin_inds == one])
                    track_params.append([track_thetas[num1], track_invrs[one]])

        track_inds = np.array(track_inds)
        track_params = np.array(track_params)

        return matrix_hough[:, 1:-1], track_inds, track_params


    def get_hit_labels(self, track_inds, n_hits):
        """
        Estimate hit labels based on the recognized tracks.

        Parameters
        ----------
        track_inds : ndarray
            List of recognized tracks. Each track is a list of its hit indexes.
        n_hits : int
            Number of hits in the event.

        Return
        ------
        labels : array-like
            Hit labels.
        """

        labels = -1. * np.ones(n_hits)
        used = np.zeros(n_hits)
        track_id = 0


        while 1:

            track_lens = np.array([len(i[used[i] == 0]) for i in track_inds])

            if len(track_lens) == 0:
                break

            max_len = track_lens.max()

            if max_len < self.min_hits:
                break

            one_track_inds = track_inds[track_lens == track_lens.max()][0]
            one_track_inds = one_track_inds[used[one_track_inds] == 0]

            used[one_track_inds] = 1
            labels[one_track_inds] = track_id
            track_id += 1

        return np.array(labels)

    def fit(self, X, y):
        pass

    def predict_one_event(self, X):
        """
        Hough Transformation and tracks pattern recognition for one event.

        Parameters
        ----------
        X : ndarray_like
            Hit features.

        Return
        ------
        Labels : array-like
            Track id labels for the each hit.
        """

        matrix_hough, track_inds, track_params = self.tranform(X)

        self.matrix_hough_ = matrix_hough
        self.track_inds_ = track_inds
        self.track_params_ = track_params


        labels = self.get_hit_labels(track_inds, len(X))

        return labels

    def predict(self, X):
        """
        Tracks pattern recognition for several events.

        Parameters
        ----------
        X : ndarray_like
            Hit features.

        Return
        ------
        Labels : array-like
            Track id labels for the each hit.
        """

        unique_event_ids = np.unique(X[:, 0])
        labels = np.empty(len(X), dtype='int')
        
        for event_id in unique_event_ids:
            event_indices = (X[:, 0] == event_id)
            # select an event and drop event ids
            X_event = X[event_indices][:, 1:]
            labels[event_indices] = self.predict_one_event(X_event)

        return np.array(labels)

