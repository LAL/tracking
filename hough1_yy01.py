__author__ = 'mikhail91'


import numpy as np

from formulate import *

def rotateArray(x, y, phi):
    c, s = np.cos(phi), np.sin(phi)
    xr=c*x-s*y
    yr=s*x+c*y
    
    return xr,yr




def calc_z(x, y):
    return funk(x,y,[],"(x**2+y**2)**(0.5)")

def calc_v(r, phi):
    return funk(r,phi,[], "2. * np.cos(y) / x")
# first coefficient affected by binning


def calc_vxy(x, y):
    return funk(x,y,[], "2. * np.cos(np.arctan2(y,x)) / np.sqrt(x**2+y**2)")


class Clusterer(object):

    def __init__(self, n_theta_bins=5000, n_radius_bins=1000, min_radius=2., min_hits=2):
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


    def transform(self, X):
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
        hit_zs = calc_z(hit_rs, hit_phis)
    
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
            v = calc_v(hit_rs, hit_phis - theta)
            xr, yr = rotateArray(x, y, theta)
            v = funk(xr,yr,[],formulate())

            # Hough Transform digitization
            bin_inds = np.digitize(v, track_invrs)
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
        
        self.min_radius = 2.
        # first coefficient affected by binning

        pass

    def predict_single_event(self, X):
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

        matrix_hough, track_inds, track_params = self.transform(X)

        self.matrix_hough_ = matrix_hough
        self.track_inds_ = track_inds
        self.track_params_ = track_params


        labels = self.get_hit_labels(track_inds, len(X))

        return labels




