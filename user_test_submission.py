import pandas as pd
import numpy as np

from sklearn.cross_validation import ShuffleSplit

import Tracking
import Clustering
import Hough
import NearestHit
import LinearApproximation

from Score_assignment import *

debug = False
coleventy = 1
coleventX = 4

filename = "data/hits_merged.csv"

def read_data(filename):
    df = pd.read_csv(filename)[['layer','iphi','x','y','particle','event']]
    y_df = df.drop(['layer','iphi','x','y'], axis=1)
    X_df = df.drop(['particle'], axis=1)
    return X_df.values, y_df.values



if __name__ == '__main__':
    print("Reading file ...")

    X, y = read_data(filename)
    events = np.unique(X[:,coleventX])

    #no training, use all sample for test:
    skf = ShuffleSplit(
    len(events), n_iter=1, test_size=0.1, random_state=57)

    print("Training file ...")
    for train_is, test_is in skf:
        print '--------------------------'

        # use dummy clustering
        #tracker = Tracking.HitToTrackAssignment()
        #tracker = Clustering.ClusterDBSCAN(eps=0.5, rscale=0.001)
        tracker = Hough.Hough(n_theta_bins=5000, n_radius_bins=1000, min_radius=20., min_hits=4)
        #tracker = NearestHit.NearestHit(min_cos_value=0.9)
        #tracker = LinearApproximation.LinearApproximation(min_hits=4, window_width=0.03)

        train_hit_is = np.where(np.in1d(y[:,coleventy],train_is))
        test_hit_is = np.where(np.in1d(y[:,coleventy],test_is))

        X_train = X[train_hit_is]
        y_train = y[train_hit_is]

        X_test = X[test_hit_is]
        y_test = y[test_hit_is]

        y_test_e = np.zeros((len(y_test),2))
        y_predicted = np.zeros((len(y_test),2))

        tracker.fit(X_train, y_train)

        y_predicted[:,0] = tracker.predict(X_test)
        y_predicted[:,1] = X_test[:,coleventX]

        # Score the result
        total_score = score(y_test, y_predicted)
        print 'average score = ', total_score

