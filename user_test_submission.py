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

filename = "data/hits_merged.csv"

def read_data(filename):
    df = pd.read_csv(filename)
    y_df = df[['particle']] + 1000 * df[['event']].values
    X_df = df.drop(['particle'], axis=1)
    return X_df.values, y_df.values



if __name__ == '__main__':
    print("Reading file ...")

    X, y = read_data(filename)
    events = np.unique(X[:,0])

    #no training, use all sample for test:
    skf = ShuffleSplit(
    len(events), n_iter=1, test_size=0.2, random_state=57)

    print("Training file ...")
    for train_is, test_is in skf:
        print '--------------------------'

        # use dummy clustering
        #tracker = Tracking.HitToTrackAssignment()
        #tracker = Clustering.ClusterDBSCAN(eps=0.5, rscale=0.001)
        #tracker = Hough.Hough(n_theta_bins=100, n_radius_bins=100, min_radius=1., min_hits=4)
        #tracker = NearestHit.NearestHit(min_cos_value=0.9)
        tracker = LinearApproximation.LinearApproximation(min_hits=4, window_width=0.03)

        train_hit_is = np.where(np.in1d(X[:,0],train_is))
        test_hit_is = np.where(np.in1d(X[:,0],test_is))

        X_train = X[train_hit_is]
        y_train = y[train_hit_is]

        X_test = X[test_hit_is]
        y_test = y[test_hit_is]

        y_test_e = np.zeros((len(y_test),2))
        y_predicted = np.zeros((len(y_test),2))

        tracker.fit(X_train, y_train)
        y_predicted[:,0] = tracker.predict(X_test)
        y_predicted[:,1] = X_test[:,0]
        print len(y_test)
        print len(X_test[:,0])
        y_test_e[:,0] = y_test[:,0]
        y_test_e[:,1] = X_test[:,0]

        # Score the result
        total_score = score(y_test_e, y_predicted)
        print 'average score = ', total_score

