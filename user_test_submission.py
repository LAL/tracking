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
    return X_df, y_df



if __name__ == '__main__':
    print("Reading file ...")

    X_df, y_df = read_data(filename)
    events = np.unique(X_df['event'].values)

    #no training, use all sample for test:
    skf = ShuffleSplit(
    len(events), n_iter=1, test_size=0.2, random_state=57)

    print("Training file ...")
    for train_is, test_is in skf:
        print '--------------------------'

        print train_is

        # use dummy clustering
        #tracker = Tracking.HitToTrackAssignment()
        #tracker = Clustering.ClusterDBSCAN(eps=0.5, rscale=0.001)
        #tracker = Hough.Hough(n_theta_bins=100, n_radius_bins=100, min_radius=1., min_hits=4)
        #tracker = NearestHit.NearestHit(min_cos_value=0.9)
        tracker = LinearApproximation.LinearApproximation(min_hits=4, window_width=0.03)

        train_hit_is = np.array([])
        test_hit_is = np.array([])

        for event in train_is:
            train_hit_is = np.append(train_hit_is,X_df.loc[X_df['event'] == event].index)

        for event in test_is:
            test_hit_is = np.append(test_hit_is,X_df.loc[X_df['event'] == event].index)

        X_train_df = X_df.iloc[train_is].copy()
        y_train_df = y_df.iloc[train_is].copy()
        X_test_df = X_df.iloc[test_is].copy()
        y_test_df = y_df.iloc[test_is].copy()
        y_test = np.zeros((len(y_test_df),2))
        y_predicted = np.zeros((len(y_test_df),2))
        y_test_events = X_test_df['event'].values

        tracker.fit(X_train_df.values, y_train_df.values)
        y_predicted[:,0] = tracker.predict(X_test_df.values)
        y_predicted[:,1] = y_test_events
        y_test[:,0] = y_test_df['particle'].values
        y_test[:,1] = y_test_events

        # Score the result
        total_score = score(y_test, y_predicted)
        print 'average score = ', total_score

