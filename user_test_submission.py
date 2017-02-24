import pandas as pd
import numpy as np

from sklearn.cross_validation import ShuffleSplit

import Tracking

from Score_assignment import *


filename = "hits_10.csv"

def read_data(filename):
    df = pd.read_csv(filename)
    y_df = df[['particle']] + 1000 * df[['event']].values
    X_df = df.drop(['hit','particle','Unnamed: 0'], axis=1)
    return X_df, y_df



if __name__ == '__main__':
    print("Reading file ...")

    X_df, y_df = read_data(filename)

    #no training, use all sample for test:
    skf = ShuffleSplit(
    len(y_df), n_iter=1, test_size=0.99, random_state=57)
    print("Training file ...")
    for train_is, test_is in skf:
        print '--------------------------'

        # tracker = Tracking.ClusterDBSCAN(eps=0.004, rscale=0.001)
        # use dummy clustering
        tracker = Tracking.HitToTrackAssignment()

        X_train_df = X_df.iloc[train_is].copy()
        y_train_df = y_df.iloc[train_is].copy()
        X_test_df = X_df.iloc[test_is].copy()
        y_test_df = y_df.iloc[test_is].copy()

        # Temporarily bypass splitting (need to avoid shuffling events)
        X_test_df = X_df.copy()
        y_test_df = y_df.copy()
        
        tracker.fit(X_train_df.values, y_train_df.values)
        y_predicted = tracker.predict(X_test_df.values)

        # Score the result
        total_score = 0.
        events = np.unique(X_test_df['event'].values)
        for ievent in events:
            event_indices=(X_test_df['event']==ievent).values
            y_event_df = y_test_df.loc[event_indices]
            y_predicted_event = y_predicted[event_indices]
            #          print y_predicted_event
            event_score = score(y_event_df.values[:,0], y_predicted_event)
            total_score += event_score
        total_score /= len(events)
        print 'average score = ', total_score

