import pandas as pd
import numpy as np

from sklearn.cross_validation import ShuffleSplit

import Tracking
import Score_assignment

filename = "hits_10.csv"

def read_data(filename):
    df = pd.read_csv(filename)
    y_df = df[['particle']]
    X_df = df.drop(['hit','particle'], axis=1)
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
        tracker = Tracking.HitToTrackAssignmet()

        X_train_df = X_df.iloc[train_is].copy()
        y_train_df = y_df.iloc[train_is].copy()
        X_test_df = X_df.iloc[test_is].copy()
        y_test_df = y_df.iloc[test_is].copy()

        # Temporarily bypass splitting (need to avoid shuffling events)
        X_test_df = X_df.copy()
        y_test_df = y_df.copy()
        
        tracker.fit(X_train_df.values, y_train_df.values)
        score = 0
        events = np.unique(X_test_df['event'].values)
        for ievent in events:
            event_df = X_test_df.loc[X_test_df['event']==ievent]
            y_event_df = y_test_df.loc[X_test_df['event']==ievent]
            y_predicted = tracker.predict(event_df.values)
            print y_predicted
            event_score = Score_assignment.evaluate(y_event_df.values[:,0], y_predicted)
            score += event_score
        score /= len(events)
        print 'average score = ', score

