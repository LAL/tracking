import pandas as pd
import numpy as np

import Clustering as trk
import Fitting as fit


if __name__ == '__main__':
    print("Reading file ...")

    df = pd.read_csv("input_2000.csv")
    y_df = df[['particle']]
    X_df = df.drop(['hit','particle','event'], axis=1)
    #replace particle with 100000000*event+particle

    y_train = df['particle'].values + df['event'].values * 1000
    tracker = trk.ClusterDBSCAN(eps = 0.04)
    
    tracker.fit(X_df.values[:1000], y_train[:1000])

    df_result = df.drop(['particle'], axis=1)
    y_predicted = np.array([])
    for ievent in np.unique(df['event'].values):
        if ievent % 100 == 0 : print "processing event ", ievent
        event_df = df.loc[df['event']==ievent]
        y_df = event_df[['particle']]
        X_df = event_df.drop(['Unnamed: 0','hit','particle','event'], axis=1)

        y_event_predicted = tracker.predict(X_df.values)
        y_event_predicted = y_event_predicted + [ievent*1000]*len(y_event_predicted)
        y_predicted = np.append(y_predicted,y_event_predicted)

    df_result=pd.concat([df_result,pd.DataFrame({'track':y_predicted})],axis=1)
    df_result.to_csv("result_2000.csv",header=True,cols=['event','track','hit', 'x', 'y'], engine='python')

    df_result=pd.concat([df_result,pd.DataFrame({'particle':y_train})],axis=1)
    df_result.to_csv("result_truth_2000.csv",header=True,cols=['event','particle','track','hit', 'x', 'y'], engine='python')



