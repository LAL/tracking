import pandas as pd
import numpy as np

import Clustering as trk
import Fitting as fit


if __name__ == '__main__':
    print("Reading file ...")

    fitter = fit.TrackFitter(B=1.)

    data_track = pd.DataFrame({'event':[0],'track':[0],'pt':[0.], 'phi':[0.], 'xVtx':[0.], 'yVtx':[0.], 'chg':[0.]})
    data_track = data_track.drop(data_track.index[[0]])

    df = pd.read_csv("hits_100.csv")
    y_df = df[['particle']]
    #    X_df = df.drop(['hit','layer','particle','event'], axis=1)
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

        for itrack in np.unique(y_event_predicted):
            xhit = X_df['x'].values[y_event_predicted[:] == itrack]
            yhit = X_df['y'].values[y_event_predicted[:] == itrack]
            pt,phi,vx,vy=fitter.fit(xhit,yhit)


    df_result=pd.concat([df_result,pd.DataFrame({'track':y_predicted})],axis=1)
    df_result.to_csv("result_100.csv",header=True,cols=['event','track','hit', 'x', 'y'], engine='python')

    df_result=pd.concat([df_result,pd.DataFrame({'particle':y_train})],axis=1)
    df_result.to_csv("result_truth_100.csv",header=True,cols=['event','particle','track','hit', 'x', 'y'], engine='python')



