import pandas as pd
import numpy as np

import Clustering as trk
import Fitting as fit

N = 1000

if __name__ == '__main__':
    print("Reading file ...")

    fitter = fit.TrackFitter(B=1.)

    data_track = pd.DataFrame({'event_id':[0],'track_id':[0],'pt':[0.], 'phi':[0.], 'xVtx':[0.], 'yVtx':[0.]})
    data_track = data_track.drop(data_track.index[[0]])

    df = pd.read_csv("hits_"+str(N)+".csv")
    y_df = df[['particle']]
    #    X_df = df.drop(['hit','layer','particle','event'], axis=1)
    #    X_df = df.drop(['cluster_id','layer','iphi'], axis=1)

#replace particle with 100000000*event+particle

    y_train = df['particle'].values + df['event'].values * 1000
    tracker = trk.ClusterDBSCAN(eps=0.004, rscale=0.001)
    
    tracker.fit(X_df.values[:1000], y_train[:1000])

    df_result = df.drop(['particle'], axis=1)
    y_predicted = tracker.predict(df_result.values)
    
    
    df_result=pd.concat([df_result,pd.DataFrame({'track':y_predicted})],axis=1)
    for col in ['event','track']:
        df_result[col] = df_result[col].astype('int32')
    df_result_truth=pd.concat([df_result,pd.DataFrame({'particle':y_train})],axis=1)
    df_result_truth['particle'] = df_result_truth['particle'].astype('int32')

    df_result.to_csv("result_"+str(N)+".csv",header=True,cols=['event','track', 'x', 'y'], engine='python')
    df_result_truth.to_csv("result_truth_"+str(N)+".csv",header=True,cols=['event','particle','track', 'x', 'y'], engine='python')
    
    
    # event loop for fitting
    for ievent in []: # np.unique(df['event'].values):
        if ievent % 100 == 0 : print("processing event ", ievent)
        event_indices=(df['event']==ievent).values
        event_df = df.loc[event_indices]
 
        y_event_predicted = y_predicted[event_indices]
        #       y_event_predicted = y_event_predicted + [ievent*1000]*len(y_event_predicted)
        y_predicted = np.append(y_predicted,y_event_predicted)

        for itrack in np.unique(y_event_predicted):
            xhit = X_df['x'].values[y_event_predicted[:] == itrack]
            yhit = X_df['y'].values[y_event_predicted[:] == itrack]
    #         pt,phi,vx,vy,chg=fitter.fit(xhit,yhit)







