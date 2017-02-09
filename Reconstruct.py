import pandas as pd
import numpy as np

import tracking as trk


if __name__ == '__main__':
    print("Reading file ...")

    df = pd.read_csv("test.csv")
    print df
    y_df = df[['particle']]
    X_df = df.drop(['hit','particle','event'], axis=1)
    #replace particle with 100000000*event+particle

    y_train = df['event'].values * 1000 + df['particle'].values
    print y_train
    tracker = trk.ClusterDBSCAN(eps = 0.04)
    
    tracker.fit(X_df.values, y_train)

    df_result = df.drop(['particle'], axis=1)
    y_predicted = np.array([])
    for ievent in np.unique(df['event'].values):
        event_df = df.loc[df['event']==ievent]
        y_df = event_df[['particle']]
        X_df = event_df.drop(['hit','particle','event'], axis=1)

        y_predicted = np.append(y_predicted,tracker.predict(X_df.values))


    df_result=pd.concat([df_result,pd.DataFrame({'track':y_predicted})],axis=1)

    df_result=pd.concat([df_result,pd.DataFrame({'particle':y_train})],axis=1)

    print df_result
