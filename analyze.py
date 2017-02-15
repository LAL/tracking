import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import score_cluster
import score_physics
import plotting


def analyze():
    
    df = pd.read_csv("result_truth_2000.csv")

    g_eff=[]
    g_fake=[]
    g_npar=[]

    events=np.unique(df['event'].values)
    for ievent in events:
        df_event=df.loc[df['event']==ievent]

        particles=np.unique(df_event['particle'].values)
        npar=len(particles)

        y_test=df_event['particle'].values.astype(int)
        y_pred=df_event['track'].values.astype(int)

        eff_event, fake_event = score_cluster.evaluate(y_test, y_pred,-1)
        g_eff = g_eff + [eff_event]
        g_fake = g_fake + [fake_event]
        g_npar = g_npar + [npar]

        for iparticle in particles:
            eff, fake = score_cluster.evaluate(y_test, y_pred,iparticle)

    return g_eff, g_fake, g_npar





