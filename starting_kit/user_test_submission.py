import pandas as pd
import numpy as np

from sklearn.cross_validation import ShuffleSplit

import Tracking

filename = "hits_10.csv"

def read_data(filename):
    df = pd.read_csv(filename)
    y_df = df[['particle']]
    X_df = df.drop(['hit','particle'], axis=1)
    return X_df, y_df

def score(y_test, y_pred):
    
    eff_total = 0.
    #    fake_total = 0.
    
    particles = np.unique(y_test)
    npart = len(particles)
    nhit = len(y_test)
    dummyarray = np.full(shape=nhit + 1,fill_value=-1, dtype='int64')
    
    assignedtrack = np.full(shape=npart,fill_value=-1, dtype='int64')
    hitintrack = np.full(shape=npart,fill_value=0, dtype='int64')
    eff = np.full(shape=npart,fill_value=0.)
    con = np.full(shape=npart,fill_value=0.)
    
    # assign tracks to particles
    ipart = 0
    for particle in particles:
        
        eff[ipart] = 0.
        con[ipart] = 0.
        
        true_hits = y_test[y_test[:] == particle]
        found_hits = y_pred[y_test[:] == particle]
        
        nsubcluster=len(np.unique(found_hits[found_hits[:] >= 0]))
        
        if(nsubcluster > 0):
            b=np.bincount(found_hits[found_hits[:] >= 0])
            a=np.argmax(b)
            
            maxcluster = a
            
            assignedtrack[ipart]=maxcluster
            hitintrack[ipart]=len(found_hits[found_hits[:] == maxcluster])
        
        ipart += 1
    
    
    # resolve duplicates and count good assignments
    ipart = 0
    sorted=np.argsort(hitintrack)
    hitintrack=hitintrack[sorted]
    assignedtrack=assignedtrack[sorted]
    print hitintrack
    for particle in particles:
        itrack=assignedtrack[ipart]
        if((itrack < 0) | (len(assignedtrack[assignedtrack[:] == itrack])>1)):
            hitintrack = np.delete(hitintrack,ipart)
            assignedtrack = np.delete(assignedtrack,ipart)
        else:
            ipart += 1
    ngood = 0.
    ngood = np.sum(hitintrack)
    eff_total = eff_total + (float(ngood) / float(nhit))
    
    # remove combinatorials
    print npart, nhit, eff_total
    return eff_total





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
        score = 0
        events = np.unique(X_test_df['event'].values)
        for ievent in events:
            event_df = X_test_df.loc[X_test_df['event']==ievent]
            y_event_df = y_test_df.loc[X_test_df['event']==ievent]
            y_predicted = tracker.predict(event_df.values)
            print y_predicted
            event_score = score(y_event_df.values[:,0], y_predicted)
            score += event_score
        score /= len(events)
        print 'average score = ', score

