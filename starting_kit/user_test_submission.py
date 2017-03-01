import pandas as pd
import numpy as np

from sklearn.cross_validation import ShuffleSplit

import Tracking

def score(y_test, y_pred):
    
    total_score = 0.
    y_events = y_test[:,1]
    y_test = y_test[:,0]
    y_pred = y_pred[:,0]
    
    events = np.unique(y_events)
    for ievent in events:
        eff_total = 0.
        event_indices=(y_events==ievent)
        y_test_event = y_test[event_indices]
        y_pred_event = y_pred[event_indices]
        
        particles = np.unique(y_test_event)
        npart = len(particles)
        nhit = len(y_test_event)
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
            
            true_hits = y_test_event[y_test_event[:] == particle]
            found_hits = y_pred_event[y_test_event[:] == particle]
            
            nsubcluster=len(np.unique(found_hits[found_hits[:] >= 0]))
            
            if(nsubcluster > 0):
                b=np.bincount((found_hits[found_hits[:] >= 0]).astype(dtype='int64'))
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
        #    print hitintrack
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
        
        total_score += eff_total
    
    
    total_score /= len(y_events)
    return eff_total



filename = "hits_10.csv"

def read_data(filename):
    df = pd.read_csv(filename)
    y_df = df[['particle']] + 1000 * df[['event']].values
    X_df = df.drop(['hit','particle'], axis=1)
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

        # tracker = Tracking.ClusterDBSCAN(eps=0.004, rscale=0.001)
        # use dummy clustering
        tracker = Tracking.HitToTrackAssignment()

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

