import pandas as pd
import numpy as np

from sklearn.cross_validation import ShuffleSplit

import Tracking

debug = False
coleventy = 1
coleventX = 4

filename = "../data/hits_merged.csv"

def read_data(filename):
    df = pd.read_csv(filename)[['layer','iphi','x','y','particle','event']]
    y_df = df.drop(['layer','iphi','x','y'], axis=1)
    X_df = df.drop(['particle'], axis=1)
    return X_df.values, y_df.values

def score(y_test, y_pred):
    
    total_score = 0.
    y_events = y_test[:,coleventy]
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
        
        total_score += eff_total
    
    
    total_score /= len(y_events)
    

    return eff_total




if __name__ == '__main__':
    print("Reading file ...")

    X, y = read_data(filename)
    events = np.unique(X[:,coleventX])

    #no training, use all sample for test:
    skf = ShuffleSplit(
    len(events), n_iter=1, test_size=0.1, random_state=57)

    print("Training file ...")
    for train_is, test_is in skf:
        print '--------------------------'

        # use dummy clustering
        tracker = Tracking.HitToTrackAssignment()

        train_hit_is = np.where(np.in1d(y[:,coleventy],train_is))
        test_hit_is = np.where(np.in1d(y[:,coleventy],test_is))

        X_train = X[train_hit_is]
        y_train = y[train_hit_is]

        X_test = X[test_hit_is]
        y_test = y[test_hit_is]

        y_predicted = np.zeros((len(y_test),2))

        tracker.fit(X_train, y_train)

        y_predicted = tracker.predict(X_test)

        # Score the result
        total_score = score(y_test, y_predicted)
        print 'average score = ', total_score

