import numpy as np


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





