import numpy as np


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

            #           eff[ipart] = hitintrack[ipart]/len(true_hits)

            # evaluate contamination
            #            overlap = (y_pred[:] == maxcluster)
            #            others = (y_test[:] != particle)
            #            mask = overlap & others
            #            noise_hits = y_pred[mask]
            #            con[ipart] = len(noise_hits) / len(true_hits)

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





