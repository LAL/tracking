import numpy as np
# import pandas as pd



def evaluate(y_test, y_pred, particle = -1):

    eff_total = 0.
    fake_total = 0.

    # evaluate hit efficiency
    listtoscore = [particle]
    if(particle < 0):
        listtoscore = np.unique(y_test)
    
    dummyarray = np.full(shape=len(y_test) + 1,fill_value=-1)

    for particle in listtoscore:
        
        eff = 0.
        fake = 0.

#        print "particle : ", particle
        true_hits = y_test[y_test[:] == particle]
        found_hits = y_pred[y_test[:] == particle]

        nsubcluster=len(np.unique(found_hits[found_hits[:] >= 0]))

#        if(particle >= 0):
            #            print "true hits : ", true_hits
            #            print "found hits : ", found_hits
            #            print "found clusters : ", nsubcluster


        if(nsubcluster > 0):
            # fix for degeneracy!

            b=np.bincount(found_hits[found_hits[:] >= 0])
            a=np.argmax(b)

            maxcluster = a
            eff = len(found_hits[found_hits[:] == maxcluster])/len(true_hits)

            # evaluate noise
            overlap = (y_pred[:] == maxcluster)
            others = (y_test[:] != particle)
            mask = overlap & others
            noise_hits = y_pred[mask]
            fake = len(noise_hits) / len(true_hits)

#            print "efficiency : ", eff
#            print "fake : ", fake

        eff_total += eff
        fake_total += fake
        
    
    # remove combinatorials
    
    return eff, fake





