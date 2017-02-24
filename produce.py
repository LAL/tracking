
import numpy as np
from Simulate import *

sim = Simulator()

N = 1000
Mmin = 1
Mmax = 1

data = pd.DataFrame({'event':[0],'particle':[0],'hit':[0],'layer':[0], 'x':[0.], 'y':[0.]})
data = data.drop(data.index[[0]])

data_particle = pd.DataFrame({'event':[0],'particle':[0],'pt':[0.], 'phi':[0.], 'xVtx':[0.], 'yVtx':[0.]})
data_particle = data_particle.drop(data_particle.index[[0]])

print "Will now produce ",N," events with ",Mmax, " tracks"
for ievent in range(0,N):

    if(ievent % 1 == 0): print "processing event : ",ievent
    event = pd.DataFrame({'event':[0],'particle':[0],'hit':[0],'layer':[0], 'x':[0.], 'y':[0.]})
    event = event.drop(event.index[[0]])

    event_particle = pd.DataFrame({'event':[0],'particle':[0],'pt':[0.], 'phi':[0.], 'xVtx':[0.], 'yVtx':[0.]})
    event_particle = event_particle.drop(event_particle.index[[0]])


    M = np.random.random_integers(Mmin,Mmax)
    sim.detector.reset()

    for p in range(0,M):
        d = 0.2
        v = 3.
        position = d*(2.*np.random.random(2)-[1.,1.])
        velocity = v*(2.*np.random.random(2)-[1.,1.])

        pt=np.linalg.norm(velocity)
        phi=np.arctan2(velocity[1],velocity[0])
        xVtx=position[0]
        yVtx=position[1]

        #DR simtrack=sim.propagate(position,velocity, step = 20, id=p)
        simtrack=sim.propagate_direct(position,velocity, id=p)
        simtrack = pd.concat(
                             [pd.DataFrame({'event':[ievent]*len(simtrack.index)}),
                              simtrack],
                             axis=1
                             )
        event=event.append(simtrack, ignore_index=True)

        event_particle=event_particle.append(pd.concat(
                                        [pd.DataFrame({'event':[ievent],'particle':[p],
                                                      'pt':[pt],'phi':[phi],
                                                      'xVtx':[xVtx],'yVtx':[yVtx]})]
                                        )
                              )

    hits=sim.detector.getHits()
    data_event=pd.concat(
                         [pd.DataFrame({'event':[ievent]*len(hits.index)}),
                          hits],
                         axis=1
                         )

    data=data.append(data_event, ignore_index=True)
    data_particle=data_particle.append(event_particle, ignore_index=True)

data.to_csv("hits_100.csv",header=True,cols=['event','particle','hit','layer', 'x', 'y'], engine='python')
data_particle.to_csv("particles_100.csv",header=True,cols=['event','particle','pt', 'phi', 'xVtx', 'yVtx'], engine='python')



