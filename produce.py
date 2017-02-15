
import numpy as np
from simulate import *

sim = Simulator()

N = 200
Mmin = 2
Mmax = 9

data = pd.DataFrame({'event':[0],'particle':[0],'hit':[0], 'x':[0.], 'y':[0.]})
data = data.drop(data.index[[0]])

data_particle = pd.DataFrame({'event':[0],'particle':[0],'pt':[0.], 'phi':[0.], 'xVtx':[0.], 'yVtx':[0.]})
data_particle = data_particle.drop(data_particle.index[[0]])


for ievent in range(0,N):

    event = pd.DataFrame({'event':[0],'particle':[0],'hit':[0], 'x':[0.], 'y':[0.]})
    event = event.drop(event.index[[0]])

    event_particle = pd.DataFrame({'event':[0],'particle':[0],'pt':[0.], 'phi':[0.], 'xVtx':[0.], 'yVtx':[0.]})
    event_particle = event_particle.drop(event_particle.index[[0]])


    M = np.random.random_integers(Mmin,Mmax)

    for p in range(0,M):
        a = 0.6
        v = 0.2
        position = a*(2.*np.random.random(2)-[1.,1.])
        velocity = v*(2.*np.random.random(2)-[1.,1.])

        pt=np.linalg.norm(velocity)
        phi=np.arctan2(velocity[1],velocity[0])
        xVtx=position[0]
        yVtx=position[1]

        sim.detector.reset()
        simtrack=sim.propagate(position,velocity, step = 20, id=p)
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

        #data=data.append(sim.detector.getHits(), ignore_index=True)

    data=data.append(event, ignore_index=True)
    data_particle=data_particle.append(event_particle, ignore_index=True)

data.to_csv("input_20.csv",header=True,cols=['event','particle','hit', 'x', 'y'], engine='python')
data_particle.to_csv("particles_20.csv",header=True,cols=['event','particle','pt', 'phi', 'xVtx', 'yVtx'], engine='python')



