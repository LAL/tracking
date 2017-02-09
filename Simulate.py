
from Generate import *

sim = Simulator()

N = 10
M = 2

data = pd.DataFrame({'event':[0],'particle':[0],'hit':[0], 'x':[0.], 'y':[0.]})
data = data.drop(data.index[[0]])

for ievent in range(0,N):

    event = pd.DataFrame({'event':[0],'particle':[0],'hit':[0], 'x':[0.], 'y':[0.]})
    event = event.drop(event.index[[0]])

    for p in range(0,M):
        a = 0.6
        v = 0.2
        position = a*(2.*np.random.random(2)-[1.,1.])
        velocity = v*(2.*np.random.random(2)-[1.,1.])
        sim.detector.reset()
        simtrack=sim.propagate(position,velocity, step = 20, id=p)
        simtrack = pd.concat(
                             [pd.DataFrame({'event':[ievent]*len(simtrack.index)}),
                              simtrack],
                             axis=1
                             )
        event=event.append(simtrack, ignore_index=True)

        #data=data.append(sim.detector.getHits(), ignore_index=True)

    data=data.append(event, ignore_index=True)


data.to_csv("test.csv",header=True,cols=['event','particle','hit', 'x', 'y'], engine='python')






