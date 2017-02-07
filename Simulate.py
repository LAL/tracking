
from Generate import *

sim = Simulator()


M = 10

data = pd.DataFrame()

for p in range(0,M):
    a = 0.6
    v = 0.1
    position = a*(2.*np.random.random(2)-[1.,1.])
    velocity = v*(2.*np.random.random(2)-[1.,1.])
    sim.detector.reset()
    sim.propagate(position,velocity)
    data=data.append(sim.detector.getHits(), ignore_index=True)

data.to_csv("test.csv")






