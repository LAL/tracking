import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Particle(object):

    def __init__(self, x = [], vx = []):

        self.history = pd.DataFrame()
        self.position = np.zeros(3);
        self.momentum = np.zeros(3);

        self.position[0] = x[0]
        self.position[1] = x[1]

        self.momentum[0] = vx[0]
        self.momentum[1] = vx[1]

        history = pd.DataFrame({'t':[22], 'x':self.position[0], 'y':self.position[1]})
        print history

        pass

    def update(self,acceleration, time, detector):
        self.momentum += acceleration;
        self.position += self.momentum;

        detector.deposit(self.position)
        self.history = self.history.append(pd.DataFrame({'t':[time], 'x':[self.position[0]], 'y':[self.position[1]]}), ignore_index=True)
        pass



class Detector(object):
    def __init__(self):
        self.Nrho = 10
        self.Nphi = 10
        self.cells_x = np.zeros(Nrho, Nphi)
        self.cells_y = np.zeros(Nrho, Nphi)
        self.cells_width = np.zeros(Nrho, Nphi)
        self.cells_hit = np.zeros(Nrho, Nphi)

    def reset(self):
        self.detsize = 0.01
        for irho in range(0, Nrho):
            for iphi in range(0,Nphi):
                phi = np.pi*iphi
                pass
            pass
        pass
        history = pd.DataFrame({'t':[22], 'position':[[0,0,0]]})

    def deposit(self,position):
        for irho in range(0, Nrho):
            for iphi in range(0,Nphi):
                if(np.linalg.norm(position - np.array(self.cells_x[Nrho,Nphi],self.cells_y[Nrho,Nphi])) < self.detsize):
                self.cells_hit[irho,iphi] = 1
        return 0




class Simulator(object):
    def __init__(self):
        self.p = Particle([0,0,0], [0,0,0])
        self.detector = Detector()


    def force(self,position, momentum):
        #        g = 0.5
        #        acc = -g * position / pow(np.linalg.norm(position),3) # gravitational force
        b = 0.04/10.
        acc = - np.cross(momentum, [0,0,b])
        return acc


    def propagate(self,x=[], v=[]):
        print "New planet"
        self.p = Particle(x,v)

        for t in range(0,100):
            acceleration = self.force(self.p.position,self.p.momentum)
            self.p.update(acceleration, t, self.detector)
#            if(t % 10 == 0):
#                print t, p.position

        return self.p.history

    def plot(self):

        x=np.dstack((self.p.history['x'].values))[0][0]
        y=np.dstack((self.p.history['y'].values))[0][0]
        plt.plot(x,y)
        plt.show()









#s = Simulator()

#print s.generate(50.0, 0.0, 0.02,0.05)
#s.generate(30.0, 0.0, 0.02,0.05)
#s.generate(30.0, 10.0, 0.02,0.05)
#s.generate(30.0, 15.0, 0.02,0.05)

