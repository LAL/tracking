import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Particle(object):

    def __init__(self, x = [], vx = [], id = 0):

        self.history = pd.DataFrame()
        self.position = np.zeros(3);
        self.momentum = np.zeros(3);
        self.id = id
        self.position[0] = x[0]
        self.position[1] = x[1]

        self.momentum[0] = vx[0]
        self.momentum[1] = vx[1]

        self.history = pd.DataFrame({'particle':[self.id],'hit':[0], 'x':self.position[0], 'y':self.position[1]})
        self.history= self.history.drop(self.history.index[[0]])
        #        print self.history

        pass

    def update(self,acceleration, time, detector, stephit = 1):
        self.momentum += acceleration;
        self.position += self.momentum;

        detector.deposit(self.position)
        if((time % stephit == 0) & (np.linalg.norm(self.position) > 5.)):
            self.history = self.history.append(pd.DataFrame({'particle':[self.id],'hit':[time], 'x':[self.position[0]], 'y':[self.position[1]]}), ignore_index=True)
        pass



class Detector(object):
    def __init__(self):
        self.Nrho = 10
        self.Nphi = 10
        self.cells_x = np.zeros((self.Nrho, self.Nphi))
        self.cells_y = np.zeros((self.Nrho, self.Nphi))
        self.cells_width = np.zeros((self.Nrho, self.Nphi))
        self.cells_hit = np.zeros((self.Nrho, self.Nphi))
        self.history = pd.DataFrame({'particle':[0],'hit':[0], 'x':[0], 'y':[1]})
        self.history= self.history.drop(self.history.index[[0]])
        self.detsize = 0.01

    def reset(self):
        self.cells_hit = np.zeros((self.Nrho, self.Nphi))
        for irho in range(0, self.Nrho):
            for iphi in range(0,self.Nphi):
                rho = 20*irho
                phi = 2.*2.*np.pi*(iphi-self.Nphi/2)/self.Nphi
                self.cells_x[irho,iphi] = rho*np.cos(phi)
                self.cells_y[irho,iphi] = -rho*np.sin(phi)
                pass
            pass
        pass
        self.history = pd.DataFrame({'hit':[22], 'x':[0.], 'y':[0.]})

    def deposit(self,position):
        for irho in range(0, self.Nrho):
            for iphi in range(0,self.Nphi):
                if(np.linalg.norm(position - np.array(self.cells_x[irho,iphi],self.cells_y[irho,iphi])) < self.detsize):
                    self.cells_hit[irho,iphi] = 1
        return 0

    def getHits(self):
        for irho in range(0, self.Nrho):
            for iphi in range(0,self.Nphi):
                if(self.cells_hit[irho,iphi] == 1):
                    self.history = self.history.append(pd.DataFrame({'hit':[time], 'x':self.cells_x[irho,iphi], 'y':self.cells_y[irho,iphi]}), ignore_index=True)

        return self.history


class Simulator(object):
    def __init__(self):
        self.p = Particle([0,0,0], [0,0,0])
        self.detector = Detector()


    def force(self,position, momentum):
        #        g = 0.5
        #        acc = -g * position / pow(np.linalg.norm(position),3) # gravitational force
        b = 0.04/40.
        # This is non relativistic!
        acc = - np.cross(momentum, [0,0,b])
        return acc


    def propagate(self,x=[], v=[], step = 1, id = 0):
        #        print "New planet"
        self.p = Particle(x,v,id)

        for t in range(0,300):
            acceleration = self.force(self.p.position,self.p.momentum)
            self.p.update(acceleration, t, self.detector, stephit = step)
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

