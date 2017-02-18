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

        detector.deposit(self.position, self.id)
        if((time % stephit == 0) & (np.linalg.norm(self.position) > 2.)):
            self.history = self.history.append(pd.DataFrame({'particle':[self.id],'hit':[time], 'x':[self.position[0]], 'y':[self.position[1]]}), ignore_index=True)
        pass



class Detector(object):
    def __init__(self):
        self.Nrho = 50
        self.Nphi = 300
        self.Npipe = 10
        self.cells_r = np.array(range(self.Npipe,self.Nrho+self.Npipe)) * 5. / self.Nrho;
        self.cells_phi = 2.*np.pi*np.array(range(0,self.Nphi))/self.Nphi
        self.detsize = self.cells_r * 2.*np.pi/self.Nphi
        self.thickness = 0.01
        self.cells_x = np.zeros((self.Nrho, self.Nphi))
        self.cells_y = np.zeros((self.Nrho, self.Nphi))
        self.hit_particle = np.zeros((self.Nrho, self.Nphi))
        self.cells_width = np.zeros((self.Nrho, self.Nphi))
        self.cells_hit = np.zeros((self.Nrho, self.Nphi))
        self.history = pd.DataFrame({'particle':[0],'hit':[0], 'x':[0], 'y':[0]})
        self.history= self.history.drop(self.history.index[[0]])

    def reset(self):
        self.cells_hit = np.zeros((self.Nrho, self.Nphi))
        for irho in range(0, self.Nrho):
            for iphi in range(0,self.Nphi):
                rho = self.cells_r[irho]
                phi = self.cells_phi[iphi]
                self.cells_x[irho,iphi] = rho*np.cos(phi)
                self.cells_y[irho,iphi] = rho*np.sin(phi)
        self.history = pd.DataFrame({'particle':[0], 'hit':[0], 'x':[0.], 'y':[0.]})
        self.history = self.history.drop(self.history.index[[0]])


    def deposit(self,position, particle=0):
        for irho in range(0, self.Nrho):
            for iphi in range(0,self.Nphi):
                #                if(np.linalg.norm(position - [self.cells_x[irho,iphi],self.cells_y[irho,iphi],0]) < self.detsize[irho]):
                if((np.fabs(np.linalg.norm(position) - self.cells_r[irho]) < self.thickness) &
                   (np.fabs(np.mod((np.arctan2(position[1],position[0])-self.cells_phi[iphi]), 360.)) < self.detsize[irho])):

                    #think about overlap
                    self.hit_particle[irho,iphi] = particle
                    self.cells_hit[irho,iphi] = 1
        return 0

    def getHits(self):
        for irho in range(0, self.Nrho):
            for iphi in range(0,self.Nphi):
                if(self.cells_hit[irho,iphi] == 1):
                    self.history = self.history.append(pd.DataFrame({'particle':self.hit_particle[irho,iphi], 'hit':[0], 'x':self.cells_x[irho,iphi], 'y':self.cells_y[irho,iphi]}), ignore_index=True)

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

        for t in range(0,100):
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

