import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Transform import *

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
        self.traceMin = 0.1
    
        self.history = pd.DataFrame({'particle':[self.id],'hit':[0], 'layer':[0], 'x':self.position[0], 'y':self.position[1]})
        self.history= self.history.drop(self.history.index[[0]])
        #        print self.history

        pass

    def update(self,acceleration, time, detector, precision, stephit = 1):
        self.momentum += acceleration;
        self.position += self.momentum/precision;

        deflect = detector.deposit(self.position, self.id)
        if(np.fabs(deflect) > 0) : self.momentum = rotate(self.momentum,deflect)
        if((time % stephit == 0) & (np.linalg.norm(self.position) > self.traceMin)):
            self.history = self.history.append(pd.DataFrame({'particle':[self.id],'hit':[time], 'layer':[0], 'x':[self.position[0]], 'y':[self.position[1]]}), ignore_index=True)
        pass



class Detector(object):
    def __init__(self):
        self.Nrho = 12
        self.Nphi = [180] * self.Nrho
        self.Npipe = 2
        self.range = 5.
        self.sigmaMS = 0.05
        self.cells_r = np.array(range(self.Npipe,self.Nrho+self.Npipe)) * self.range / self.Nrho;
        self.cells_phi = np.zeros((self.Nrho, np.max(self.Nphi)))
        self.cells_x = np.zeros((self.Nrho, np.max(self.Nphi)))
        self.cells_y = np.zeros((self.Nrho, np.max(self.Nphi)))
        self.dphi = np.zeros(self.Nrho)
        self.detsize = np.zeros(self.Nrho)

        for irho in range(0, self.Nrho):
             self.dphi[irho] = 2.*np.pi / self.Nphi[irho]
             self.detsize = self.cells_r[irho] * 2.*np.pi/self.Nphi[irho]
             for iphi in range(0,self.Nphi[irho]):
                 self.cells_phi[irho,iphi] = 2.*np.pi*iphi/self.Nphi[irho]
                 rho = self.cells_r[irho]
                 phi = self.cells_phi[irho,iphi]
                 self.cells_x[irho,iphi] = rho*np.cos(phi)
                 self.cells_y[irho,iphi] = rho*np.sin(phi)

        self.thickness = 0.02
        self.hit_particle = np.zeros((self.Nrho, self.Nphi[0]))
        self.cells_width = np.zeros((self.Nrho, self.Nphi[0]))
        self.cells_hit = np.zeros((self.Nrho, self.Nphi[0]))
        self.history = pd.DataFrame({'particle':[0],'hit':[0], 'layer':[0], 'x':[0], 'y':[0]})
        self.history= self.history.drop(self.history.index[[0]])

    def reset(self):
        self.cells_hit = np.zeros((self.Nrho, self.Nphi[0]))
        for irho in range(0, self.Nrho):
            for iphi in range(0,self.Nphi[0]):
                rho = self.cells_r[irho]
                phi = self.cells_phi[irho,iphi]
        self.history = pd.DataFrame({'particle':[0], 'hit':[0], 'layer':[0], 'x':[0.], 'y':[0.]})
        self.history = self.history.drop(self.history.index[[0]])


    def deposit(self,position, particle):
        deflect=0.
        for irho in range(0, self.Nrho):
            for iphi in range(0,self.Nphi[irho]):
                #                if(np.linalg.norm(position - [self.cells_x[irho,iphi],self.cells_y[irho,iphi],0]) < self.detsize[irho]):
                if(
                   (np.fabs(np.linalg.norm(position) - self.cells_r[irho]) < self.thickness)
                   &
                   (deltaPhiAbs(np.arctan2(position[1],position[0]),self.cells_phi[irho,iphi]) < self.dphi[irho])
                   ):

                    #think about overlap
                    self.hit_particle[irho,iphi] = particle
                    self.cells_hit[irho,iphi] = 1
                    deflect = np.random.normal(0.,self.sigmaMS)
        return deflect

    def getHits(self):
        ihit=0
        for irho in range(0, self.Nrho):
            for iphi in range(0,self.Nphi[0]):
                if(self.cells_hit[irho,iphi] == 1):
                    self.history = self.history.append(pd.DataFrame({'particle':self.hit_particle[irho,iphi], 'hit':[ihit], 'layer':[irho], 'x':self.cells_x[irho,iphi], 'y':self.cells_y[irho,iphi]}), ignore_index=True)
                    ihit+=1
        self.history=self.history.sort(['particle','layer','hit'])

        return self.history


class Simulator(object):
    def __init__(self):
        self.p = Particle([0,0,0], [0,0,0])
        self.detector = Detector()
        self.precision = 100


    def force(self,position, momentum):
        #        g = 0.5
        #        acc = -g * position / pow(np.linalg.norm(position),3) # gravitational force
        b = 1./self.precision
        # This is non relativistic!
        acc = - np.cross(momentum, [0,0,b])
        return acc


    def propagate(self,x=[], v=[], step = 1, id = 0):
        #        print "New planet"
        self.p = Particle(x,v,id)

        for t in range(0,self.precision):
            acceleration = self.force(self.p.position,self.p.momentum)
            self.p.update(acceleration, t, self.detector, self.precision, stephit = step)
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

