
from scipy.optimize import curve_fit

from Transform import *

def circular_path(x, r, vx, vy):
    return r - np.sqrt(r**2-(x-vx)**2) + vy

class TrackFitter():
    def __init__(self,B):
        self.B = B
        pass
    
    def fit(self,x,y):
        xr,yr,phi=rotateArrayToQuadrant(x,y);
        p, cov = curve_fit(circular_path, xr, yr,
                           p0=[200.,0.,0.],
                           bounds=([5.,-25.,-25.],
                                   [1000., 25, 25]
                                   )
                           )

        r = p[0]
        pt = self.B*r
        vx = p[1]
        vy = p[2]
        chg = 1
        return  pt,phi,vx,vy,chg



