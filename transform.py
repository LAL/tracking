import numpy as np

def polar(hits, rscale = 0.0000001):
    ptsnew = np.zeros(hits.shape)
    xy = hits[:,0]**2 + hits[:,1]**2
    ptsnew[:,0] = np.sqrt(xy)*rscale
    ptsnew[:,1] = np.arctan2(hits[:,1], hits[:,0])
    return ptsnew


def rotate(x, y, phi):
    c, s = np.cos(phi), np.sin(phi)
    xr=c*x-s*y
    yr=s*x+c*y

    return xr,yr

def rotate(momentum, phi):
    x,y,z=momentum[0],momentum[1],momentum[2]
    c, s = np.cos(phi), np.sin(phi)
    xr=c*x-s*y
    yr=s*x+c*y
    
    return [xr,yr,z]

def deltaPhiAbs(phi1,phi0):
    dphi = phi1-phi0
    if(dphi > 2.*np.pi) : dphi -= 2.*np.pi
    if(dphi < 0) : dphi += 2.*np.pi

    return np.fabs(dphi)


