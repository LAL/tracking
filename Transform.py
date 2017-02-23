import numpy as np

def polar(hits, rscale = 0.0001):
    ptsnew = np.zeros(hits.shape)
    xy = hits[:,0]**2 + hits[:,1]**2
    ptsnew[:,0] = np.sqrt(xy)*rscale
    ptsnew[:,1] = np.arctan2(hits[:,1], hits[:,0])
    return ptsnew


def rotateArray(x, y, phi):
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



def rotateArrayToQuadrant(x,y):

    seedx=x[1]-x[0]
    seedy=y[1]-y[0]
    phi0=np.arctan2(y[0],x[0])
    phi1=np.arctan2(seedy,seedx)
    phi=phi1
    xr,yr=rotateArray(x,y,-phi)
    return xr,yr,phi



def deltaPhiAbs(phi1,phi0):
    dphi = phi1-phi0
    if(dphi > 2.*np.pi) : dphi -= 2.*np.pi
    if(dphi < 0) : dphi += 2.*np.pi

    return np.fabs(dphi)


