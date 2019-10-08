import pandas as pd
import numpy as np

from formulate import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Score_assignment
import Score_cluster
import Score_physics
from Plotting import *

from scipy.optimize import curve_fit

def rotateToQuadrant(x,y):
    seedx=x[1]-x[0]
    seedy=y[1]-y[0]
    phi0=np.arctan2(y[0],x[0])
    phi1=np.arctan2(seedy,seedx)
    phi=phi1
    c, s = np.cos(-phi), np.sin(-phi)
    xr=c*x-s*y
    yr=s*x+c*y
    print(phi)
    return xr,yr

def func(x, r, vx, vy):
    return r - np.sqrt(r**2-(x-vx)**2) + vy





df = pd.read_csv("result_truth.csv")

ievent = 0
iparticle=6
itrack =5

df_event=df.loc[df['event']==ievent]
df_particle=df_event[df['particle']==iparticle]
df_track=df_event[df['track']==itrack]


x=df_track['x'].values
y=df_track['y'].values

plt.scatter(x,y)
plt.show()

popt, pcov = curve_fit(func, xr, yr, p0=[200.,0.,0.], bounds=([5.,-25.,-25.], [1000., 25, 25]))

xp=np.arange(5.,80.,5.)
plt.plot(xp,func(xp,popt[0],popt[1],popt[2]),'k')
plt.scatter(xr,yr)
plt.show()

popt


