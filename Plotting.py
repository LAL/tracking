import matplotlib.pyplot as plt
import numpy as np

def plotinbins(x,y,binsize,xmax=50):
    x, y = list(zip(*sorted((xMin, np.mean([yVal for a, yVal in zip(x, y) if ((a >= xMin) & (a < xMin+binsize))])) for xMin in range(0, xmax, binsize))))
    return x,y
