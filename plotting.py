import matplotlib.pyplot as plt
import numpy as np

def plotinbins(x,y):
    x, y = zip(*sorted((xMin, np.mean([yVal for a, yVal in zip(x, y) if ((a >= xMin) & (a < xMin+5))])) for xMin in range(0, 50, 5)))
    return x,y
