import numpy as np
import pandas as pd

class Tracking(object):

    def __init__(self):
        self.depth = 5  
        pass


    def convert(self, orbit = pd.DataFrame()):

        timedata=orbit.drop("t",axis=1)
        for i in range(0,self.depth):
#            newcolumn=timedata[i+1:len(timedata)-self.depth+i].reset_index()["x"]

            newcolumn=timedata[i+1:]
            newcolumn=newcolumn.reset_index()["position"]


            timedata = pd.concat([timedata.copy(),newcolumn.copy()],axis=1,ignore_index=True)

        return timedata








