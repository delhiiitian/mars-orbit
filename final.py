# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats.mstats import gmean 


df = pd.read_csv("datafile.csv")

sL = []
for i in range(12):
    sL.append(np.radians((df.ZodiacIndex[i]*30)+(df.Degree[i])+(df.Minute[i]/60)+(df.Second[i]/3600)))

aL = []
for i in range(12):
    aL.append(np.radians((df.ZodiacIndexAverageSun[i]*30)+(df.DegreeMean[i])+(df.MinuteMean[i]/60)+(df.SecondMean[i]/3600)))

def radius(x,y):
    return np.sqrt(x**2 + y**2)


def optm(d):
    p = []
    for i in range(12):
        A = np.array([[np.tan(aL[i]),-1],[np.tan(sL[i]),-1]])
        B = np.array([-d*np.tan(aL[i]),1])
        p.append(np.linalg.solve(A,B))
    p= np.asmatrix(p)
    r = []
    x = p[:,0]
    y = p[:,1]
    for i in range(12):
        r.append(radius(p[i,0],p[i,1]))
    return x,y,r


def sse(a,L):
    s = 0.0
    for i in range(len(L)):
        s = s+(a - L[i])**2
    return s/len(L)


def optF(d):
    x,y,r = optm(d)
    avgR = sum(r)/len(r)
    airM = sse(avgR,r)
    geoM = gmean(r)
    return np.log(airM/geoM)


minT = minimize_scalar(optF)
print(minT.x)


x,y,r = optm(minT.x)
rav = sum(r)/len(r)
ax = plt.subplots()
plt.plot(x,y,'r.')
plt.Circle((0,0), rav, color='g',label="Mars Path")
plt.show()

