# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 17:51:03 2021

@author: Thibault
"""

import numpy as np
import scipy.stats as stats
import time
import matplotlib.pyplot as plt

"""
Timing difference between iteratively evaluating pdf's and using pdf's
"""


ws = np.linspace(0.01,5,200)

#For each test sample
reconstructedDensitiesList = []

nbrOfNormalDists = 2
nbrOfPoles = 2
predicData = [[0.01,0.1,2,1,0.1,1]]

_norm_pdf_C = np.sqrt(2*np.pi)
zlim = 37

#Calculates the pdf of a normal distribution
def custompdf(w,mean,std):
    z = (w-mean)/std
    if z < zlim and z > -zlim:
        return np.exp(-(w-mean)**2/(2.0*(std**2))) / (_norm_pdf_C*std)
    else:
        return 0

start = time.time()
for i in range(len(predicData)):
    rhoReconstructed = []
    #Reconstruct the spectral density function
    for distr in range(nbrOfNormalDists):
        #Mean, standard deviation and weight of normal distribution
        meani = predicData[i][3*distr]
        #Stddev has to be positive
        stdi = abs(predicData[i][3*distr + 1])
        wi = predicData[i][3*distr + 2]
        
        # print("mean_{}:".format(distr),meani.round(5), \
        #       "   std_{}:".format(distr),stdi.round(5),"   w_{}:".format(distr),wi.round(5))
        
        if (distr == 0):
            for w in ws:
                rhoReconstructed.append(wi*custompdf(w,meani,stdi))
                
        else:            
            for i in range(len(ws)):
                rhoReconstructed[i] = rhoReconstructed[i] + wi*custompdf(ws[i],meani,stdi)

print("{} seconds".format(time.time() - start))
# print(rhoReconstructed)
plt.figure()
plt.plot(ws,rhoReconstructed)


for i in range(len(predicData)):
    rhoCutoff = []
    zlim = 25
    #Reconstruct the spectral density function
    for distr in range(nbrOfNormalDists):
        #Mean, standard deviation and weight of normal distribution
        meani = predicData[i][3*distr]
        #Stddev has to be positive
        stdi = abs(predicData[i][3*distr + 1])
        wi = predicData[i][3*distr + 2]
        
        # print("mean_{}:".format(distr),meani.round(5), \
        #       "   std_{}:".format(distr),stdi.round(5),"   w_{}:".format(distr),wi.round(5))
        
        if (distr == 0):
            for w in ws:
                rhoCutoff.append(wi*custompdf(w,meani,stdi))
                
        else:            
            for i in range(len(ws)):
                rhoCutoff[i] = rhoCutoff[i] + wi*custompdf(ws[i],meani,stdi)

MAE = 0
for i in range(len(ws)):
    MAE += abs(rhoReconstructed[i] - rhoCutoff[i])
print("Z lim: {}, MAE: {}".format(zlim,MAE))

"""Results of MAE vs Zlim cutoff: (max Zlim 37)
    Zlim 5 : 5e-5
    Zlim 10: 1e-22
    Zlim 20: 1e-88
    Zlim 25: 1e-138
"""