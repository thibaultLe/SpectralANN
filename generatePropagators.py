# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:18:08 2021

@author: Thibault
"""

import numpy as np
from scipy import integrate
import itertools
import pandas as pd
import os
from functools import lru_cache
import time
import random
import matplotlib.pyplot as plt

np.seterr('raise')
cacheSize = 1024

#w,Z,m2,lam2 = floats
#N1 = int
#ABCs = [[A1,B1,C1],[A2,B2,C2]...[AN1,BN1,CN1]] list of list of floats
@lru_cache(maxsize=cacheSize)
def rho1(w,Z,m2,lam2,N1,ABCs):
    gamma = 13/22
    w2 = w ** 2
    #Calculate the ksum
    ksum = 0
    for k in range(N1):
        ksum += (ABCs[k][0]*w2) / (ABCs[k][1]*w2 + ((ABCs[k][2] - w2) ** 2))
    #Multiply the ksum with -Z/ln(...)
    ln = np.log((w2 + m2) / lam2) ** (1 + gamma)
    mult = -Z / ln
    
    return mult * ksum

#w = float
#N2,N3 = ints
#gabl = [[gam1,alfa1,beta1],[gam2,alfa2,beta2]...[gamN2,alfaN2,betaN2]] list of list of floats
@lru_cache(maxsize=cacheSize)
def rho2(w,N2,gabl,N3,gabi):
    w2 = w ** 2
    lsum = 0
    #Sum of normal distributions
    #If w2 > ~5.2: underflow in exp
    for l in range(N2):
        #Still needs a sqrt in the stddev
        # lsum += np.sqrt(2*np.pi) * gabl[l][0] * norm.pdf(w2,loc=gabl[l][1],scale=gabl[l][2]/2)
        lsum += gabl[l][0] * (np.exp(-((w2 - gabl[l][1])**2)/gabl[l][2]))
    
    isum = 0
    #Sum of derivatives of normal distributions
    for i in range(N3):        
        # isum += np.sqrt(2*np.pi) * gabi[i][0] * norm.pdf(w2,loc=gabi[i][1],scale=gabi[i][2]/2)
        isum += gabi[i][0] * w2 * (np.exp(-((w2 - gabi[i][1])**2)/gabi[i][2]))
        
    return lsum + isum

@lru_cache(maxsize=cacheSize)
def rho(w,Z,m2,lam2,N1,ABCs,N2,gabl,N3,gabi):
    return rho1(w,Z,m2,lam2,N1,ABCs) + rho2(w,N2,gabl,N3,gabi)

#Correct form under the integral
def rhoint(w,Z,m2,lam2,N1,ABCs,N2,gabl,N3,gabi,p2):
    return rho(w,Z,m2,lam2,N1,ABCs,N2,gabl,N3,gabi)/(w**2 + p2)

    
@lru_cache(maxsize=cacheSize)
def poles(p2,N,abcds):
    jsum = 0
    #Using alternate but equivalent form of pole sum (without i)
    for j in range(N):
        # nom = 2 * ( ajs[j]  *   (cjs[j] + p2)      + bjs[j]   *     djs[j])
        nom = 2 * ( abcds[j][0] * (abcds[j][2] + p2) + abcds[j][1] * abcds[j][3])
        # denom=(cjs[j] ** 2)    + (2 * cjs[j] * p2) 
        #     + (djs[j] ** 2) +      (p2 ** 2)
        denom = (abcds[j][2] ** 2) + (2 * abcds[j][2] * p2) + \
                (abcds[j][3] ** 2) + (p2 ** 2)
        #Extra constraint: not (d == 0 and p2 == -c)
        jsum += nom/denom
    return jsum

#Calculates the propagator out of the given parameters for the spectral
# density function and complex conjugate poles
def calcPropagator(Z,m2,lam2,N,abcds,N1,ABCs,N2,gabl,N3,gabi,p2):
    return integrate.quad(rhoint,0.01,5,args=(Z,m2,lam2,N1,ABCs,N2,gabl,N3,gabi,p2))[0] \
        + poles(p2,N,abcds)
        
"""
TODO: implement sigma (cutoff for integral)
TODO: Check correctness of spectral densities (some -> inf at zero)
"""
        
        
#Log space for propagator
pstart = 0.1
pend = 10
nbrPoints = 100
# ps = np.geomspace(pstart,pend,nbrPoints)
ps = np.linspace(pstart,pend,nbrPoints)
#Lin space for density function
ws = np.linspace(0.01,5,200)


#Parameters:
#Z in [0,10]
# Z = 1
Zs = np.linspace(1,8,2)
#m2 in [0,5]
# m2 = 3
m2s = np.linspace(1,4,2)
#lam2 in [0,5]
# lam2 = 2
lam2s = np.linspace(0.5,4,2)
#Extra constraint: w^2 + m^2 / lam2 > 1

#N1,N2,N3 in [1,5] (int)
# N1s = [1,3]
Ns  = [i for i in range(1,5,2)]
N1s = [i for i in range(1,5,2)]
N2s = [i for i in range(1,5,2)]
N3s = [i for i in range(1,5,2)]

def checkSingularity(w,B,C):
    w2 = w**2
    if B*w2 + C**2 -2*C*w2 + w2**2 <= 10**(-7):
        return True
    else:
        return False

# A,B,Cks in [-5,5]     
T = list(itertools.product(*[[-3,3],[-4,4],[-3,3]]))
noSingularitiesT = []
for ABC in T:
    singularityFound = False
    for w in ws:
        if checkSingularity(w,ABC[1],ABC[2]):
            singularityFound = True
    if not singularityFound:
        noSingularitiesT.append(ABC)
    else:
        print("Removed B={},C={} due to possible singularity".format(ABC[1],ABC[2]))
    
# print(T)
ABCNs = []
for N1 in N1s:
    ABCs = list(itertools.product(noSingularitiesT,repeat=N1))        
    ABCNs.append(ABCs)

   

# #alfa,beta in [0,5]
# #gamma in [-5,5]
# gaml = N2 * [1]
# alfal = N2 * [1]
# betal = N2 * [1]
T = list(itertools.product(*[[-3,3],[1,3],[1,3]]))

gablNs = []
for N2 in N2s:
    gabls = list(itertools.product(T,repeat=N2))
    # print(gabls)
    # print("L funcs:",N2,"len:",len(gabls))
    gablNs.append(gabls)

# gami = N3 * [1]
# alfai = N3 * [1]
# betai = N3 * [1]
gabiNs = gablNs.copy()

# #N in [0,3] (int)
# N = 1

# #a,b,c,d in [-5,5] 
# A,B,Cks in [-5,5]     
T = list(itertools.product([-3,3],repeat=4))
abcdNs = []
for N in Ns:
    abcds = list(itertools.product(T,repeat=N))
    # print(abcds)
    # print("poles:",N,"len:",len(abcds))
    abcdNs.append(abcds)
# ajs = N * [1]
# bjs = N * [1]
# cjs = N * [1]
# djs = N * [1]

# for i in range(len(Ns)):
#     for j in range(len(N1s)):
#         for k in range(len(N2s)):
#             for l in range(len(N3s)):
#                 sub = [Z,m2,lam2,abcdNs[i],ABCNs[j],gablNs[k],gabiNs[l]]
#                 listOfAllParams.append(list(itertools.product(*sub)))
                
# print(len(Zs)*len(m2s)*len(lam2s)*len(abcdNs[1])*len(ABCNs[1]*len(gablNs[1])*len(gabiNs[0])))
#Cartesian products become very large, very quickly...
#CSV :                 2400 points -> 9MB

#30 per second, 100000 total = 1 hour data generation

#0,0,0,0 -> len =         65536 
#1,0,0,0 -> len =      16777216 
#1,1,0,0 -> len =    1073741824
#1,1,1,0 -> len =   68719476736
#1,1,1,1 -> len = 4398046511104

#Iterator over all possible combinations of parameters
#Currently only 1 pole,ABC,gami and gaml
"""
TODO: more poles, ABCs in data generation 
"""
sub = [Zs,m2s,lam2s,abcdNs[0],ABCNs[0],gablNs[0],gabiNs[0]]
iterator = itertools.product(*sub)



path = "C:/Users/Thibault/Documents/Universiteit/Honours/Deel 2, interdisciplinair/Code/NN/Datasets/"
    
#Write data to these files (first deletes old ones)
propTrain_csv = path+'DTraining.csv'
if os.path.exists(propTrain_csv):
    os.remove(propTrain_csv)
propValid_csv = path+'DValidation.csv'
if os.path.exists(propValid_csv):
    os.remove(propValid_csv)
propTest_csv = path+'DTest.csv'
if os.path.exists(propTest_csv):
    os.remove(propTest_csv)
rhoTrain_csv = path+'rhoTraining.csv'
if os.path.exists(rhoTrain_csv):
    os.remove(rhoTrain_csv)
rhoValid_csv = path+'rhoValidation.csv'
if os.path.exists(rhoValid_csv):
    os.remove(rhoValid_csv)
rhoTest_csv = path+'rhoTest.csv'
if os.path.exists(rhoTest_csv):
    os.remove(rhoTest_csv)
    
    

print("Generating data...")

start = time.time()

random.seed(64)

counter = 0
propTempList = []
rhoTempList = []

#Iterate over parameters (too large to fit entirely in memory)
for item in iterator:
    if random.random() <= 0.015:
        # print(item)
        
        
        Z = item[0]
        m2 = item[1]
        lam2 = item[2]
        #Possible singularity if m2<lam2 -> skip this iteration
        if m2 < lam2:
            continue
        abcds = item[3]
        N = len(abcds)
        ABCs = item[4]
        N1 = len(ABCs)
        gabls = item[5]
        N2 = len(gabls)
        gabis = item[6]
        N3 = len(gabis)
        
        #Calculate propagator
        dps = []
        for p in ps:
            dps.append(calcPropagator(Z,m2,lam2,N,abcds,N1,ABCs,N2,gabls,N3,gabis,p))
        
        #Calculate density function (already in cache so should be fast)
        rhos = []
        for w in ws:
            rhos.append(rho(w,Z,m2,lam2,N1,ABCs,N2,gabls,N3,gabis))
        
            
        propTempList.append(dps)
        rhoTempList.append(rhos)
            
        counter += 1
        
        
        #Write to csv in batches
        batchSize = 200
        if counter % batchSize == 0:
            # print(rho1.cache_info())
            
            # plt.figure()
            # plt.plot(ps,dps)
            # plt.title("Propagator")
            # plt.figure()
            # plt.title("Spectral density")
            # plt.plot(ws,rhos)
            
            #3 out of every 4 points to the training set
            #1 out of every 4 points alternatively to validation and testing
            #75% train, 12.5% validation, 12.5% testing
            propTraining   = np.array(propTempList)[np.mod(np.arange(batchSize),4) != 0]
            propValidation = propTempList[4::8]
            propTest       = propTempList[::8]
            
            rhoTraining = np.array(rhoTempList)[np.mod(np.arange(batchSize),4) != 0]
            rhoValidation = rhoTempList[4::8]
            rhoTest       = rhoTempList[::8]
            
            propTraindf = pd.DataFrame(propTraining)
            propTraindf.to_csv(propTrain_csv,index=False,header=False,mode='a')
            
            propValiddf = pd.DataFrame(propValidation)
            propValiddf.to_csv(propValid_csv,index=False,header=False,mode='a')
            
            propTestdf = pd.DataFrame(propTest)
            propTestdf.to_csv(propTest_csv,index=False,header=False,mode='a')
            
            rhoTraindf = pd.DataFrame(rhoTraining)
            rhoTraindf.to_csv(rhoTrain_csv,index=False,header=False,mode='a')
            
            rhoValiddf = pd.DataFrame(rhoValidation)
            rhoValiddf.to_csv(rhoValid_csv,index=False,header=False,mode='a')
            
            rhoTestdf = pd.DataFrame(rhoTest)
            rhoTestdf.to_csv(rhoTest_csv,index=False,header=False,mode='a')
            
            
            
            #Reset templist
            propTempList = []
            rhoTempList = []
            
            print("\nWrote to file")
            print("{} points per second".format(round(batchSize/(time.time()-start),2)))
            print("{} points done".format(counter))
            start = time.time()
        
        #Calculate density function
        #Tradeoff: calculate and store it (more storage usage), faster eval in NN
        #          or calculate during eval in NN -> slower NN eval, less heavy data
        # rhos = []
        # for w in ws:
        #     rhofile.write(str(rho(w,Z,m2,lam2,N1,ABCs,N2,gabls,N3,gabis)))
            # rhos.append(rho(w,Z,m2,lam2,N1,ABCs,N2,gabls,N3,gabis))
        
    

print("\nData generation succesfull")
nbrOfTrainingPoints = round(counter*3/4 if counter % 200 == 0 else (counter - counter % 200)*3/4)
print("{} training points".format(nbrOfTrainingPoints))
nbrOfValidationPoints = round(counter/8 if counter % 200 == 0 else (counter - counter % 200)/8)
print("{} validation/testing points".format(nbrOfValidationPoints))
    
nbrOfNormalDists = 64
nbrOfPoles = 2
#Write parameters to file:
params = open("inputParameters.py","w")
params.write("nbrOfNormalDists = {}\n".format(nbrOfNormalDists))
params.write("nbrOfPoles = {}\n".format(nbrOfPoles))
params.write("trainingPoints = {}\n".format(nbrOfTrainingPoints))
params.write("validationPoints = {}\n".format(nbrOfValidationPoints))
params.write("pstart = {}\n".format(pstart))
params.write("pend = {}\n".format(pend))
params.write("nbrPoints = {}\n".format(nbrPoints))
     
params.close()

# print(dps)
# plt.plot(ps,dps,"o")
# plt.figure()
# plt.plot(ws,rhos)
#     counter += 1
#     file.write(str(item))
# file.close()

# a = [Z,m2,ABCNs[1]]
# print(list(itertools.product(*a)))
# print(len(listOfAllParams))


# ps = np.linspace(0.01,5,200)

# dpswithpoles = []
# for p in ps:
#     #Assume p's are p^2's
#     # dpswithpoles.append(integrate.quad(rhoint,0.01,5,p)[0] + poles(p))
#     dpswithpoles.append(calcPropagator(Z,m2,lam2,N,abcds,N1,ABCs,N2,gabl,N3,gabi,p))




