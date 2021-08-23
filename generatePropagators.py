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
cacheSize = 2048

#w,Z,m2,lam2 = floats
#N1 = int
#ABCs = [[A1,B1,C1],[A2,B2,C2]...[AN1,BN1,CN1]] list of list of floats
@lru_cache(maxsize=cacheSize)
def rho1(w2,Z,m2,lam2,N1,ABCs):
    gamma = 13/22
    # w2 = w ** 2
    #Calculate the ksum
    ksum = 0
    for k in range(N1):
        ksum += (ABCs[k][0]*w2) / (ABCs[k][1]*w2 + ((ABCs[k][2] - w2) ** 2))
    #Multiply the ksum with -Z/ln(...)
    ln = np.log((w2 + m2) / lam2) ** (1 + gamma)
    mult = -Z / ln
    
    return mult * ksum


#Calculates the pdf of a normal distribution
#Avoids huge exponents by cutting off the distribution after zlim
def custompdf(w,mean,std):
    zlim = 15
    z = (w-mean)/std
    if z < zlim and z > -zlim:
        return np.exp(-(w-mean)**2/(std))
    else:
        return 0
    
#w = float
#N2,N3 = ints
#gabl = [[gam1,alfa1,beta1],[gam2,alfa2,beta2]...[gamN2,alfaN2,betaN2]] list of list of floats
@lru_cache(maxsize=cacheSize)
def rho2(w2,N2,gabl,N3,gabi):
    # w2 = w ** 2
    lsum = 0
    #Sum of normal distributions
    #If w2 > ~5.2: underflow in exp
    for l in range(N2):
        # lsum += np.sqrt(2*np.pi) * gabl[l][0] * norm.pdf(w2,loc=gabl[l][1],scale=gabl[l][2]/2)
        # lsum += gabl[l][0] * (np.exp(-((w2 - gabl[l][1])**2)/gabl[l][2]))        
        lsum += gabl[l][0] * custompdf(w2,gabl[l][1],gabl[l][2])

    
    isum = 0
    #Sum of derivatives of normal distributions
    for i in range(N3):        
        # isum += np.sqrt(2*np.pi) * gabi[i][0] * norm.pdf(w2,loc=gabi[i][1],scale=gabi[i][2]/2)
        # isum += gabi[i][0] * w2 * (np.exp(-((w2 - gabi[i][1])**2)/gabi[i][2]))        
        isum += gabi[i][0] * w2 * custompdf(w2,gabi[i][1],gabi[i][2])

        
    return lsum + isum

@lru_cache(maxsize=cacheSize)
def rho(w,Z,m2,lam2,N1,ABCs,N2,gabl,N3,gabi):
    return rho1(w,Z,m2,lam2,N1,ABCs) + rho2(w,N2,gabl,N3,gabi)

#Correct form under the integral
def rhoint(w,Z,m2,lam2,N1,ABCs,N2,gabl,N3,gabi,p2):
    # return rho(w,Z,m2,lam2,N1,ABCs,N2,gabl,N3,gabi)/(w**2 + p2)
    return rho(w,Z,m2,lam2,N1,ABCs,N2,gabl,N3,gabi)/(w + p2)

    
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
        jsum += nom/denom
    return jsum

#Calculates the propagator out of the given parameters for the spectral
# density function and complex conjugate poles
def calcPropagator(Z,m2,lam2,N,abcds,N1,ABCs,N2,gabl,N3,gabi,sigma,p2):
    try:
        #Integrate over w^2
        res = integrate.quad(rhoint,0.01+sigma,5,args=(Z,m2,lam2,N1,ABCs,N2,gabl,N3,gabi,p2))[0] \
              + poles(p2,N,abcds)
    except:
        print(Z,m2,lam2,ABCs,gabl,gabi,p2)
        raise RuntimeError("Error when calculating integral for propagator")
        
    return res
        
"""
TODO: Check correctness of spectral densities (some -> inf at zero)
"""
        

if __name__ == "__main__":
    
    pstart = 0.01
    pend = 10
    nbrPoints = 100
    # ps = np.geomspace(pstart,pend,nbrPoints)
    ps = np.linspace(pstart,pend,nbrPoints)
    #Lin space for density function
    nbrWs = 200
    ws = np.linspace(0.01,10,nbrWs)
    
    
    """
    #######################################
    Input Parameters:
    #######################################
    """
    #Sigma in [0,1]
    sigmas = np.linspace(0,1,2)
    #Z in [1,10]
    # Z = 1
    Zs = np.linspace(1,10,2)
    #m2 in [0,5]
    # m2 = 3
    m2s = np.linspace(2,5,2)
    #lam2 in ]0,5] 
    # lam2 = 2
    lam2s = np.linspace(1,4,2)
    #Extra constraint: w^2 + m^2 / lam2 > 1
    #Extra constraint: lam2 > 0
    #Extra constraint: Beta L/i > 0
    
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
    # B = 0 caused large propagator in:
    # 5.0 1.0 0.5 ((-5, 5, 2, -2),) ((-3, 0, 3),) ((-5, 5, 1),) ((-2, 3, 5),)
    # C = 0 caused large propagator in: 
    # 5.0 5.0 4.5 ((-5, 5, 2, -2),) ((-3, 1, 0),) ((-2, 5, 5),) ((-2, 3, 5),)
    # T = list(itertools.product(*[[-3,0,3],[-3,1,3],[-3,1,3]]))
    T = list(itertools.product(*[[-3,3],[-3,3],[-3,3]]))
    noSingularitiesT = []
    for ABC in T:
        singularityFound = False
        for w in ws:
            if checkSingularity(w,ABC[1],ABC[2]):
                singularityFound = True
        if not singularityFound:
            noSingularitiesT.append(ABC)
            # print("Kept", ABC[1],ABC[2])
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
    # T = list(itertools.product(*[[-5,-2,2,5],[1,3,5],[1,3,5]]))
    # T = list(itertools.product(*[[-5,1,5],[1,3,5],[1,3,5]]))
    T = list(itertools.product(*[[-3,3],[1,5],[1,5]]))
    
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
    
    #Complex poles:
    # #a,b,c,d in [-5,5] 
    # T = list(itertools.product([-3,3],repeat=4))
    T = list(itertools.product(*[[-3,3],[1,5],[-3,3],[1,5]]))
    abcdNs = []
    for N in Ns:
        abcds = list(itertools.product(T,repeat=N))
        # print("poles:",N,"len:",len(abcds))
        abcdNs.append(abcds)
        
        
    
    #Cartesian products become very large, very quickly...
    #50 per second, 200000 total = 1 hour data generation
    
    
    #Iterator over all possible combinations of parameters
    #Currently only 1 pole,ABC,gami and gaml
    """
    TODO: more poles, ABCs in data generation 
    TODO: Check physicality of spectral density function/propagator
    """
    
    names = ["sig","Z","m2","lam2","abcd","ABC","gabl","gabi"]
    sub = [sigmas,Zs,m2s,lam2s,abcdNs[0],ABCNs[0],gablNs[0],gabiNs[0]]
    
    totalSize = 1
    print("Parameters:")
    for i in range(len(sub)):
        totalSize = totalSize * len(sub[i])
        print(names[i],sub[i])
    print("Size of all data points:",totalSize)
    iterator = itertools.product(*sub)
    
    
    
    path = "C:/Users/Thibault/Documents/Universiteit/Honours/Deel 2, interdisciplinair/Code/NN/Datasets/"
        
    #Write data to these files (first deletes old ones)
    propTrain_csv = path+'DTrainingRaw.csv'
    if os.path.exists(propTrain_csv):
        os.remove(propTrain_csv)
    propValid_csv = path+'DValidationRaw.csv'
    if os.path.exists(propValid_csv):
        os.remove(propValid_csv)
    propTest_csv = path+'DTestRaw.csv'
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
    params_csv = path+'params.csv'
    if os.path.exists(params_csv):
        os.remove(params_csv)
        
        
    
    print("Generating data...")
    
    start = time.time()
    
    random.seed(64)
    
    counter = 0
    counterPos = 0
    counterNeg = 0
    counterAccepted = 0
    propTempList = []
    rhoTempList = []
    paramTempList = []
    
    desiredDataSize = 10000
    print("Desired training data size:",desiredDataSize)
    print("Percentage of training set selected:",round(100*desiredDataSize/totalSize,2),"%")
    
    # Print iterations progress
    def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        # Print New Line on Complete
        if iteration == total: 
            print()
    
    #Iterate over parameters (too large to fit entirely in memory)  
    printProgressBar(0, desiredDataSize, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for item in iterator:
        #Random sampling over the resulting parameter combinations
        if random.random() <= 4.5*1.5*3*2*desiredDataSize/totalSize:
            # print(item)
            
            sigma = item[0]
            Z = item[1]
            m2 = item[2]
            lam2 = item[3]
            #Possible singularity if m2<lam2 -> skip this iteration
            if m2 < lam2:
                continue
            abcds = item[4]
            N = len(abcds)
            ABCs = item[5]
            N1 = len(ABCs)
            gabls = item[6]
            N2 = len(gabls)
            gabis = item[7]
            N3 = len(gabis)
            
            #Calculate propagator (non negative)
            
            #Add normal noise to the values:
            noiseLevel = 1e-5
            
            dps = []
            negativeProp = False
            for p in ps:
                prop = calcPropagator(Z,m2,lam2,N,abcds,N1,ABCs,N2,gabls,N3,gabis,sigma,p)
                if prop < 0:
                    negativeProp = True
                    break
                dps.append(prop + np.random.normal(scale=noiseLevel))
                # dps.append(calcPropagator(Z,m2,lam2,N,abcds,N1,ABCs,N2,gabls,N3,gabis,p))
            #Propagators are >= 0, skip if it is negative anywhere
            if negativeProp: 
                counterNeg += 1
                continue
            else:
                counterPos += 1
            
            if dps[0] > 1000:
                print(Z,m2,lam2,ABCs,gabls,gabis,abcds)
                raise RuntimeError("Propagator -> inf at 0, check parameters ^")
            
                
            #Calculate density function (already in cache so should be fast)
            rhos = []
            for w in ws:
                rhos.append(rho(w,Z,m2,lam2,N1,ABCs,N2,gabls,N3,gabis))
            
            
            #Check derivative equation
            dp0 = dps[0] - poles(ps[0],N,abcds)
            dp1 = dps[1] - poles(ps[1],N,abcds)
            derivativeProp = (dp1 - dp0)/(ps[1]-ps[0])
            derivativeRho = (rhos[1] - rhos[0])/(ws[1]-ws[0])
            idealDerivative = -derivativeRho/2
            if derivativeProp < idealDerivative - 0.5 or \
               derivativeProp > idealDerivative + 0.5:
                continue
            
            # print(derivativeProp,idealDerivative)
            
            #Add poles to the list: (1 value at a time, add zeroes if not enough poles)
            for polecouple in abcds:
                for poleval in polecouple:
                    rhos.append(poleval)
            nbrOfMissingPoles = 3 - len(abcds)
            missingPoles = 4 * nbrOfMissingPoles * [0]
            for missingPole in missingPoles:
                rhos.append(missingPole)
                
            
            propTempList.append(dps)
            rhoTempList.append(rhos)
            paramTempList.append([Z,m2,lam2,abcds,ABCs,gabls,gabis])
                
            counter += 1
            
            
            #Write to csv in batches
            batchSize = 400
            if counter % batchSize == 0:
                # print("Positive prop:", counterPos)
                # print("Negative prop:", counterNeg)
                
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
                propValidation = propTempList[::8]
                propTest       = propTempList[4::8]
                
                rhoTraining = np.array(rhoTempList)[np.mod(np.arange(batchSize),4) != 0]
                rhoValidation = rhoTempList[::8]
                rhoTest       = rhoTempList[4::8]
                
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
                
                paramsdf = pd.DataFrame(paramTempList)
                paramsdf.to_csv(params_csv,index=False,header=False,mode='a')
                
                
                
                #Reset templists
                propTempList = []
                rhoTempList = []
                paramTempList = []
                
                pointsPerSecond = round(batchSize/(time.time()-start),2)
                minsRemaining = round((desiredDataSize - counter*4/3) / (pointsPerSecond*60),2)
                # print("\nWrote to file")
                # print("{} points done".format(counterPos))
                # print("{} points per second".format(round(batchSize/(time.time()-start),2)))
                # print("{}% finished\n".format(round(100*counter/(desiredDataSize*1.5),2)))
                printProgressBar(counter*4/3, desiredDataSize, prefix = 'Progress:', \
                                 suffix = 'Complete,{} mins remaining, {} per sec'.format(minsRemaining,pointsPerSecond), length = 30)
                            
                #Stop when desired data size is reached
                if counter*4/3 >= desiredDataSize:
                    break
                start = time.time()
                

            
            #Calculate density function
            #Tradeoff: calculate and store it (more storage usage), faster eval in NN
            #          or calculate during eval in NN -> slower NN eval, less heavy data
            # rhos = []
            # for w in ws:
            #     rhofile.write(str(rho(w,Z,m2,lam2,N1,ABCs,N2,gabls,N3,gabis)))
                # rhos.append(rho(w,Z,m2,lam2,N1,ABCs,N2,gabls,N3,gabis))
            
        
    
    nbrOfTrainingPoints = round(counter*3/4 if counter % 400 == 0 else (counter - counter % 400)*3/4)
    nbrOfValidationPoints = round(counter/8 if counter % 400 == 0 else (counter - counter % 400)/8)
    print("\nData generation succesfull")
    print("{} training points".format(nbrOfTrainingPoints))
    print("{} validation points".format(nbrOfValidationPoints))
    print("{} testing points".format(nbrOfValidationPoints))
        
    nbrOfNormalDists = 66
    nbrOfPoles = 3
    maxDegreeOfLegFit = 30
    nbrOfPCAcomponents = 15
    #Write parameters to file:
    params = open("inputParameters.py","w")
    params.write("nbrOfNormalDists = {}\n".format(nbrOfNormalDists))
    params.write("nbrOfPoles = {}\n".format(nbrOfPoles))
    params.write("maxDegreeOfLegFit = {}\n".format(maxDegreeOfLegFit))
    params.write("nbrOfPCAcomponents = {}\n".format(nbrOfPCAcomponents))
    params.write("trainingPoints = {}\n".format(nbrOfTrainingPoints))
    params.write("validationPoints = {}\n".format(nbrOfValidationPoints))
    params.write("pstart = {}\n".format(pstart))
    params.write("pend = {}\n".format(pend))
    params.write("nbrPoints = {}\n".format(nbrPoints))
    params.write("nbrWs = {}\n".format(nbrWs))
         
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
    



