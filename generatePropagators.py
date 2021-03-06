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

np.seterr('raise')
cacheSize = 2048
random.seed(64)

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


#Approximates the pdf of a normal distribution (exact but cuts off at z=15)
#Avoids huge exponents by cutting off the distribution after zlim
def custompdf(w,mean,std):
    zlim = 7
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
    for l in range(N2):       
        lsum += gabl[l][0] * custompdf(w2,gabl[l][1],gabl[l][2])

    
    isum = 0
    #Sum of derivatives of normal distributions
    for i in range(N3):           
        isum += gabi[i][0] * w2 * custompdf(w2,gabi[i][1],gabi[i][2])

        
    return lsum + isum

@lru_cache(maxsize=cacheSize)
def rho(w,Z,m2,lam2,N1,ABCs,N2,gabl,N3,gabi):
    return rho1(w,Z,m2,lam2,N1,ABCs) + rho2(w,N2,gabl,N3,gabi)

#Correct form under the integral
def rhoint(w,Z,m2,lam2,N1,ABCs,N2,gabl,N3,gabi,p2):
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
def calcPropagator(Z,m2,lam2,N,abcds,N1,ABCs,N2,gabl,N3,gabi,sigma,p):
    try:
        #Integrate over w^2
        p2 = p**2
        res = integrate.quad(rhoint,0.01+sigma,10,args=(Z,m2,lam2,N1,ABCs,N2,gabl,N3,gabi,p2))[0] \
              + poles(p2,N,abcds)
    except:
        print(Z,m2,lam2,ABCs,gabl,gabi,p2)
        raise RuntimeError("Error when calculating integral for propagator")
        
    return res


#Returns true if the constraint is roughly satisfied (+-0.5)
def constraint15(Z,m2,lam2,N1,ABCs,N2,gabl,N3,gabi,abcds):
    res = integrate.quad(rho,0.01+sigma,10,args=(Z,m2,lam2,N1,ABCs,N2,gabl,N3,gabi))[0]
    
    jsum = 0
    for j in range(N):
        jsum += 2 * abcds[j][0]
    
    res += jsum
    
    if res < 0.5 and res > -0.5:
        return True
    return False


#Returns true if the constraint is satisfied
@lru_cache(maxsize=cacheSize)
def derivativeConstraint(dp0,dp1,rho0,rho1,abcds,N):    
    derivativeRho = (rho1 - rho0)/(ws[1]-ws[0])
    derivativeProp = (dp1 - dp0)/(ps[1]**2-ps[0]**2)
    
    derivativePoles = 0
    for j in range(N):
        a = abcds[j][0]
        b = abcds[j][1]
        c = abcds[j][2]
        d = abcds[j][3]
        #Alternate but equivalent formula:
        derivativePoles += (-2*a*c**2 + 2*a*d**2 + 4*b*c*d)/((c**2 + d**2)**2)
        
    idealDerivative = -np.pi*derivativeRho - derivativePoles
    
    if derivativeProp < idealDerivative - 1 or \
        derivativeProp > idealDerivative + 1:
        return False
    
    return True
        
        

if __name__ == "__main__":
    
    
    path = "C:/Users/Thibault/Documents/Universiteit/Honours/Deel 2, interdisciplinair/Code/NN/Datasets/"
    
    #Desired training data size:
    desiredDataSize = 30000
    
    pstart = 0
    pend = 8.25
    #8.25 -> p = 1 is exactly in ps
    
    nbrPoints = 100
    ps = np.linspace(pstart,pend,nbrPoints)
    nbrWs = 200
    #ws = w**2s: assumes the ws are already squared as they always appear in squared form
    ws = np.linspace(0.01,10,nbrWs)
    
    
    """
    #######################################
    Input Parameters:
    #######################################
    """
    sigmas = np.linspace(0,1,4)
    Zs = np.linspace(1,10,1)
    m2s = np.linspace(2,5,4)
    lam2s = np.linspace(1,4,4)
    
    Ns  = [i for i in range(1,4)]
    N1s = [i for i in range(1,4)]
    N2s = [i for i in range(1,4)]
    N3s = [i for i in range(1,4)]
    
    def checkSingularity(w2,B,C):
        if B*w2 + C**2 -2*C*w2 + w2**2 <= 10**(-7):
            return True
        else:
            return False
    
    T = list(itertools.product(*[[-0.5,-0.16,0.16,0.5],[-5,-1.6,1.6,5],[-5,-1.6,1.6,5]]))
    noSingularitiesT = []
    for ABC in T:
        singularityFound = False
        for w in ws:
            if checkSingularity(w,ABC[1],ABC[2]):
                singularityFound = True
        if not singularityFound:
            noSingularitiesT.append(ABC)
            # print("Kept", ABC[1],ABC[2])
        # else:
        #     print("Removed B={},C={} due to possible singularity".format(ABC[1],ABC[2]))
    
    ABCNs = []
    for N1 in N1s:
        ABCs = list(itertools.combinations_with_replacement(noSingularitiesT,N1))
        sampled_ABCs = []
        for ABC in ABCs:
            if random.random() < 0.0015:
                sampled_ABCs.append(ABC)
        ABCNs.append(sampled_ABCs)
        
    
    #The first parameters in Eq. 5:
    T = list(itertools.product(*[[-0.5,-0.16,0.16,0.5],[1,2.3,3.6,5],[1,2.3,3.6,5]]))
    gablNs = []
    for N2 in N2s:
        gabls = list(itertools.combinations_with_replacement(T,N2))
        sampled_gabls = []
        for gab in gabls:
            if random.random() < 0.0005:
                sampled_gabls.append(gab)
        gablNs.append(sampled_gabls)
    
    #The second parameters in Eq. 5:
    T = list(itertools.product(*[[-0.5,-0.16,0.16,0.5],[1,2.3,3.6,5],[1,2.3,3.6,5]]))
    gabiNs = []
    for N3 in N3s:
        gabis = list(itertools.combinations_with_replacement(T,N3))
        sampled_gabis = []
        for gab in gabis:
            if random.random() < 0.0009:
                sampled_gabis.append(gab)
        gabiNs.append(sampled_gabis)
        
    #Complex poles:
    T = list(itertools.product(*[[-1,-0.33,0.33,1],[0,0.33,0.66,1], \
                     [0.2,0.25,0.3,0.35],[0.3,0.45,0.6,0.75]]))
    abcdNs = []
    for N in Ns:
        abcds = list(itertools.combinations_with_replacement(T,N))
        sampled_abcds = []
        for abcd in abcds:
            if random.random() < 0.000015:
                sampled_abcds.append(abcd)
        
        abcdNs.append(sampled_abcds)
        
    
    
    #Iterator over all possible combinations of parameters
    names = ["sig","Z","m2","lam2","abcd","ABC","gabl","gabi"]
    sub = [sigmas,Zs,m2s,lam2s,abcdNs[2],ABCNs[2],gablNs[2],gabiNs[2]]
    
    
    totalSize = 1
    for i in range(len(sub)):
        totalSize = totalSize * len(sub[i])
        print(names[i],len(sub[i]))
    print("\nMax number of data points:",totalSize)
    iterator = itertools.product(*sub)
    
        
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
    
    
    counter = 0
    counterPos = 0
    counterNeg = 0
    counterAccepted = 0
    propTempList = []
    rhoTempList = []
    paramTempList = []
    
    counterConstraint15True = 0
    counterConstraint15False = 0
    
    print("Desired training data size:",desiredDataSize)
    print("Percentage of training set selected:",round(100*desiredDataSize/totalSize,4),"%")
    
    # Print iterations progress
    def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '???', printEnd = "\r"):
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
    
    
    """
    #######################################
    Calculate spectral function and propagator:
    #######################################
    """
    
    for item in iterator:
        #Random sampling over the resulting parameter combinations
        if random.random() <= 2.15*800*desiredDataSize/totalSize:
            #                 ^ multiplicity factor to account for unlawful combinations
            
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
            
            #Check constraint:
            if not constraint15(Z,m2,lam2,N1,ABCs,N2,gabls,N3,gabis,abcds):
                counterConstraint15False += 1
                continue
            counterConstraint15True += 1
            
            dps = []
            negativeProp = False
            rescaling = calcPropagator(Z,m2,lam2,N,abcds,N1,ABCs,N2,gabls,N3,gabis,sigma,ps[12]) * 1
            
            for i in range(len(ps)):
                prop = calcPropagator(Z,m2,lam2,N,abcds,N1,ABCs,N2,gabls,N3,gabis,sigma,ps[i]) / rescaling
                dps.append(prop)
                
                #Check for negative prop
                if prop < 0:
                    negativeProp = True
                    break
                
                #Check derivative constraint
                if i == 1:
                    rhos0 = rho(ws[0],Z,m2,lam2,N1,ABCs,N2,gabls,N3,gabis)/rescaling
                    rhos1 = rho(ws[1],Z,m2,lam2,N1,ABCs,N2,gabls,N3,gabis)/rescaling
                    dp0 = dps[0] 
                    dp1 = dps[1]
                    
                    if not derivativeConstraint(dp0,dp1,rhos0,rhos1,abcds,N):
                        negativeProp = True
                        break
                
            #Skip if propagator is negative anywhere or derivative constraint is not satisfied
            if negativeProp: 
                counterNeg += 1
                continue
            else:
                counterPos += 1
            
                
            #Calculate density function (already in cache so should be fast)
            rhos = []
            for w in ws:
                rhos.append(rho(w,Z,m2,lam2,N1,ABCs,N2,gabls,N3,gabis)/rescaling)
            
        
            #Add poles to the list: (1 value at a time, add zeroes if not enough poles)
            for polecouple in abcds:
                for i in range(len(polecouple)):
                    #rescale residues, not poles
                    if i < 2:
                        rhos.append(polecouple[i]/rescaling)
                    else:
                        rhos.append(polecouple[i])
                    
            nbrOfMissingPoles = 3 - len(abcds)
            missingPoles = 4 * nbrOfMissingPoles * [0]
            for missingPole in missingPoles:
                rhos.append(missingPole)
            
            #Add sigma at end of spectral function
            rhos.append(sigma)
                
            propTempList.append(dps)
            rhoTempList.append(rhos)
            paramTempList.append([sigma,Z,m2,lam2,abcds,ABCs,gabls,gabis])
                
            counter += 1
            
            #Write to csv in batches
            batchSize = 400
            if counter % batchSize == 0:                
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
                minsRemaining = round((desiredDataSize - counter*3/4) / (pointsPerSecond*60),2)
                printProgressBar(counter*3/4, desiredDataSize, prefix = 'Progress:', \
                                 suffix = 'Complete,{} mins remaining, {} per sec'.format(minsRemaining,pointsPerSecond), length = 30)
                            
                #Stop when desired data size is reached
                if counter*3/4 >= desiredDataSize:
                    print("Reached desired data size")
                    break
                start = time.time()
                


            
    print("Constraint15 true:", counterConstraint15True)
    print("Constraint15 false:", counterConstraint15False)
    
    print("Counter pos:", counterPos)
    print("Counter neg:", counterNeg)
    
    nbrOfTrainingPoints = round(counter*3/4 if counter % 400 == 0 else (counter - counter % 400)*3/4)
    nbrOfValidationPoints = round(counter/8 if counter % 400 == 0 else (counter - counter % 400)/8)
    print("\nData generation succesfull")
    print("{} training points".format(nbrOfTrainingPoints))
    print("{} validation points".format(nbrOfValidationPoints))
    print("{} testing points".format(nbrOfValidationPoints))
        
    nbrOfPoles = 3
    #Write input parameters to file:
    params = open("inputParameters.py","w")
    params.write("nbrOfPoles = {}\n".format(nbrOfPoles))
    params.write("trainingPoints = {}\n".format(nbrOfTrainingPoints))
    params.write("validationPoints = {}\n".format(nbrOfValidationPoints))
    params.write("pstart = {}\n".format(pstart))
    params.write("pend = {}\n".format(pend))
    params.write("nbrPoints = {}\n".format(nbrPoints))
    params.write("nbrWs = {}\n".format(nbrWs))
         
    params.close()
    
    