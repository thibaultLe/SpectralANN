# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 11:22:43 2021

@author: Thibault
"""
import torch
from ACANN import ACANN
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import inputParameters as config
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from  torch.utils.data import TensorDataset
from scipy import integrate
from scipy.interpolate import interp1d
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


#Load test data
path = "C:/Users/Thibault/Documents/Universiteit/Honours/Deel 2, interdisciplinair/Code/NN/MonteCarloDataset/"


# filename = "gluon_bare_64x4_1801Conf_20000bootsamples.dat"
filename = "gluon_bare_80x4_1801Conf_18010bootsamples.dat"

x = np.loadtxt(path+filename)



#Load input parameters from inputParameters.py
nbrWs = config.nbrWs
nbrOfPoles = config.nbrOfPoles
sizeOfTraining = config.trainingPoints 
sizeOfValidation = config.validationPoints
outputSize = nbrWs + (4 * nbrOfPoles) + 1
pstart = config.pstart
pend = config.pend
nbrPoints = config.nbrPoints
inputSize=nbrPoints
print("NN input size {}, output size {} plus {} poles and sigma".format(inputSize,outputSize-4*nbrOfPoles-1,nbrOfPoles))

ps = np.linspace(pstart,pend,nbrPoints)
psInterp = np.linspace(-1,1,nbrPoints)
ws = np.linspace(0.01,10,nbrWs)

# #Load the saved NN model (made in train_ACANN.py)
saved = "savedNNmodel.pth"

#Note: Make sure the dimensions are the same
model = ACANN(inputSize,outputSize,6*[600],drop_p=0.1).double()
model.load_state_dict(torch.load(saved))
model.eval()

#Parse text file:
propLists = []
currentList = []
lastP = 0
for i in range(len(x)):
    if x[i][0] < lastP-7:
        currentList.sort(key=lambda x: x[0])
        currentList.append([10,0,currentList[-1][2]])
        propLists.append(currentList)
        currentList = [x[i]]
        lastP = 0
        
    else:
        currentList.append(x[i])
        lastP = x[i][0]

#Remove first (has more datapoints):
propLists = propLists[1:]


print("Loaded propagators")
        
maxAmount = 50

actualPropagators = []
NNinputs = []
for i in range(len(propLists)):
    psT = [item[0] for item in propLists[i]]
    dp2snoscale = [item[1] for item in propLists[i]]
    
    dp2sFunc = interp1d(psT,dp2snoscale)
    dp2sInter = dp2sFunc(ps)
    dp2s = [item/(dp2sInter[12]) for item in dp2sInter]
    
    actualPropagators.append(dp2s)
    
    NNinputs.append(dp2s)
    
    if i > maxAmount:
        break


NNinputs = pd.DataFrame(NNinputs)

testloader = DataLoader(TensorDataset((torch.tensor(NNinputs.values).double()).to("cuda:0")))


iterloader = iter(testloader)

print("Data Loaded")

#Use NN to predict
predicList = []
with torch.no_grad():
    for i in range(len(testloader)):
        propData = next(iterloader)[0] 
        prediction = model.forward(propData)
        predicData = prediction.to("cpu").numpy()            
        predicList.append(predicData)
        
predicData = []
for i in range(len(predicList)):
    predicData.append(predicList[i][0])



def plotPolesForIndex(i,ax):
    polemarkers = ["o","^","*"]
    msizes = [7,9,11]
    for j in range(nbrOfPoles):
        #Only plot the poles
        cj = predicData[i][nbrWs + 4*j + 2]
        dj = predicData[i][nbrWs + 4*j + 3]
        
        ax.plot(cj,dj,polemarkers[j],color="cyan",label="Reconstructed poles",markersize=msizes[j])
        
    ax.grid()
    ax.set_xlabel("Re(q)")
    ax.set_ylabel("Im(q)")
    

def plotResiduesForIndex(i,ax):
    resmarkers = ["o","^","*"]
    msizes = [7,9,11]
    for j in range(nbrOfPoles):
        #Only plot the poles
        aj = predicData[i][nbrWs + 4*j]
        bj = predicData[i][nbrWs + 4*j + 1]
        ax.plot(aj,bj,marker=resmarkers[j],color="lawngreen",label="Reconstructed residues",markersize=msizes[j])
        
    ax.grid()
    ax.set_xlabel("Re(R)")
    ax.set_ylabel("Im(R)")
    
#Reconstruct propagator from reconstructed spectral function and poles:
def poles(p,N,poleList):
    jsum = 0
    for j in range(N):
        a = poleList[j*4]
        b = poleList[j*4+1]
        c = poleList[j*4+2]
        d = poleList[j*4+3]
        nom = 2*(a*(c+p)+b*d)
        denom = c**2 + 2*c*p + d**2 + p**2
    
        jsum += nom/denom
    return jsum

def reconstructProp(index):
    sigma = predicData[index][-1]
    wscutoff = min(range(len(ws)), key=lambda i: abs(ws[i]-sigma))
    
    reconstructedPropSigma = []
    for p in ps:
        spectrFunc = []
        for i in range(wscutoff,len(ws)):
            spectrFunc.append(predicData[index][i]/(p**2+ws[i]))
        
        integral = integrate.simpson(spectrFunc,x=ws[wscutoff:])
        prop = integral + poles(p**2,3, predicData[index][nbrWs:nbrWs+12])
        reconstructedPropSigma.append(prop)
    
    rescaling = reconstructedPropSigma[12]
    for i in range(len(ps)):
        reconstructedPropSigma[i] = reconstructedPropSigma[i]/rescaling
        
    return reconstructedPropSigma
    
    
    
def constraint15(index):
    sigma = predicData[index][-1]
    wscutoff = min(range(len(ws)), key=lambda i: abs(ws[i]-sigma))
    res = integrate.simpson(predicData[index][wscutoff:nbrWs],x=ws[wscutoff:])
    
    jsum = 0
    for j in range(3):
        jsum += 2 * predicData[index][nbrWs+j*4]
    
    
    res += jsum
    
    if res < 0.5 and res > -0.5:
        return True
    return False

def derivativeconstraint(index): 
    dp0 = actualPropagators[index][0] 
    dp1 = actualPropagators[index][1]
    rho0 = predicData[index][0]
    rho1 = predicData[index][1]
    derivativeRho = (rho1 - rho0)/(ws[1]-ws[0])
    derivativeProp = (dp1 - dp0)/(ps[1]**2-ps[0]**2)
    
    # print((ps[1]**2-ps[0]**2),(ps[1]-ps[0]))
    
    derivativePoles = 0
    for j in range(3):
        a = predicData[index][nbrWs+j*4]
        b = predicData[index][nbrWs+j*4+1]
        c = predicData[index][nbrWs+j*4+2]
        d = predicData[index][nbrWs+j*4+3]
        #Alternate but equivalent formula:
        derivativePoles += (-2*a*c**2 + 2*a*d**2 + 4*b*c*d)/((c**2 + d**2)**2)
        
    idealDerivative = -np.pi*derivativeRho - derivativePoles
    
    if derivativeProp < idealDerivative - 1 or \
        derivativeProp > idealDerivative + 1:
        return False
    
    return True

def positivepropconstraint(index):
    if min(reconstructProp(index)) < 0:
        return False
    return True


def testConstraints(index):
    if constraint15(index) and derivativeconstraint(index) and positivepropconstraint(index):
        return True
    return False    
    
getBestAndWorst = True
if getBestAndWorst:
    #Get the best and worst test cases:
    maxMAEindex = 0
    maxMAE = 0
    minMAEindex = 0
    minMAE = 100000
    
    
    constraintsSatisfied1 = 0
    constraintsSatisfied2 = 0
    constraintsSatisfied3 = 0
    constraintsSatisfiedAll = 0
    
    
    fullSortedList = []
    reconProps = []
    for i in range(len(actualPropagators)):
        MAE = 0
        #MAE on propagator
        reconProp = reconstructProp(i)
        reconProps.append(reconProp)
        
        combListProp = zip(actualPropagators[i],reconProp)
        for orig, recon in combListProp:
            MAE += abs(orig-recon)
            
        if MAE > maxMAE: 
            maxMAE = MAE
            maxMAEindex = i
        if MAE < minMAE:
            minMAE = MAE
            minMAEindex = i
            
            
        fullSortedList.append((MAE,i))
        
        
        if positivepropconstraint(i):
            constraintsSatisfied1 += 1
        if derivativeconstraint(i):
            constraintsSatisfied2 += 1
        if constraint15(i):
            constraintsSatisfied3 += 1
        if testConstraints(i):
            constraintsSatisfiedAll += 1
            
    
    
    print(str(constraintsSatisfied1)+"/"+str(len(propLists)), "recons satisfied constraint 1")
    print(str(constraintsSatisfied2)+"/"+str(len(propLists)), "recons satisfied constraint 2")
    print(str(constraintsSatisfied3)+"/"+str(len(propLists)), "recons satisfied constraint 3")
    print(str(constraintsSatisfiedAll)+"/"+str(len(propLists)), "recons satisfied all constraints")
    
    fullSortedList.sort()
    
    # print("Sorted all rhos")
    
    print("Min. MAE:",minMAE)
    print("Max. MAE:",maxMAE)
    
    percentile50th = fullSortedList[round(2*len(fullSortedList)/4)][1]
    print("index of the best, median, and worst:", \
          [minMAEindex,percentile50th,maxMAEindex])
    
    
    fig, ((ax11,ax12,ax13,ax14),(ax31,ax32,ax33,ax34), \
          (ax51,ax52,ax53,ax54)) = plt.subplots(3,4)
    
    plotPolesForIndex(minMAEindex, ax13)
    plotPolesForIndex(percentile50th,ax33)
    plotPolesForIndex(maxMAEindex, ax53)
    
    plotResiduesForIndex(minMAEindex, ax14)
    plotResiduesForIndex(percentile50th,ax34)
    plotResiduesForIndex(maxMAEindex, ax54)
    
    indices = [minMAEindex,percentile50th,maxMAEindex]
    
    propaxes = [ax11,ax31,ax51]
    ps = np.linspace(pstart,pend,nbrPoints)
    for i in range(len(propaxes)):
        propaxes[i].plot(ps,actualPropagators[indices[i]],label="Propagator")
        propaxes[i].plot(ps,reconstructProp(indices[i]),"--",label="Reconstructed propagator",color="red")
        propaxes[i].set_xlabel("p")
        propaxes[i].set_ylabel("D(p²)")
            
    rhoaxes = [ax12,ax32,ax52]
    for i in range(len(rhoaxes)):
        rhoaxes[i].plot(ws,predicData[indices[i]][:nbrWs],"--",label="Reconstructed spectral function",color="red")
        rhoaxes[i].set_xlabel("ω²")
        rhoaxes[i].set_ylabel("ρ(ω)")
    
    
    handles, labels = ax11.get_legend_handles_labels()
    ax11.legend(handles,labels,loc="upper center",bbox_to_anchor=(0.5,1.5))
    
    handles, labels = ax12.get_legend_handles_labels()
    ax12.legend(handles,labels,loc="upper center",bbox_to_anchor=(0.5,1.5))
    
    handles,labels = ax13.get_legend_handles_labels()
    reconTuple = (handles[0],handles[1],handles[2])
    labels = ["Reconstructed poles"]
    ax13.legend((reconTuple,),labels,scatterpoints=3,
                numpoints=1, handler_map={tuple: HandlerTuple(ndivide=3,pad=1.3)},
                loc="upper center",bbox_to_anchor=(0.5,1.5),handlelength=4)
    
    handles,labels = ax14.get_legend_handles_labels()
    reconTuple = (handles[0],handles[1],handles[2])
    labels = ["Reconstructed residues"]
    ax14.legend((reconTuple,),labels,scatterpoints=3,
                numpoints=1, handler_map={tuple: HandlerTuple(ndivide=3,pad=1.3)},
                loc="upper center",bbox_to_anchor=(0.5,1.5),handlelength=4)
    
    fig.set_tight_layout(True)
    
        