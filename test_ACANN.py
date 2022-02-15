# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 10:53:43 2021

@author: Thibault
"""
import torch
from ACANN import ACANN
from Database import Database
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import inputParameters as config
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from scipy import integrate
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


#Load input parameters from inputParameters.py
nbrWs = config.nbrWs
nbrOfPoles = config.nbrOfPoles
sizeOfTraining = config.trainingPoints 
sizeOfValidation = config.validationPoints
outputSize = nbrWs + (4 * nbrOfPoles) + 1
pstart = config.pstart
pend = config.pend
nbrPoints = config.nbrPoints
inputSize = nbrPoints
print("NN input size {}, output size {} plus {} poles and sigma".format(inputSize,outputSize-4*nbrOfPoles-1,nbrOfPoles))


#Load the saved NN model (made in train_ACANN.py)
saved = "savedNNmodel.pth"

#Note: Make sure the dimensions are the same
model = ACANN(inputSize,outputSize,6*[600],drop_p=0.1).double()
model.load_state_dict(torch.load(saved))
model.eval()



#Load test data
path = "C:/Users/Thibault/Documents/Universiteit/Honours/Deel 2, interdisciplinair/Code/NN/Datasets/"
test_data = Database(csv_target= path + "rhoTest.csv", \
                     csv_input= path + "DTest.csv",nb_data=sizeOfValidation).get_loader()
testloader = DataLoader(test_data,batch_size=sizeOfValidation)

testloadList = list(testloader)
#Convert tensor to numpy array:
rhovaluesList = testloadList[0][1].to("cpu").numpy()
print(len(rhovaluesList),"testing points")

prop_data = pd.read_csv(path+'DTest.csv',header=None,nrows=sizeOfValidation)
propList = prop_data.values.tolist()
print(len(propList),"propagators")

params_data = pd.read_csv(path+'params.csv',header=None,nrows=sizeOfTraining+2*sizeOfValidation)
paramsList = params_data.values.tolist()

print("Data Loaded")

origProp = []
#Use NN to predict
with torch.no_grad():
    D_test,rho_test = next(iter(testloader))
    # print(D_test)
    prediction = model.forward(D_test)
    # print("output:",prediction)
    predicData = prediction.to("cpu").numpy()
    # print("output:",predicData)
    origProp.append(D_test.to("cpu").numpy())

#Evaluate output:
ps = np.linspace(pstart,pend,nbrPoints)
ws = np.linspace(0.01,10,nbrWs)


    
def plotPolesForIndex(i,ax):
    polemarkers = ["o","^","*"]
    msizes = [7,9,11]
    for j in range(nbrOfPoles):
        #Only plot the poles
        cj = predicData[i][nbrWs + 4*j + 2]
        dj = predicData[i][nbrWs + 4*j + 3]
        cjOrig = rhovaluesList[i][nbrWs + 4*j + 2]
        djOrig = rhovaluesList[i][nbrWs + 4*j + 3]
        
        if i != -1:
            ax.plot(cjOrig,djOrig,polemarkers[j],color="blue",label="Original poles",markersize=msizes[j])
        ax.plot(cj,dj,polemarkers[j],color="cyan",label="Reconstructed poles",markersize=msizes[j])
        
                
    ax.set_xlim([0.15,0.4])
    ax.set_ylim([0.2,0.8])
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
        ajOrig = rhovaluesList[i][nbrWs + 4*j]
        bjOrig = rhovaluesList[i][nbrWs + 4*j + 1]
        
        if i != -1:
            ax.plot(ajOrig,bjOrig,marker=resmarkers[j],color="green",label="Original residues",markersize=msizes[j])
        ax.plot(aj,bj,marker=resmarkers[j],color="lawngreen",label="Reconstructed residues",markersize=msizes[j])
        
    
    ax.set_xlim([-2,2])
    ax.set_ylim([-0.2,1])
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
    dp0 = origProp[0][index][0] 
    dp1 = origProp[0][index][1]
    rho0 = predicData[index][0]
    rho1 = predicData[index][1]
    derivativeRho = (rho1 - rho0)/(ws[1]-ws[0])
    derivativeProp = (dp1 - dp0)/(ps[1]**2-ps[0]**2)
    
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
    for i in range(len(rhovaluesList)):
        MAE = 0
        
        combListAll = zip(rhovaluesList[i],predicData[i])
        scale = max(abs(rhovaluesList[i]))
        for orig, recon in combListAll:
            MAE += abs(orig-recon)/scale
        
        #MAE on spectral function only
        # combList = zip(rhovaluesList[i][:nbrWs],predicData[i][:nbrWs])
        # scale = max(abs(rhovaluesList[i][:nbrWs]))
        # for orig, recon in combList:
        #     MAE += abs(orig-recon)/scale
            
        #MAE on poles only
        # combList = zip(rhovaluesList[i][nbrWs:],predicData[i][nbrWs:])
        # for orig, recon in combList:
        #     MAE += abs(orig-recon)
            
        #MAE on propagator only
        # reconProp = reconstructProp(i)
        # combListProp = zip(propList[i],reconProp)
        # scale = abs(max(propList[i]))
        # for orig, recon in combListProp:
        #     MAE += abs(orig-recon)/scale
            
        
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
        

    
    fullSortedList.sort()
    
    # print("Sorted all rhos")
    
    
    print(str(constraintsSatisfied1)+"/"+str(len(rhovaluesList)), "recons satisfied constraint 1")
    print(str(constraintsSatisfied2)+"/"+str(len(rhovaluesList)), "recons satisfied constraint 2")
    print(str(constraintsSatisfied3)+"/"+str(len(rhovaluesList)), "recons satisfied constraint 3")
    print(str(constraintsSatisfiedAll)+"/"+str(len(rhovaluesList)), "recons satisfied all constraints")
    
    
    print("Min. MAE:",minMAE)
    print("Max. MAE:",maxMAE)

    percentile25th = fullSortedList[round(len(fullSortedList)/4)][1]
    percentile50th = fullSortedList[round(2*len(fullSortedList)/4)][1]
    percentile75th = fullSortedList[round(3*len(fullSortedList)/4)][1]
    print("index of the best,25prct,50prct,75prct,worst:", \
          [minMAEindex,percentile25th,percentile50th,percentile75th,maxMAEindex])
    
    
    
    fig, ((ax11,ax12,ax13,ax14),(ax21,ax22,ax23,ax24),(ax31,ax32,ax33,ax34), \
          (ax41,ax42,ax43,ax44),(ax51,ax52,ax53,ax54)) = plt.subplots(5,4)
    
    plotPolesForIndex(minMAEindex, ax13)
    plotPolesForIndex(percentile25th,ax23)
    plotPolesForIndex(percentile50th,ax33)
    plotPolesForIndex(percentile75th,ax43)
    plotPolesForIndex(maxMAEindex, ax53)
    
    plotResiduesForIndex(minMAEindex, ax14)
    plotResiduesForIndex(percentile25th,ax24)
    plotResiduesForIndex(percentile50th,ax34)
    plotResiduesForIndex(percentile75th,ax44)
    plotResiduesForIndex(maxMAEindex, ax54)
    
    indices = [minMAEindex,percentile25th,percentile50th,percentile75th,maxMAEindex]
    
    propaxes = [ax11,ax21,ax31,ax41,ax51]
    for i in range(len(propaxes)):
        propaxes[i].plot(ps,propList[indices[i]],label="Propagator")
        propaxes[i].plot(ps,reconstructProp(indices[i]),"--",label="Reconstructed propagator",color="red")
        propaxes[i].set_xlabel("p")
        propaxes[i].set_ylabel("D(p²)")
            
    rhoaxes = [ax12,ax22,ax32,ax42,ax52]
    for i in range(len(rhoaxes)):
        if indices[i] != -1:
            rhoaxes[i].plot(ws,rhovaluesList[indices[i]][:nbrWs],label="Spectral function")
        rhoaxes[i].plot(ws,predicData[indices[i]][:nbrWs],"--",label="Reconstructed spectral function",color="red")
        rhoaxes[i].set_xlabel("ω²")
        rhoaxes[i].set_ylabel("ρ(ω)")
    
    
    
    handles, labels = ax11.get_legend_handles_labels()
    ax11.legend(handles,labels,loc="upper center",bbox_to_anchor=(0.5,1.5))
    
    handles, labels = ax12.get_legend_handles_labels()
    ax12.legend(handles,labels,loc="upper center",bbox_to_anchor=(0.5,1.5))
    
    handles,labels = ax13.get_legend_handles_labels()
    origTuple = (handles[0],handles[2],handles[4])
    reconTuple = (handles[1],handles[3],handles[5])
    labels = ["Original poles", "Reconstructed poles"]
    ax13.legend((origTuple,reconTuple),labels,scatterpoints=3,
                numpoints=1, handler_map={tuple: HandlerTuple(ndivide=3,pad=1.3)},
                loc="upper center",bbox_to_anchor=(0.5,1.5),handlelength=4)
    
    handles,labels = ax14.get_legend_handles_labels()
    origTuple = (handles[0],handles[2],handles[4])
    reconTuple = (handles[1],handles[3],handles[5])
    labels = ["Original residues", "Reconstructed residues"]
    ax14.legend((origTuple,reconTuple),labels,scatterpoints=3,
                numpoints=1, handler_map={tuple: HandlerTuple(ndivide=3,pad=1.3)},
                loc="upper center",bbox_to_anchor=(0.5,1.5),handlelength=4)
    
    fig.set_tight_layout(True)
    
    



