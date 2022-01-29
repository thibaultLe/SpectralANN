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


#Load test data
path = "C:/Users/Thibault/Documents/Universiteit/Honours/Deel 2, interdisciplinair/Code/NN/MonteCarloDataset/"


filename = "boot_samples.dat"
# filename = "gluon_bare_64x4_1801Conf_20000bootsamples.dat"
# filename = "gluon_bare_80x4_1801Conf_18010bootsamples.dat"

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
# saved = "savedNNmodel8x1000,0.190,100k.pth"

#Note: Make sure the dimensions are the same
model = ACANN(inputSize,outputSize,6*[800],drop_p=0.05).double()
## 6*800, 100k training -> maxmae 880, huge errors
##          10k training -> maxmae 4.9, last one is noisy
# model = ACANN(inputSize,outputSize,8*[1000],drop_p=0.05).double()
# model = ACANN(inputSize,outputSize,4*[100],drop_p=0.05).double()
# model = ACANN(inputSize,outputSize,4*[400],drop_p=0.05).double()
model.load_state_dict(torch.load(saved))
model.eval()


from scipy.interpolate import interp1d

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

#Remove first:
propLists = propLists[1:]

print("Loaded propagators")
        

# for i in range(len(propLists)):
#     if len(propLists[i]) != 169:
#         print(propLists[i][0])
#         print(propLists[i][-1])
#         print("Length doesnt match",len(propLists[i]))
    
#Proplists: First is length 169, rest is length 168
#ps from 0 -> move 0.01 to the right
# print("Len x:",len(x))
# print("Len props:",len(propLists))
# print("Len first props:",len(propLists[0]))
# print(propLists[0])

# meanProps = []
# plt.figure()

actualPropagators = []
NNinputs = []
for i in range(len(propLists)):
    psT = [item[0] for item in propLists[i]]
    dp2snoscale = [item[1] for item in propLists[i]]
    # print(psT[73])
    # if i == 1:
    #     print(psT[37],psT[38],psT[39])
    #psT[37] = 0.998
    
    # if psT[0]<0.01:
    #     psT[0] = 0.01
    
    # plt.plot(psT,dp2s,alpha=0.05)
    # meanProps.append(dp2s)
    
    dp2sFunc = interp1d(psT,dp2snoscale)
    dp2sInter = dp2sFunc(ps)
    dp2s = [item/(dp2sInter[11]) for item in dp2sInter]
    
    actualPropagators.append(dp2s)
    
    NNinputs.append(dp2s)

# plt.figure()
# plt.plot(psT,np.mean(meanProps,axis=0))
# plt.plot(psT,np.std(meanProps,axis=0))

NNinputs = pd.DataFrame(NNinputs)
# print(NNinputs)

testloader = DataLoader(TensorDataset((torch.tensor(NNinputs.values).double()).to("cuda:0")))
# print(testloader)


# print(NNinputs[0])
# print(propLists[-1])

iterloader = iter(testloader)

print("Data Loaded")

#Use NN to predict
predicList = []
with torch.no_grad():
    # D_test = next(iter(testloader))
    # print(D_test)
    for i in range(len(testloader)):
        propData = next(iterloader)[0] 
        
        # if i == 1:
        #     print(propData)
        #     propData = 2* propData
        #     print(propData)
        prediction = model.forward(propData)
        # print("output:",prediction)
        predicData = prediction.to("cpu").numpy()
        # print(predicData)
            
        predicList.append(predicData)
        
            

print(len(predicList))
predicData = []
for i in range(len(predicList)):
    predicData.append(predicList[i][0])


# print(predicData)

# print(predicData[-1])


def plotPolesForIndex(i,ax):
    polemarkers = ["o","^","*"]
    msizes = [7,9,11]
    for j in range(nbrOfPoles):
        #Only plot the poles
        cj = predicData[i][nbrWs + 4*j + 2]
        dj = predicData[i][nbrWs + 4*j + 3]
        
        ax.plot(cj,dj,polemarkers[j],color="cyan",label="Reconstructed poles",markersize=msizes[j])
        
                
                
    # ax.set_xlim([-7,7])
    # ax.set_ylim([0,7])
    ax.grid()
    ax.set_xlabel("Re(q)")
    ax.set_ylabel("Im(q)")
    

def plotResiduesForIndex(i,ax):
    resmarkers = ["o","^","*"]
    # resmarkers = ["$1$","$2$","$3$"]
    msizes = [7,9,11]
    for j in range(nbrOfPoles):
        #Only plot the poles
        aj = predicData[i][nbrWs + 4*j]
        bj = predicData[i][nbrWs + 4*j + 1]
        ax.plot(aj,bj,marker=resmarkers[j],color="lawngreen",label="Reconstructed residues",markersize=msizes[j])
        
        #Draw lines in between:
        # ax.plot([ajOrig,aj],[bjOrig,bj],color="green")
        
    
    # ax.set_xlim([-7,7])
    # ax.set_ylim([0,7])
    ax.grid()
    ax.set_xlabel("Re(R)")
    ax.set_ylabel("Im(R)")
    
#Reconstruct propagator from reconstructed spectral function and poles:
def poles(p,N,poleList):
    # print(poleList)
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
    from scipy import integrate
    sigma = predicData[index][-1]
    #If negative, -> 0
    #If more than 1 -> 1
    wscutoff = min(max(int(round(sigma/0.04995)),0),1)
    
    reconstructedPropSigma = []
    for p in ps:
        spectrFunc = []
        for i in range(wscutoff,len(ws)):
            spectrFunc.append(predicData[index][i]/(p**2+ws[i]))
        
        integral = integrate.simpson(spectrFunc,x=ws[wscutoff:])
        prop = integral + poles(p**2,3, predicData[index][nbrWs:nbrWs+12])
        reconstructedPropSigma.append(prop)
    
    # rescaling = reconstructedPropSigma[0]*0.1
    # rescaling = reconstructedPropSigma[51]*16
    rescaling = reconstructedPropSigma[11]
    for i in range(len(ps)):
        reconstructedPropSigma[i] = reconstructedPropSigma[i]/rescaling
        
    return reconstructedPropSigma
    
    
    
getBestAndWorst = True
if getBestAndWorst:
    #Get the best and worst test cases:
    maxMAEindex = 0
    maxMAE = 0
    minMAEindex = 0
    minMAE = 100000
    
    
    fullSortedList = []
    # nbrlargemaes = 0
    for i in range(len(actualPropagators)):
        MAE = 0
            
        #MAE on propagator
        reconProp = reconstructProp(i)
        combListProp = zip(actualPropagators[i],reconProp)
        scale = abs(max(actualPropagators[i],key=abs))
        for orig, recon in combListProp:
            MAE += abs(orig-recon)**2/scale
    
        # For 10k (6x800)
        # if MAE > maxMAE and i != 1229: 
        if MAE > maxMAE: 
            maxMAE = MAE
            maxMAEindex = i
        if MAE < minMAE:
            minMAE = MAE
            minMAEindex = i
            
        # if MAE > 1000:
        #      nbrlargemaes += 1
        #      print(MAE)
            
        #skip if prediction is outlier
        # if abs(max(predicData[i][:nbrWs])) < 20:
        fullSortedList.append((MAE,i))
    
    # print(len(actualPropagators))
    print("len sortedlist:",len(fullSortedList))
    fullSortedList.sort()
    
    print("Sorted all rhos")
    
    print("Min. MAE:",minMAE)
    print("Max. MAE:",maxMAE)
    
    # print(fullSortedList)

    # print("best:",minMAEindex)
    percentile25th = fullSortedList[round(len(fullSortedList)/4)][1]
    # print("25prct",percentile25th)
    percentile50th = fullSortedList[round(2*len(fullSortedList)/4)][1]
    # print("50prct",percentile50th)
    percentile75th = fullSortedList[round(3*len(fullSortedList)/4)][1]
    # print("75prct",percentile75th)
    # print("worst:",maxMAEindex)
    print("best,25prct,50prct,75prct,worst:", \
          [minMAEindex,percentile25th,percentile50th,percentile75th,maxMAEindex])
    
    #2nd best instead of 25th percentile:
    # percentile25th = fullSortedList[1][1]
    
    # percentile25th = fullSortedList[round(len(fullSortedList)/8)][1]
    # percentile50th = fullSortedList[2][1]
        
    
    #Test actual propagator:
    # maxMAEindex = -1
    # percentile75th = 8654
            
    
    #Find closest training propagator to a given monte carlo prop
    # MAEclosest = 10000
    # MAEcloseIndex = 0
    # indexOfComparison = 1
    
    # for i in range(len(alldata)):
    #     MAEc = 0
    #     scale = abs(max(alldata[i]))
    #     for j in range(len(alldata[i])):
    #         #TODO: debug
    #         MAEc += abs(alldata[i][j] - actualPropagators[indexOfComparison][j])
    #     MAEc = MAEc/scale
    #     if MAEc < MAEclosest:
    #         MAEcloseIndex = i
    #         MAEclosest = MAEc
    
    
    
    
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
    ps = np.linspace(pstart,pend,nbrPoints)
    for i in range(len(propaxes)):
        # if i == 0:
        #     propaxes[i].plot(ps,alldata[MAEcloseIndex],label="Most similar training propagator")
        propaxes[i].plot(ps,actualPropagators[indices[i]],label="Propagator")
        propaxes[i].plot(ps,reconstructProp(indices[i]),"--",label="Reconstructed propagator",color="red")
        propaxes[i].set_xlabel("p")
        propaxes[i].set_ylabel("D(p²)")
            
    rhoaxes = [ax12,ax22,ax32,ax42,ax52]
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
    
    
    



