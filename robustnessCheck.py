# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 17:54:33 2021

@author: Thibault
"""


#1: add noise 20 times to same propagator
#2: Convert to correct input
#3: Input to NN
#4: Calc average and stddev of spectral functions

#1e-3 noise,100k:
    #were erased??
# indices = [2608, 4750, 14605, 6786, 4920]
# indices =  [41, 1453, 1552, 1322, 1232]


# indices = [4241, 1499, 4321, 3744, 1755]
# indices = [2944, 341, 2332, 2048, 1558]
indices =[366, 61, 189, 269, 730]

nbrOfSamples = 100
noiseSize = 1e-2


from Database import Database
from torch.utils.data import DataLoader
import inputParameters as config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple

np.random.seed(64)

#Load input parameters from inputParameters.py
sizeOfTraining = config.trainingPoints 
sizeOfValidation = config.validationPoints
pstart = config.pstart
pend = config.pend
nbrPoints = config.nbrPoints


path = "C:/Users/Thibault/Documents/Universiteit/\
Honours/Deel 2, interdisciplinair/Code/NN/Datasets/"

train_data = Database(csv_target= path + "rhoTraining.csv",\
                      csv_input= path + "DTrainingRaw.csv",nb_data=sizeOfTraining).get_loader()
trainloader = DataLoader(train_data,batch_size=sizeOfTraining)
print("Training data loaded")

test_data = Database(csv_target= path + "rhoTest.csv",\
    csv_input= path + "DTestRaw.csv",nb_data=sizeOfValidation).get_loader()
testloader = DataLoader(test_data,batch_size=sizeOfValidation)
print("Test data loaded")

#get propagator data:
alldatatensors = list(trainloader)
alldata = alldatatensors[0][0].to("cpu").numpy()

alldatatensorsTest = list(testloader)
alldataTest = alldatatensorsTest[0][0].to("cpu").numpy()


print(len(alldata),"training points")


path = "C:/Users/Thibault/Documents/Universiteit/Honours/Deel 2, interdisciplinair/Code/NN/Datasets/"
  

"""
########################
Test robustness of NN:
########################
"""

import torch
from ACANN import ACANN
from Database import Database
from torch.utils.data import DataLoader
import numpy as np
import inputParameters as config

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
print("NN input size {}, output size {} plus {} poles".format(inputSize,outputSize-4*nbrOfPoles,nbrOfPoles))

#Load the saved NN model (made in train_ACANN.py)
# saved = "savedNNmodel.pth"
# #Note: Make sure the dimensions are the same
# model = ACANN(inputSize,outputSize,6*[800],drop_p=0.05).double()

# saved = "savedNNmodel8x1000,0.190,100k.pth"
saved = "savedNNmodel.pth"

#Note: Make sure the dimensions are the same
model = ACANN(inputSize,outputSize,6*[800],drop_p=0.05).double()
# model = ACANN(inputSize,outputSize,8*[1000],drop_p=0.05).double()
# model = ACANN(inputSize,outputSize,4*[400],drop_p=0.05).double()
model.load_state_dict(torch.load(saved))
model.eval()



#Load test data
path = "C:/Users/Thibault/Documents/Universiteit/Honours/Deel 2, interdisciplinair/Code/NN/Datasets/"
test_data = Database(csv_target= path + "rhoTest.csv", \
                     csv_input= path + "DTestRaw.csv",nb_data=sizeOfValidation).get_loader()
testloader = DataLoader(test_data,batch_size=sizeOfValidation)

testloadList = list(testloader)
rhovaluesList = testloadList[0][1].to("cpu").numpy()
print(len(rhovaluesList),"testing points")

prop_data = pd.read_csv(path+'DTestRaw.csv',header=None,nrows=sizeOfValidation)
propList = prop_data.values.tolist()
print(len(propList),"propagators")

params_data = pd.read_csv(path+'params.csv',header=None,nrows=sizeOfTraining+2*sizeOfValidation)
paramsList = params_data.values.tolist()

print("Data Loaded")


#Evaluate output:
ps = np.linspace(pstart,pend,nbrPoints)
ws = np.linspace(0.01,10,nbrWs)


def getMeanAndStdReconstruction(index):
    noisyPropsPerIndex = []
    for j in range(nbrOfSamples):
        noise = np.random.normal(1,noiseSize,nbrPoints)
        noisyPropsPerIndex.append(alldataTest[index] * noise)
        
    
    NNinputsTensor = []
    for j in range(nbrOfSamples):
        NNinputsTensor.append(noisyPropsPerIndex[j])
        
    NNinputsTensor = torch.DoubleTensor(np.array(NNinputsTensor)).cuda()
        
    #Use NN to predict
    with torch.no_grad():
        prediction = model.forward(NNinputsTensor)
        predicData = prediction.to("cpu").numpy()
    
    filteredPredicData = []
    for j in range(nbrOfSamples):
        if max(predicData[j][:nbrWs]) < 10:
            filteredPredicData.append(predicData[j])
        
    poles = []
    for j in range(len(filteredPredicData)):
        poles.append(filteredPredicData[j][nbrWs:])
    
    means = np.mean(filteredPredicData,axis=0)[:nbrWs]
    stddevs = np.std(filteredPredicData,axis=0)[:nbrWs]
    
    props = []
    for j in range(len(filteredPredicData)):
        props.append(reconstructProp(filteredPredicData[j]))
    
    propmeans = np.mean(props,axis=0)
    propstddevs = np.std(props,axis=0)
    
    return means, stddevs, poles, propmeans, propstddevs

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

def reconstructProp(reconstruction):
    from scipy import integrate
    sigma = reconstruction[-1]
    #If negative, -> 0
    #If more than 1 -> 1
    wscutoff = min(max(int(round(sigma/0.04995)),0),1)
    
    reconstructedPropSigma = []
    for p in ps:
        spectrFunc = []
        for i in range(wscutoff,len(ws)):
            spectrFunc.append(reconstruction[i]/(p**2+ws[i]))
        
        integral = integrate.simpson(spectrFunc,x=ws[wscutoff:])
        prop = integral + poles(p**2,3, reconstruction[nbrWs:nbrWs+12])
        reconstructedPropSigma.append(prop)
    
    # rescaling = reconstructedPropSigma[51]*16
    rescaling = reconstructedPropSigma[11]
    for i in range(len(ps)):
        reconstructedPropSigma[i] = reconstructedPropSigma[i]/rescaling
    return reconstructedPropSigma



fig, ((ax11,ax12,ax13,ax14),(ax21,ax22,ax23,ax24),(ax31,ax32,ax33,ax34), \
          (ax41,ax42,ax43,ax44),(ax51,ax52,ax53,ax54)) = plt.subplots(5,4)
    
propaxes = [ax11,ax21,ax31,ax41,ax51]
spectralaxes = [ax12,ax22,ax32,ax42,ax52]
polesaxes = [ax13,ax23,ax33,ax43,ax53]
resaxes = [ax14,ax24,ax34,ax44,ax54]

for i in range(len(indices)):   
    means,stddevs,polesAndSigma,propmeans,propstddevs = getMeanAndStdReconstruction(indices[i])
    
    propaxes[i].plot(ps,propList[indices[i]],label="Propagator")
    propaxes[i].plot(ps,propmeans,"--",label="Mean reconstruction",color="red")
    propaxes[i].fill_between(ps,propmeans-propstddevs,propmeans+propstddevs,alpha=0.2, facecolor="red",
                    label='Standard deviation')
    propaxes[i].set_xlabel("p")
    propaxes[i].set_ylabel("D(p²)")
    # propaxes[i].set_xscale('log')
    
    #Plot spectral function:
    spectralaxes[i].set_xlabel("ω²")
    spectralaxes[i].set_ylabel("ρ(ω)")
    
    spectralaxes[i].plot(ws,rhovaluesList[indices[i]][:nbrWs],label="Original function")
    spectralaxes[i].plot(ws,means,"--",label="Mean reconstruction",color="red")
    spectralaxes[i].fill_between(ws,means-stddevs,means+stddevs,alpha=0.2, facecolor="red",
                    label='Standard deviation')
    
    #Plot poles:
    aksA,bksA,cksA,dksA = [], [], [], []
    for j in range(len(polesAndSigma)):
        aks,bks,cks,dks = [], [], [], []
        for k in range(3):
            aks.append(polesAndSigma[j][4*k])
            bks.append(polesAndSigma[j][4*k + 1])
            cks.append(polesAndSigma[j][4*k + 2])
            dks.append(polesAndSigma[j][4*k + 3])
        aksA.append(aks)
        bksA.append(bks)
        cksA.append(cks)
        dksA.append(dks)
    
    aM = np.mean(aksA,axis=0)
    bM = np.mean(bksA,axis=0)
    cM = np.mean(cksA,axis=0)
    dM = np.mean(dksA,axis=0)
    
    resmarkers = ["o","^","*"]
    msizes = [7,9,11]
    
    # from scipy.spatial import ConvexHull
    
    
    for j in range(3):
        #Plot all reconstructions:
        for k in range(len(aksA)):
            resaxes[i].plot(aksA[k][j],bksA[k][j],marker=resmarkers[j],markersize=msizes[j],color="red",alpha=0.1,label="All reconstructions")
        
        for k in range(len(aksA)):
            polesaxes[i].plot(cksA[k][j],dksA[k][j],marker=resmarkers[j],markersize=msizes[j],color="red",alpha=0.1,label="All reconstructions")
        
    
    for j in range(3):
        ajOrig = rhovaluesList[indices[i]][nbrWs + 4*j]
        bjOrig = rhovaluesList[indices[i]][nbrWs + 4*j + 1]
        resaxes[i].plot(ajOrig,bjOrig,marker=resmarkers[j],color="green",label="Original residues",markersize=msizes[j])
        resaxes[i].plot(aM[j],bM[j],marker=resmarkers[j],color="lawngreen",label="Mean reconstruction",markersize=msizes[j])
        
        #Plot convex hull:
        # aHull = [elem[j] for elem in aksA]
        # bHull = [elem[j] for elem in bksA]
        # points = np.asarray([list(elem) for elem in zip(aHull,bHull)])
        # hull = ConvexHull(points)
        # resaxes[i].fill(points[hull.vertices,0],points[hull.vertices,1],'red',alpha=0.4)
        
        
        cjOrig = rhovaluesList[indices[i]][nbrWs + 4*j + 2]
        djOrig = rhovaluesList[indices[i]][nbrWs + 4*j + 3]
        polesaxes[i].plot(cjOrig,djOrig,resmarkers[j],color="blue",label="Original poles",markersize=msizes[j])
        polesaxes[i].plot(cM[j],dM[j],resmarkers[j],color="cyan",label="Mean reconstruction",markersize=msizes[j])
        
        #Plot convex hull:
        # cHull = [elem[j] for elem in cksA]
        # dHull = [elem[j] for elem in dksA]
        # points = np.asarray([list(elem) for elem in zip(cHull,dHull)])
        # hull = ConvexHull(points)
        # polesaxes[i].fill(points[hull.vertices,0],points[hull.vertices,1],'red',alpha=0.4)
        
    resaxes[i].set_xlim([-1.5,1.5])
    resaxes[i].set_ylim([0,1.2])
    resaxes[i].grid()
    resaxes[i].set_xlabel("Re(R)")
    resaxes[i].set_ylabel("Im(R)")
    
    polesaxes[i].set_xlim([0.1,0.4])
    polesaxes[i].set_ylim([0.2,0.7])
    polesaxes[i].grid()
    polesaxes[i].set_xlabel("Re(q)")
    polesaxes[i].set_ylabel("Im(q)")
    
    """
    TODO: Mean, stddev of poles:
    """
    print("Mean, stddev of sigma:", np.mean(polesAndSigma,axis=0)[-1],np.std(polesAndSigma,axis=0)[-1])
    
        

handles, labels = ax11.get_legend_handles_labels()
ax11.legend(handles,labels,loc="upper center",bbox_to_anchor=(0.5,1.7))

handles, labels = ax12.get_legend_handles_labels()
ax12.legend(handles,labels,loc="upper center",bbox_to_anchor=(0.5,1.7))
fig.set_tight_layout(True)

handles,labels = ax13.get_legend_handles_labels()
# print(handles)
origTuple = (handles[-6],handles[-4],handles[-2])
reconTuple = (handles[-5],handles[-3],handles[-1])
allRecon = (handles[0],handles[int(round(len(handles)/2))],handles[-7])
labels = ["Original poles", "Mean reconstruction","All reconstructions"]
ax13.legend((origTuple,reconTuple,allRecon),labels,scatterpoints=3,
            numpoints=1, handler_map={tuple: HandlerTuple(ndivide=3,pad=1.3)},
            loc="upper center",bbox_to_anchor=(0.5,1.7),handlelength=4)

handles,labels = ax14.get_legend_handles_labels()
origTuple = (handles[-6],handles[-4],handles[-2])
reconTuple = (handles[-5],handles[-3],handles[-1])
allRecon = (handles[0],handles[int(round(len(handles)/2))],handles[-7])
labels = ["Original residues", "Mean reconstruction","All reconstructions"]
ax14.legend((origTuple,reconTuple,allRecon),labels,scatterpoints=3,
            numpoints=1, handler_map={tuple: HandlerTuple(ndivide=3,pad=1.3)},
            loc="upper center",bbox_to_anchor=(0.5,1.7),handlelength=4)
    
# avgRhos.append(avgForIndex)
# print(means)
            
# print(NNinputs[0])
# plt.figure()
# # plt.plot(ws,predicData[0][:nbrWs],label="Reconstructed spectral function")
# plt.plot(ws,avgForIndex,label="Average reconstruction")
# plt.plot(ws,rhovaluesList[314][:nbrWs],label="Original spectral function")
# plt.legend()



