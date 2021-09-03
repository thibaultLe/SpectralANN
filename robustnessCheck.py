# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 17:54:33 2021

@author: Thibault
"""


#1: add noise 20 times to same propagator
#2: Convert to correct input
#3: Input to NN
#4: Calc average and stddev of spectral functions

#1e-3 noise
indices = [210, 40, 922, 982, 277]
nbrOfSamples = 100
noiseSize = 1e-3


from Database import Database
from torch.utils.data import DataLoader
import inputParameters as config
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

np.random.seed(64)

#Load input parameters from inputParameters.py
maxDegreeOfLegFit = config.maxDegreeOfLegFit
nbrOfPCAcomponents = config.nbrOfPCAcomponents
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

psForFit = np.linspace(-1,1,nbrPoints)

maxdegree = maxDegreeOfLegFit

#Legendre fit to all propagators in training set, keeps coefficients
coefficientsList = []
for i in range(len(alldata)):
    coefficientsList.append(np.polynomial.legendre.legfit(psForFit,alldata[i],maxdegree))


x = coefficientsList

#Normalise the attributes
scaler=StandardScaler()#instantiate
scaler.fit(x) # compute the mean and standard which will be used in the next command
X_scaled=scaler.transform(x)

pca=PCA(n_components=nbrOfPCAcomponents) 
pca.fit(X_scaled) 
X_pca=pca.transform(X_scaled) 


path = "C:/Users/Thibault/Documents/Universiteit/Honours/Deel 2, interdisciplinair/Code/NN/Datasets/"
  



print("Data conversion to PCA components succesfull.")



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
inputSize = config.nbrOfPCAcomponents
nbrWs = config.nbrWs
nbrOfPoles = config.nbrOfPoles
sizeOfTraining = config.trainingPoints 
sizeOfValidation = config.validationPoints
outputSize = nbrWs + (4 * nbrOfPoles) + 1
pstart = config.pstart
pend = config.pend
nbrPoints = config.nbrPoints
print("NN input size {}, output size {} plus {} poles".format(inputSize,outputSize-4*nbrOfPoles,nbrOfPoles))

#Load the saved NN model (made in train_ACANN.py)
saved = "savedNNmodel.pth"
#Note: Make sure the dimensions are the same
model = ACANN(inputSize,outputSize,6*[800],drop_p=0.05).double()
model.load_state_dict(torch.load(saved))
model.eval()



#Load test data
path = "C:/Users/Thibault/Documents/Universiteit/Honours/Deel 2, interdisciplinair/Code/NN/Datasets/"
test_data = Database(csv_target= path + "rhoTest.csv", \
                     csv_input= path + "DTest.csv",nb_data=sizeOfValidation).get_loader()
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
        noise = np.random.normal(0,noiseSize,nbrPoints)
        noisyPropsPerIndex.append(alldataTest[index] + noise)
        
    
    coefficientsListPerIndex = []
    for j in range(nbrOfSamples):
        coefficientsListPerIndex.append(np.polynomial.legendre.legfit(psForFit,noisyPropsPerIndex[j],maxdegree))
    
    #Normalise the attributes
    X_scaled=scaler.transform(coefficientsListPerIndex)
    X_pcaTest=pca.transform(X_scaled) 
    
    NNinputsTensor = []
    for j in range(nbrOfSamples):
        NNinputsTensor.append(X_pcaTest[j])
        
    NNinputsTensor = torch.DoubleTensor(NNinputsTensor).cuda()
        
    #Use NN to predict
    with torch.no_grad():
        prediction = model.forward(NNinputsTensor)
        predicData = prediction.to("cpu").numpy()
        
    spectralFs = []
    for j in range(nbrOfSamples):
        spectralFs.append(predicData[j])
    
    means = np.mean(spectralFs,axis=0)[:nbrWs]
    stddevs = np.std(spectralFs,axis=0)[:nbrWs]
    
    return means, stddevs



fig, ((ax11),(ax21),(ax31), \
          (ax41),(ax51)) = plt.subplots(5,1)
axes = [ax11,ax21,ax31,ax41,ax51]

for i in range(len(indices)):
    means,stddevs = getMeanAndStdReconstruction(indices[i])
    
    # plt.figure()
    # for f in spectralFs:
    #     plt.plot(ws,f[:nbrWs],alpha=0.3)
    # plt.plot(ws,means)
    
    # plt.figure()
    # plt.plot(ws,means,"--",label="Mean reconstruction",color="red")
    # plt.fill_between(ws,means-stddevs,means+stddevs,alpha=0.2, facecolor="red",
    #                 label='Standard deviation')
    
    axes[i].set_xlabel("ω")
    axes[i].set_ylabel("ρ(ω)")
    
    axes[i].plot(ws,rhovaluesList[indices[i]][:nbrWs],label="Original function")
    axes[i].plot(ws,means,"--",label="Mean reconstruction",color="red")
    axes[i].fill_between(ws,means-stddevs,means+stddevs,alpha=0.2, facecolor="red",
                    label='Standard deviation')

handles, labels = ax11.get_legend_handles_labels()
ax11.legend(handles,labels,loc="upper center",bbox_to_anchor=(0.5,1.7))
fig.set_tight_layout(True)

    
# avgRhos.append(avgForIndex)
# print(means)
            
# print(NNinputs[0])
# plt.figure()
# # plt.plot(ws,predicData[0][:nbrWs],label="Reconstructed spectral function")
# plt.plot(ws,avgForIndex,label="Average reconstruction")
# plt.plot(ws,rhovaluesList[314][:nbrWs],label="Original spectral function")
# plt.legend()



