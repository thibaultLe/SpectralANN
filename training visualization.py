# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 22:02:52 2021

@author: Thibault
"""
from Database import Database
from torch.utils.data import DataLoader
import numpy as np
import inputParameters as config
import matplotlib.pyplot as plt
import pandas as pd

#Load input parameters from inputParameters.py
nbrWs = config.nbrWs
nbrOfPoles = config.nbrOfPoles
sizeOfTraining = config.trainingPoints 
sizeOfValidation = config.validationPoints
outputSize = nbrWs + (4 * nbrOfPoles) + 1
pstart = config.pstart
pend = config.pend
nbrPoints = config.nbrPoints
print("outputsize:",outputSize)
print("Input parameters loaded")

inputSize = 100
ws = np.linspace(0.01,10,nbrWs)
ps = np.linspace(pstart,pend,nbrPoints)

#Load test data
path = "C:/Users/Thibault/Documents/Universiteit/Honours/Deel 2, interdisciplinair/Code/NN/MonteCarloDataset/"

x = np.loadtxt(path+"boot_samples.dat")

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

from scipy.interpolate import interp1d
actualPropagators = []
NNinputs = []
for i in range(len(propLists)):
    psT = [item[0] for item in propLists[i]]
    dp2snoscale = [item[1] for item in propLists[i]]
    
    dp2sFunc = interp1d(psT,dp2snoscale)
    dp2sInter = dp2sFunc(ps)
    dp2s = [item/(dp2sInter[12]) for item in dp2sInter]
    
    actualPropagators.append(dp2s)


plt.figure()

# print(actualPropagators[0][0])


epochs = 50
batch_size_train = 1

# Import the data
path = "C:/Users/Thibault/Documents/Universiteit/Honours/Deel 2, interdisciplinair/Code/NN/Datasets/"
train_data = Database(csv_target= path + "rhoTraining.csv",csv_input= path + "DTrainingRaw.csv",nb_data=sizeOfTraining).get_loader()
validation_data=Database(csv_target= path + "rhoValidation.csv",csv_input= path + "DValidationRaw.csv",nb_data=sizeOfValidation).get_loader()

trainloader = DataLoader(train_data,batch_size=batch_size_train,shuffle=False)


params_data = pd.read_csv(path+'params.csv',header=None,nrows=sizeOfTraining+2*sizeOfValidation)
paramsList = np.array(params_data.values.tolist())[np.mod(np.arange(len(params_data.values)),4) != 0]
print(len(paramsList))
names = ['sigma','Z','m2','lam2','abcds','ABCs','gabls','gabis']

def printparams(index):
    for i in range(len(paramsList[index])):
        if i != 1:
            print(names[i]+':',paramsList[index][i],end=' ')
        
    print("\n")
print("Data Loaded")


counter = 0 
closestMAE = 10000
closestIndex = 0
for D,rho in trainloader:
    prop = D.to("cpu").numpy()[0]
    if prop[0] < 4.7 and prop[0] > 4.5:
    #     printparams(counter)
    #     plt.plot(ps,prop,color="black",alpha=0.1)
    #     # plt.xscale("log")
    
    # MAE = 0
    # for i in range(len(ps)):
    #     MAE += abs(actualPropagators[0][i] - prop[i])
    
    # if MAE < closestMAE:
    #     closestMAE = MAE
    #     closestIndex = counter
    #     if MAE < 0.65:
    #         print(MAE)
    #         printparams(counter)
        plt.plot(ps,prop,color="black",alpha=0.2)
        
    counter += 1

print("Closest prop:",closestIndex,closestMAE)
# printparams(closestIndex)


plt.plot(ps,actualPropagators[0])

plt.xlabel("p")
plt.ylabel("D(p²)")
plt.title("All training propagators")
# plt.xscale("log")
# plt.yscale("log")
    
