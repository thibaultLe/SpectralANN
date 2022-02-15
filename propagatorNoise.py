# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 15:06:14 2021

@author: Thibault
"""

from Database import Database
from torch.utils.data import DataLoader
import inputParameters as config
import numpy as np
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

noiseSize = 5e-3


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

validation_data = Database(csv_target= path + "rhoValidation.csv",\
    csv_input= path + "DValidationRaw.csv",nb_data=sizeOfValidation).get_loader()
validationloader = DataLoader(validation_data,batch_size=sizeOfValidation)
print("Validation data loaded")

test_data = Database(csv_target= path + "rhoTest.csv",\
    csv_input= path + "DTestRaw.csv",nb_data=sizeOfValidation).get_loader()
testloader = DataLoader(test_data,batch_size=sizeOfValidation)
print("Test data loaded")

alldatatensors = list(trainloader)
alldata = alldatatensors[0][0].to("cpu").numpy()

alldatatensorsValid = list(validationloader)
alldataValid = alldatatensorsValid[0][0].to("cpu").numpy()

alldatatensorsTest = list(testloader)
alldataTest = alldatatensorsTest[0][0].to("cpu").numpy()

#Add noise:
for i in range(len(alldata)):
    noise = np.random.normal(1,noiseSize,nbrPoints)
    alldata[i] = alldata[i] * noise
    
for i in range(len(alldataValid)):
    noise = np.random.normal(1,noiseSize,nbrPoints)
    alldataValid[i] = alldataValid[i] * noise

for i in range(len(alldataTest)):
    noise = np.random.normal(1,noiseSize,nbrPoints)
    alldataTest[i] = alldataTest[i] * noise


print(len(alldata),"training points")



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



#Write data to files
propTraindf = pd.DataFrame(alldata)
propTraindf.to_csv(propTrain_csv,index=False,header=False,mode='a')

propValiddf = pd.DataFrame(alldataValid)
propValiddf.to_csv(propValid_csv,index=False,header=False,mode='a')


propTestdf = pd.DataFrame(alldataTest)
propTestdf.to_csv(propTest_csv,index=False,header=False,mode='a')

print("Succesfully added artificial noise to training data.")


    
    
    
    