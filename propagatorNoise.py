# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 15:06:14 2021

@author: Thibault
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 12:04:32 2021

@author: Thibault
"""


from Database import Database
from torch.utils.data import DataLoader
import inputParameters as config
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

noiseSize = 1e-2


# Noise levels: 1e-2 p=0, 4e-3 p=0.5, 3e-4 p=3

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

# params_data = pd.read_csv(path+'params.csv',header=None,nrows=sizeOfTraining+2*sizeOfValidation)
# print(params_data)
# print("Parameters data loaded")
# firstTest = params_data.iloc[4]
# print(firstTest)

#get propagator data:
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


"""
Test of actual propagator:
"""
TEST_ACTUAL = False
if TEST_ACTUAL:
    #Read test data and convert to proper list
    file = open("testPropData.txt","r")
    basestring = file.read().split("\n")
    
    newlist = []
    for elem in basestring:
        newelem = elem.translate(str.maketrans("","","`} ")).split("{")[1:]
        newlist.append(newelem)
    
    floatlist = []
    for elem in newlist:
        floatlist.append(elem[0].split(","))
    
    newlist = []
    for elem in floatlist:
        partlist = []
        for nbr in elem:
            if nbr != "":
                partlist.append(float(nbr))
        newlist.append(partlist)
        
    
    psT = [item[0]**2 for item in newlist]
    dp2s = [item[1] for item in newlist]
    errors = [item[2] for item in newlist]
    
    #Sort the arrays as some p's are out of order:
    from more_itertools import sort_together
    lists = sort_together([psT,dp2s,errors])
    psT = list(lists[0])
    dp2s = list(lists[1])
    errors = list(lists[2])
    relerrors = [errors[i]/dp2s[i] for i in range(len(dp2s))]
    
    plt.figure()
    plt.plot(psT,relerrors)
    plt.yscale('log')
    plt.xlabel("p²")
    plt.ylabel("Relative errors")
    
    # plt.figure()
    # plt.plot(psT,dp2s,"o")
    # plt.figure()
    # plt.plot(psT,dp2s)
    
    # psScaled = np.linspace(-1,1,len(psT))
    
    # from scipy.interpolate import interp1d
    
    # psNew = np.linspace(pstart,pend,nbrPoints)
    # dp2sFunc = interp1d(psT,dp2s)
    # dp2sInter = dp2sFunc(psNew)
    
    
    # import csv
    # with open(path+'DTestRaw.csv') as inf:
    #     reader = csv.reader(inf.readlines())

    # with open(path+'DTestRaw.csv', 'w') as outf:
    #     writer = csv.writer(outf)
    #     counter = 0
    #     for line in reader:
    #         if counter == 17949:
    #             writer.writerow(dp2sInter)
    #             print("########################")
    #             print("Wrote dp2sInter")
    #             print("########################")
    #             print("########################")
    #             break
    #         else:
    #             counter += 1
    #             writer.writerow(line)
                
    #     writer.writerows(reader)





    
    
    
    