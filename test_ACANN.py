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
import scipy.stats as stats
import matplotlib.pyplot as plt
import inputParameters as config

#Load input parameters from inputParameters.py
nbrOfNormalDists = config.nbrOfNormalDists
nbrOfPoles = config.nbrOfPoles
sizeOfTraining = config.trainingPoints 
sizeOfValidation = config.validationPoints
outputSize = (3 * nbrOfNormalDists) + (4 * nbrOfPoles)
print("NN output size {} with {} distributions, {} poles".format(outputSize,nbrOfNormalDists,nbrOfPoles))

#Load the saved NN model (made in train_ACANN.py)
saved = "savedNNmodel.pth"
#Note: Make sure the dimensions are the same
model = ACANN(200,outputSize,[200,200,200],drop_p=0.09).double()
model.load_state_dict(torch.load(saved))
model.eval()



#Load test data
path = "C:/Users/Thibault/Documents/Universiteit/Honours/Deel 2, interdisciplinair/Code/NN/Datasets/"
test_data = Database(csv_target= path + "rhoTest.csv", \
                     csv_input= path + "DTest.csv",nb_data=sizeOfValidation).get_loader()
testloader = DataLoader(test_data,batch_size=sizeOfValidation)

testloadList = list(testloader)
# print("Test data:",testloadList)
# print("Rho test:",testloadList[0][1])
#Convert tensor to numpy array:
rhovaluesList = testloadList[0][1].to("cpu").numpy()
# print("Rho test values:",rhovaluesList)
print(len(rhovaluesList),"testing points")
print("Data Loaded")

#Use NN to predict
with torch.no_grad():
    D_test,rho_test = next(iter(testloader))
    prediction = model.forward(D_test)
    # print("output:",prediction)
    predicData = prediction.to("cpu").numpy()
    print("output:",predicData)

#Evaluate output:
nbrOfNormalDists = 2
nbrOfPoles = 2
ws = np.linspace(0.01,0.2,200)

#For each test sample
reconstructedDensitiesList = []
polesList = []

for i in range(len(rhovaluesList)):
    rhoReconstructed = []
    #Reconstruct the spectral density function
    for distr in range(nbrOfNormalDists):
        #Mean, standard deviation and weight of normal distribution
        meani = predicData[i][3*distr]
        #Stddev has to be positive
        stdi = abs(predicData[i][3*distr + 1])
        wi = predicData[i][3*distr + 2]
        
        print("mean_{}:".format(distr),meani.round(5), \
              "   std_{}:".format(distr),stdi.round(5),"   w_{}:".format(distr),wi.round(5))
        
        #TODO: find fix for this
        #       maybe just set to 0 (~ignore) if underflow happens
        #exp underflow when ws >> standard deviation stdi
        if (distr == 0):
            rhoReconstructed = wi*stats.norm.pdf(ws,meani,stdi)
        else:            
            rhoReconstructed += wi*stats.norm.pdf(ws,meani,stdi)
            
    reconstructedDensitiesList.append(rhoReconstructed)
            
    polesReconstructed = []
    #Reconstruct complex poles
    skipNbr = 3 * nbrOfNormalDists
    for pole in range(nbrOfPoles):
        #Real and Imaginary part of poles and residues
        qiRe = predicData[i][4*pole + skipNbr]
        qiIm = predicData[i][4*pole + skipNbr + 1]
        RiRe = predicData[i][4*pole + skipNbr + 2]
        RiIm = predicData[i][4*pole + skipNbr + 3]
        
        print("Re(q_{}):".format(pole),qiRe.round(5), \
              "Im(q_{}):".format(pole),qiIm.round(5), \
                "Re(R_{}):".format(pole),RiRe.round(5), \
                "Im(R_{}):".format(pole),RiIm.round(5))
        
        recPole = [qiRe,qiIm,RiRe,RiIm]
        #polesReconstructed is the list of poles of the current test sample
        polesReconstructed.append(recPole)
        #polesList holds the lists of list of poles for all test samples
        polesList.append(polesReconstructed)
    
    
plot = False
if plot:
    #Plot the density function
    #currently only the last test sample
    plt.figure()
    # print(reconstructedDensitiesList[-1])
    plt.plot(ws,reconstructedDensitiesList[-1])
    plt.xlabel("ω")
    plt.ylabel("ρ(ω)")
    plt.title("Reconstructed spectral density function")
    
    #Plot the poles
    #currently only the last test sample
    plt.figure()
    for i in range(nbrOfPoles):
        #Only plot the poles
        qiRe = polesList[-1][i][0]
        qiIm = polesList[-1][i][1]
        plt.plot(qiRe,qiIm,"o",color="C0")
        plt.text(qiRe,qiIm+0.1,"({}, {})".format(round(qiRe,2),round(qiIm,2)))
        
            
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.grid()
    plt.xlabel("Re(q)")
    plt.ylabel("Im(q)")
    plt.title("Reconstructed complex poles")
    
    #TODO: compare with original density function


        
    
    
 





