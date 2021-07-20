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
inputSize = config.nbrOfPCAcomponents
nbrWs = config.nbrWs
nbrOfNormalDists = config.nbrOfNormalDists
nbrOfPoles = config.nbrOfPoles
sizeOfTraining = config.trainingPoints 
sizeOfValidation = config.validationPoints
outputSize = nbrWs + (4 * nbrOfPoles)
print("NN input size {}, output size {} with {} distributions, {} poles".format(inputSize,outputSize,nbrOfNormalDists,nbrOfPoles))

#Load the saved NN model (made in train_ACANN.py)
saved = "savedNNmodel.pth"
#Note: Make sure the dimensions are the same
model = ACANN(inputSize,outputSize,[100,200,300,200],drop_p=0.09).double()
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
    # print("output:",predicData)

#Evaluate output:
# nbrOfNormalDists = 2
# nbrOfPoles = 2
ws = np.linspace(0.01,5,nbrWs)

_norm_pdf_C = np.sqrt(2*np.pi)
zlim = 37

#Calculates the pdf of a normal distribution
def custompdf(w,mean,std):
    z = (w-mean)/std
    if z < zlim and z > -zlim:
        return np.exp(-(w-mean)**2/(2.0*(std**2))) / (_norm_pdf_C*std)
    else:
        return 0

#For each test sample
def eval_output(predicData):
    #For each testing value in the batch:
    reconstructedList = []
    # print(predicData[0])
    # print(actual[0])
    # print(len(actual[0]))
    for i in range(len(predicData)):
        rhoReconstructed = []
        #Reconstruct the spectral density function
        for distr in range(nbrOfNormalDists):
            #Mean, standard deviation and weight of normal distribution
            meani = predicData[i][3*distr]
            #Stddev has to be positive
            stdi = abs(predicData[i][3*distr + 1])
            wi = predicData[i][3*distr + 2]
            
            # print("mean_{}:".format(distr),meani.round(5), \
            #       "   std_{}:".format(distr),stdi.round(5),"   w_{}:".format(distr),wi.round(5))
            
            if (distr == 0):
                for w in ws:
                    rhoReconstructed.append(wi*custompdf(w,meani,stdi))
                    
            else:            
                for j in range(len(ws)):
                    rhoReconstructed[j] = rhoReconstructed[j] + wi*custompdf(ws[j],meani,stdi)
        
        polesReconstructed = []
        #Reconstruct complex poles
        skipNbr = 3 * nbrOfNormalDists
        for pole in range(nbrOfPoles):
            #Real and Imaginary part of poles and residues
            qiRe = predicData[i][4*pole + skipNbr]
            qiIm = predicData[i][4*pole + skipNbr + 1]
            RiRe = predicData[i][4*pole + skipNbr + 2]
            RiIm = predicData[i][4*pole + skipNbr + 3]
            
            # print("Re(q_{}):".format(pole),qiRe.round(5), \
            #       "Im(q_{}):".format(pole),qiIm.round(5), \
            #         "Re(R_{}):".format(pole),RiRe.round(5), \
            #         "Im(R_{}):".format(pole),RiIm.round(5))
            
            recPole = [qiRe,qiIm,RiRe,RiIm]
            #polesReconstructed is the list of poles of the current test sample
            # polesReconstructed.append(recPole)
            polesReconstructed = polesReconstructed + recPole
    
    
        reconstructedList.append(rhoReconstructed + polesReconstructed)
        #Actual values (length 200 + 12 : ws + 3 poles):
        #Calculate difference with actual values:
        #Difference in spectral density function:
        # for k in range(len(ws)):
        #     MAE += abs(rhoReconstructed[k] - actual[i][k])
        # #Difference in poles:
        # actualPoles = actual[i][-4*nbrOfPoles:]
        # for k in range(4*nbrOfPoles):
        #     MAE += abs(polesReconstructed[k] - actualPoles[k])
            
    return reconstructedList
    
    
plot = True
if plot:
    #Plot the density function
    #currently only the last test sample
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    fig.suptitle("Reconstructed spectral density functions")
    i = 0
    step = int(sizeOfValidation/9)
    
    ax1.plot(ws,rhovaluesList[i][:nbrWs],label="Original")
    ax1.plot(ws,predicData[i][:nbrWs],label="Reconstructed")
    ax1.legend()
    ax1.set_xlabel("ω")
    ax1.set_ylabel("ρ(ω)")
    
    ax2.plot(ws,rhovaluesList[i+step][:nbrWs],label="Original")
    ax2.plot(ws,predicData[i+step][:nbrWs],label="Reconstructed")
    ax2.legend()
    ax2.set_xlabel("ω")
    ax2.set_ylabel("ρ(ω)")
    
    ax3.plot(ws,rhovaluesList[i+2*step][:nbrWs],label="Original")
    ax3.plot(ws,predicData[i+2*step][:nbrWs],label="Reconstructed")
    ax3.legend()
    ax3.set_xlabel("ω")
    ax3.set_ylabel("ρ(ω)")
    
    ax4.plot(ws,rhovaluesList[i+3*step][:nbrWs],label="Original")
    ax4.plot(ws,predicData[i+3*step][:nbrWs],label="Reconstructed")
    ax4.legend()
    ax4.set_xlabel("ω")
    ax4.set_ylabel("ρ(ω)")
    
        # plt.title("Reconstructed spectral density function")
    
    #Plot the poles
    #currently only the ith test sample
    plotPoles = False
    
    if plotPoles:
        plt.figure()
        for j in range(nbrOfPoles):
            #Only plot the poles
            aj = predicData[i][nbrWs + 4*j]
            bj = predicData[i][nbrWs + 4*j + 1]
            cj = predicData[i][nbrWs + 4*j + 2]
            dj = predicData[i][nbrWs + 4*j + 3]
            ajOrig = rhovaluesList[i][nbrWs + 4*j]
            bjOrig = rhovaluesList[i][nbrWs + 4*j + 1]
            cjOrig = rhovaluesList[i][nbrWs + 4*j + 2]
            djOrig = rhovaluesList[i][nbrWs + 4*j + 3]
            
            
            plt.plot(ajOrig,bjOrig,"o",color="red")
            plt.plot(aj,bj,"o",color="magenta")
            plt.plot(cjOrig,djOrig,"o",color="blue")
            plt.plot(cj,dj,"o",color="cyan")
            
        
            # plt.text(ajOrig,bjOrig+0.1,"({}, {})".format(round(ajOrig,2),round(bjOrig,2)))
            # plt.text(aj,bj+0.1,"({}, {})".format(round(aj,2),round(bj,2)))
            
            # plt.text(cjOrig,djOrig+0.1,"({}, {})".format(round(cjOrig,2),round(djOrig,2)))
            # plt.text(cj,dj+0.1,"({}, {})".format(round(cj,2),round(dj,2)))
            
        #Again for correct legend:
        plt.plot(cjOrig,djOrig,"o",color="blue",label="Original poles")
        plt.plot(cj,dj,"o",color="cyan",label="Reconstructed poles")
        plt.plot(ajOrig,bjOrig,"o",color="red",label="Original residues")
        plt.plot(aj,bj,"o",color="magenta",label="Reconstructed residues")
        
        plt.xlim(-4,4)
        plt.ylim(-4,4)
        plt.legend()
        plt.grid()
        plt.xlabel("Re(q)")
        plt.ylabel("Im(q)")
        plt.title("Reconstructed complex poles and residues")
    

        
    
    
 





