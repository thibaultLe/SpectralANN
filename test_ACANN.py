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

import robustnessCheck

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
print("NN input size {}, output size {} plus {} poles and sigma".format(inputSize,outputSize-4*nbrOfPoles-1,nbrOfPoles))

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
# print("Test data:",testloadList)
# print("Rho test:",testloadList[0][1])
#Convert tensor to numpy array:
rhovaluesList = testloadList[0][1].to("cpu").numpy()
print(len(rhovaluesList),"testing points")

prop_data = pd.read_csv(path+'DTestRaw.csv',header=None,nrows=sizeOfValidation)
propList = prop_data.values.tolist()
print(len(propList),"propagators")

params_data = pd.read_csv(path+'params.csv',header=None,nrows=sizeOfTraining+2*sizeOfValidation)
paramsList = params_data.values.tolist()

print("Data Loaded")

#Use NN to predict
with torch.no_grad():
    D_test,rho_test = next(iter(testloader))
    prediction = model.forward(D_test)
    # print("output:",prediction)
    predicData = prediction.to("cpu").numpy()
    # print("output:",predicData)

#Evaluate output:
ps = np.linspace(pstart,pend,nbrPoints)
ws = np.linspace(0.01,10,nbrWs)

_norm_pdf_C = np.sqrt(2*np.pi)
zlim = 37

#Calculates the pdf of a normal distribution
def custompdf(w,mean,std):
    z = (w-mean)/std
    if z < zlim and z > -zlim:
        return np.exp(-(w-mean)**2/(2.0*(std**2))) / (_norm_pdf_C*std)
    else:
        return 0

    
plot = False
if plot:
    #Plot the density function
    #currently only the last test sample
    amountOfTests = 5
    for i in range(0,round(len(rhovaluesList)),round(len(rhovaluesList)/amountOfTests)):
        
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
        fig.suptitle("Reconstructed spectral density functions")
        step = int(sizeOfValidation/sizeOfValidation)
        
        print("\ni =",i,"and",i+step,"out of",len(rhovaluesList))
        print("Params:",paramsList[4::8][i])
        
        ax2.plot(ws,rhovaluesList[i][:nbrWs],label="Original")
        ax2.plot(ws,predicData[i][:nbrWs],label="Reconstructed")
        ax2.legend()
        ax2.set_xlabel("ω")
        ax2.set_ylabel("ρ(ω)")
        
        print("Params:",paramsList[4::8][i+step])
        
        ax4.plot(ws,rhovaluesList[i+step][:nbrWs],label="Original")
        ax4.plot(ws,predicData[i+step][:nbrWs],label="Reconstructed")
        ax4.legend()
        ax4.set_xlabel("ω")
        ax4.set_ylabel("ρ(ω)")
        
        ax1.plot(ps,propList[i],label="Propagator")
        # ax3.plot(ps,predicData[i+2*step][:nbrWs],label="Reconstructed")
        ax1.legend()
        ax1.set_xlabel("p")
        ax1.set_ylabel("D(p²)")
        
        ax3.plot(ps,propList[i+step],label="Propagator")
        # ax4.plot(ws,predicData[i+3*step][:nbrWs],label="Reconstructed")
        ax3.legend()
        ax3.set_xlabel("p")
        ax3.set_ylabel("D(p²)")
    
        # plt.title("Reconstructed spectral density function")
    
    #Plot the poles
    #currently only the ith test sample
    plotPoles = True
    if plotPoles:
        polemarkers = ["^","o","*"]
        for i in range(0,round(len(rhovaluesList)),round(len(rhovaluesList)/amountOfTests)):
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
                
                # plt.plot(ajOrig,bjOrig,"o",color="green",markersize=10)
                # plt.plot(aj,bj,"o",color="lawngreen",markersize=10)
                
                plt.plot(cjOrig,djOrig,polemarkers[j],color="blue",markersize=15)
                plt.plot(cj,dj,polemarkers[j],color="cyan",markersize=15)
                
                
            #Again for correct legend:
            plt.plot(cjOrig,djOrig,polemarkers[j],color="blue",label="Original poles",markersize=15)
            plt.plot(cj,dj,polemarkers[j],color="cyan",label="Reconstructed poles",markersize=15)
            
            # plt.plot(ajOrig,bjOrig,"o",color="green",label="Original residues",markersize=10)
            # plt.plot(aj,bj,"o",color="lawngreen",label="Reconstructed residues",markersize=10)
            
            plt.xlim(-6,6)
            plt.ylim(-6,6)
            plt.legend()
            plt.grid()
            plt.xlabel("Re(q)")
            plt.ylabel("Im(q)")
            plt.title("Reconstructed complex poles and residues")
    
    
    

from matplotlib.legend_handler import HandlerTuple

    
def plotPolesForIndex(i,ax):
    polemarkers = ["o","^","*"]
    msizes = [7,9,11]
    for j in range(nbrOfPoles):
        #Only plot the poles
        cj = predicData[i][nbrWs + 4*j + 2]
        dj = predicData[i][nbrWs + 4*j + 3]
        cjOrig = rhovaluesList[i][nbrWs + 4*j + 2]
        djOrig = rhovaluesList[i][nbrWs + 4*j + 3]
        
        ax.plot(cjOrig,djOrig,polemarkers[j],color="blue",label="Original poles",markersize=msizes[j])
        ax.plot(cj,dj,polemarkers[j],color="cyan",label="Reconstructed poles",markersize=msizes[j])
        
                
                
    ax.set_xlim([-7,7])
    ax.set_ylim([0,7])
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
        ajOrig = rhovaluesList[i][nbrWs + 4*j]
        bjOrig = rhovaluesList[i][nbrWs + 4*j + 1]
        
        ax.plot(ajOrig,bjOrig,marker=resmarkers[j],color="green",label="Original residues",markersize=msizes[j])
        ax.plot(aj,bj,marker=resmarkers[j],color="lawngreen",label="Reconstructed residues",markersize=msizes[j])
        
        #Draw lines in between:
        # ax.plot([ajOrig,aj],[bjOrig,bj],color="green")
        
    
    ax.set_xlim([-7,7])
    ax.set_ylim([0,7])
    ax.grid()
    ax.set_xlabel("Re(q)")
    ax.set_ylabel("Im(q)")
    
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
            spectrFunc.append(predicData[index][i]/(p+ws[i]))
        
        integral = integrate.simpson(spectrFunc,x=ws[wscutoff:])
        prop = integral + poles(p,3, predicData[index][nbrWs:nbrWs+12])
        reconstructedPropSigma.append(prop)
    
    return reconstructedPropSigma
    
    
    
getBestAndWorst = True
if getBestAndWorst:
    #Get the best and worst test cases:
    maxMAEindex = 0
    maxMAE = 0
    minMAEindex = 0
    minMAE = 100000
    
    fullSortedList = []
    
    for i in range(len(rhovaluesList)):
        MAE = 0
        
        #MAE on propagator
        reconProp = reconstructProp(i)
        combListProp = zip(propList[i],reconProp)
        
        #MAE on spectral function
        combList = zip(rhovaluesList[i][:nbrWs],predicData[i][:nbrWs])
        
        scale = abs(max(rhovaluesList[i][:nbrWs]))
        for orig, recon in combList:
            MAE += abs(orig-recon)/(scale)
            
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
    
    fullSortedList.sort()
    
    
    print("Min. MAE:",minMAE)
    print("Max. MAE:",maxMAE)

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
    
    
    ax11.plot(ps,propList[minMAEindex],label="Propagator")
    ax11.plot(ps,reconstructProp(minMAEindex),"--",label="Reconstructed propagator",color="red")
    ax11.set_xlabel("p")
    ax11.set_ylabel("D(p²)")
    
    ax12.plot(ws,rhovaluesList[minMAEindex][:nbrWs],label="Spectral function")
    ax12.plot(ws,predicData[minMAEindex][:nbrWs],"--",label="Reconstructed spectral function",color="red")
    ax12.set_xlabel("ω")
    ax12.set_ylabel("ρ(ω)")
    
    ax21.plot(ps,propList[percentile25th],label="Propagator")
    ax21.plot(ps,reconstructProp(percentile25th),"--",label="Reconstructed propagator",color="red")
    ax21.set_xlabel("p")
    ax21.set_ylabel("D(p²)")
    
    ax22.plot(ws,rhovaluesList[percentile25th][:nbrWs],label="Spectral function")
    ax22.plot(ws,predicData[percentile25th][:nbrWs],"--",label="Reconstructed spectral function",color="red")
    ax22.set_xlabel("ω")
    ax22.set_ylabel("ρ(ω)")
    
    ax31.plot(ps,propList[percentile50th],label="Propagator")
    ax31.plot(ps,reconstructProp(percentile50th),"--",label="Reconstructed propagator",color="red")
    ax31.set_xlabel("p")
    ax31.set_ylabel("D(p²)")
    
    ax32.plot(ws,rhovaluesList[percentile50th][:nbrWs],label="Spectral function")
    ax32.plot(ws,predicData[percentile50th][:nbrWs],"--",label="Reconstructed spectral function",color="red")
    ax32.set_xlabel("ω")
    ax32.set_ylabel("ρ(ω)")
    
    ax41.plot(ps,propList[percentile75th],label="Propagator")
    ax41.plot(ps,reconstructProp(percentile75th),"--",label="Reconstructed propagator",color="red")
    ax41.set_xlabel("p")
    ax41.set_ylabel("D(p²)")
    
    ax42.plot(ws,rhovaluesList[percentile75th][:nbrWs],label="Spectral function")
    ax42.plot(ws,predicData[percentile75th][:nbrWs],"--",label="Reconstructed spectral function",color="red")
    ax42.set_xlabel("ω")
    ax42.set_ylabel("ρ(ω)")
    
    ax51.plot(ps,propList[maxMAEindex],label="Propagator")
    ax51.plot(ps,reconstructProp(maxMAEindex),"--",label="Reconstructed propagator",color="red")
    ax51.set_xlabel("p")
    ax51.set_ylabel("D(p²)")
    
    ax52.plot(ws,rhovaluesList[maxMAEindex][:nbrWs],label="Spectral function")
    ax52.plot(ws,predicData[maxMAEindex][:nbrWs],"--",label="Reconstructed spectral function",color="red")
    ax52.set_xlabel("ω")
    ax52.set_ylabel("ρ(ω)")
    
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
    
    
    
    # index = percentile75th
    
    
    # plt.figure()
    # plt.plot(ps,propList[index],label="Original propagator")
    # plt.plot(ps,reconstructedProp,label="Reconstructed propagator,sigma=0")
    # plt.plot(ps,reconstructProp(index),label="Reconstructed propagator")
    # plt.legend()
    
    
    
    
    

    
    
 





