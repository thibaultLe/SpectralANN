# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 12:04:32 2021

@author: Thibault
"""


from Database import Database
from torch.utils.data import DataLoader
import inputParameters as config
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import pandas as pd

noiseSize = 1e-3

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
    noise = np.random.normal(0,noiseSize,nbrPoints)
    alldata[i] = alldata[i] + noise
    
for i in range(len(alldataValid)):
    noise = np.random.normal(0,noiseSize,nbrPoints)
    alldataValid[i] = alldataValid[i] + noise

for i in range(len(alldataTest)):
    noise = np.random.normal(0,noiseSize,nbrPoints)
    alldataTest[i] = alldataTest[i] + noise


print(len(alldata),"training points")


# ps = np.geomspace(pstart,1,nbrPoints)
#When using logspace, the fit is atrocious -> use linspace
ps = np.linspace(-1,1,nbrPoints)


#Calculates the MAE for combinations of maxdegrees and nbr of PCA components
#Index is the index of the propagator we calculate the MAE for
def MAE(ps,index):
    
    maxdegrees = [25,30,35]
    # maxdegrees = [30]
    nbrPCAcomps = [10,15,20]
    # nbrPCAcomps = [5]
    
    MAEs = []
    for maxdegree in maxdegrees:
        coefficientsList = []
        for i in range(len(alldata)):
            coefficientsList.append(np.polynomial.legendre.legfit(ps,alldata[i],maxdegree))
        
        scaler=StandardScaler()#instantiate
        scaler.fit(coefficientsList) # compute the mean and standard which will be used in the next command
        X_scaled=scaler.transform(coefficientsList)
        
        for nbrComps in nbrPCAcomps:
            if nbrComps < maxdegree:
                pca=PCA(n_components=nbrComps) 
                pca.fit(X_scaled) 
                
                np.random.seed(2)
                noise = np.random.normal(1,0.1,len(ps))
                #Possible alternative: adding instead of multiplying
                propWithNoise = alldata[index] * noise
                
                # plt.plot(ps,propWithNoise,"o",label="Noisy propagator")
                pWNlegfit = np.polynomial.legendre.legfit(ps,propWithNoise,maxdegree)
                # plt.plot(ps,np.polynomial.legendre.legval(ps,pWNlegfit),label="Legendre fit to noisy propagator")
                
                pWNlegfitreshaped = pWNlegfit.reshape(1,-1)
                noisePCAd = pca.transform(scaler.transform(pWNlegfitreshaped))
                noisePCAreconstructed = scaler.inverse_transform(pca.inverse_transform(noisePCAd))
                # plt.plot(ps,np.polynomial.legendre.legval(ps,noisePCAreconstructed[0])
                
                actualYs = alldata[index]
                pcaReconstructedYs = np.polynomial.legendre.legval(ps,noisePCAreconstructed[0])
                
                MAE = 0
                for i in range(len(actualYs)):
                    MAE += abs(actualYs[i] - pcaReconstructedYs[i])
                MAEs.append([maxdegree,nbrComps,MAE])
    return MAEs

def most_frequent(List):
    return max(set(List), key = List.count)

# MAEcounter = []
# for i in range(0,len(alldata),150):
#     MAEs = MAE(ps,i)
#     minMAE = MAEs[0]
#     for MAElist in MAEs:
#         # print("MaxDeg {}, PCs {}, MAE {}".format(MAElist[0],MAElist[1],round(MAElist[2],4)))
        
#         if MAElist[2] < minMAE[2]:
#             minMAE = MAElist
#     MAEcounter.append((minMAE[0],minMAE[1]))
#     print("Idx: {}, Minimal MAE: MaxDeg {}, PCs {}, MAE {}".format(i,minMAE[0],minMAE[1],round(minMAE[2],4)))

# print("Most frequent:",most_frequent(MAEcounter))


#Fit with legendre, build dataset of coefficients & Run PCA

#maxdegree of 30 = 31 values
maxdegree = maxDegreeOfLegFit

#Legendre fit to all propagators in training set, keeps coefficients
coefficientsList = []
for i in range(len(alldata)):
    coefficientsList.append(np.polynomial.legendre.legfit(ps,alldata[i],maxdegree))


x = coefficientsList

#Normalise the attributes
scaler=StandardScaler()#instantiate
scaler.fit(x) # compute the mean and standard which will be used in the next command
X_scaled=scaler.transform(x)
# print("Means:")
# print(scaler.mean_)
# print("Variances:")
# print(scaler.var_)

pca=PCA(n_components=nbrOfPCAcomponents) 
pca.fit(X_scaled) 
X_pca=pca.transform(X_scaled) 


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


""" 
Use same scaling and PCA coefficients as on the training set to scale
the validation and test set. Then write these PCA'd coefficients to files.
"""

coefficientsListValid = []
for i in range(len(alldataValid)):
    coefficientsListValid.append(np.polynomial.legendre.legfit(ps,alldataValid[i],maxdegree))

#Normalise the attributes
X_scaled=scaler.transform(coefficientsListValid)
X_pcaValid=pca.transform(X_scaled) 

coefficientsListTest = []
for i in range(len(alldataTest)):
    coefficientsListTest.append(np.polynomial.legendre.legfit(ps,alldataTest[i],maxdegree))

#Normalise the attributes
X_scaled=scaler.transform(coefficientsListTest)
X_pcaTest=pca.transform(X_scaled) 

#Write data to files
propTraindf = pd.DataFrame(X_pca)
propTraindf.to_csv(propTrain_csv,index=False,header=False,mode='a')

propValiddf = pd.DataFrame(X_pcaValid)
propValiddf.to_csv(propValid_csv,index=False,header=False,mode='a')


print("Data conversion to PCA components succesfull.")


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
    
    
    
    # plt.figure()
    # plt.plot(psT,dp2s,"o")
    # plt.figure()
    # plt.plot(psT,dp2s)
    
    # psScaled = np.linspace(-1,1,len(psT))
    from scipy.interpolate import interp1d
    
    psNew = np.linspace(pstart,pend,nbrPoints)
    dp2sFunc = interp1d(psT,dp2s)
    dp2sInter = dp2sFunc(psNew)
    
    coefficients = np.polynomial.legendre.legfit(ps,dp2sInter,maxdegree).reshape(1,-1)
    
    
    X_scaled=scaler.transform(coefficients)
    X_pcaTestActual=pca.transform(X_scaled) 
    
    
    print(X_pcaTestActual)
    # print(len(X_pcaTestActual))
    # print(X_pcaTest)
    # print(len(X_pcaTest))
    X_pcaTest[-1] = X_pcaTestActual
    
    import csv
    with open(path+'DTestRaw.csv') as inf:
        reader = csv.reader(inf.readlines())

    with open(path+'DTestRaw.csv', 'w') as outf:
        writer = csv.writer(outf)
        counter = 0
        for line in reader:
            if counter == 17949:
                writer.writerow(dp2sInter)
                print("########################")
                print("Wrote dp2sInter")
                print("########################")
                print("########################")
                break
            else:
                counter += 1
                writer.writerow(line)
                
        writer.writerows(reader)

propTestdf = pd.DataFrame(X_pcaTest)
propTestdf.to_csv(propTest_csv,index=False,header=False,mode='a')
# print(X_pcaTest[-1])




#Restore after pca = inverse scaling after inverse PCA 
# x_restore = scaler.inverse_transform(pca.inverse_transform(X_pca))

# plt.figure()
# plt.plot(ps,np.polynomial.legendre.legval(ps,x[i]),label="Original Legendre fitted")
# plt.plot(ps,np.polynomial.legendre.legval(ps,x_restore[i]),label="PCA reconstruction")
# plt.title("PCA reconstruction")
# plt.legend()


"""
Visual noise removal and PCA reconstruction test:
"""
visualPlot = True
if visualPlot:
    # i = 7650
    i=0
    
    plt.figure()
    plt.plot(ps,alldata[i],label="Original propagator")
    # plt.plot(ps,np.polynomial.legendre.legval(ps,x[i]),label="Propagator Legendre fit")
    # plt.xlim(ps[0]-0.5,ps[-1]+0.5)
    # plt.ylim(min(alldata[i])-1,max(alldata[i])+10)
    
    np.random.seed(2)
    # noise = np.random.normal(1,0.000001,len(alldataTest[0]))
    #Possible alternative: adding instead of multiplying
    propWithNoise = alldata[i]
    
    plt.plot(ps,propWithNoise,"o",label="Noisy propagator")
    pWNlegfit = np.polynomial.legendre.legfit(ps,propWithNoise,maxdegree)
    
    ps = np.linspace(-1,1,500)
    plt.plot(ps,np.polynomial.legendre.legval(ps,pWNlegfit),label="Legendre fit to noisy propagator")
    
    pWNlegfitreshaped = pWNlegfit.reshape(1,-1)
    noisePCAd = pca.transform(scaler.transform(pWNlegfitreshaped))
    # print("PCA cfts on original:", X_pca[i])
    print("PCA cfts for noisy prop:",noisePCAd)
    
    psActual = np.linspace(pstart,pend,nbrPoints)
    # PCAreconstructed = scaler.inverse_transform(pca.inverse_transform(X_pca[i]))
    # # print("PCA reconstructed:",PCAreconstructed)
    # plt.plot(psActual,np.polynomial.legendre.legval(ps,PCAreconstructed),label="PCA reconstruction of original")
    
    # noisePCAd = pca.transform(scaler.transform(pWNlegfit))
    noisePCAreconstructed = scaler.inverse_transform(pca.inverse_transform(noisePCAd))
    plt.plot(ps,np.polynomial.legendre.legval(ps,noisePCAreconstructed[0]),label="PCA reconstruction of noisy")
    
    print("Original Leg cfts:",pWNlegfit)
    print("Reconstructed:",noisePCAreconstructed[0])
    
    #Adding noise to pWNlegfit coefficients:
    # legnoise = np.random.normal(1,0.01,len(pWNlegfit))
    # print(legnoise[:10])
    # legnoiseCfts = pWNlegfit * legnoise
    # plt.plot(ps,np.polynomial.legendre.legval(ps,legnoiseCfts),label="Noisy legendre fit")
    # print("Noisy legendre:",legnoiseCfts)
    plt.title("Denoising with PCA")
    plt.legend()
    
    #Calculate max error of PCA reconstruction:
    maxError = 0
    avgError = 0
    reconstructed = np.polynomial.legendre.legval(ps,noisePCAreconstructed[0])
    print(reconstructed[::5])
    print(propWithNoise)
    for i in range(len(propWithNoise)):
        diff = abs(propWithNoise[i] - reconstructed[::5][i])
        avgError += diff
        if diff > maxError:
            maxError = diff
    print("Max error of reconstruction:",maxError)
    print("Average error:",avgError/len(propWithNoise))

#Problem: huge errors after p = 1
#Try to scale input data to improve ill conditioning:
#Resolved: x interval needs to be between -1 and 1 for optimal conditioning
    
"""
With scaled input data:
"""
# #Scale input data in attempt to improve conditioning of fitting
# scalerD=StandardScaler()#instantiate
# scalerD.fit(alldata) # compute the mean and standard which will be used in the next command
# scaledRawData = scalerD.transform(alldata)

# coefficientsList = []
# for i in range(len(alldata)):
#     coefficientsList.append(np.polynomial.legendre.legfit(ps,scaledRawData[i],maxdegree))
# x = coefficientsList

# #Normalise the attributes
# scaler=StandardScaler()#instantiate
# scaler.fit(x) # compute the mean and standard which will be used in the next command
# X_scaled=scaler.transform(x)

# pca=PCA(n_components=10) 
# pca.fit(X_scaled) 
# X_pca=pca.transform(X_scaled) 

# plt.figure()
# # plt.plot(ps,alldata[i],label="Original propagator")
# # plt.plot(ps,np.polynomial.legendre.legval(ps,x[i]),label="Propagator Legendre fit")
# plt.xlim(ps[0]-0.5,ps[-1]+0.5)
# # plt.ylim(min(scaledRawData[i])-1,max(scaledRawData[i])+1)

# np.random.seed(2)
# noise = np.random.normal(1,0.01,len(ps))
# #Possible alternative: adding instead of multiplying
# propWithNoise = scaledRawData[i] * noise
# pWNlegfit = np.polynomial.legendre.legfit(ps,propWithNoise,maxdegree)
# plt.plot(ps,np.polynomial.legendre.legval(ps,pWNlegfit),label="Legendre fit to propagator")

# pWNlegfitreshaped = pWNlegfit.reshape(1,-1)
# noisePCAd = pca.transform(scaler.transform(pWNlegfitreshaped))
# # noisePCAd = pca.transform(scaler.transform(pWNlegfit))
# noisePCAreconstructed = scaler.inverse_transform(pca.inverse_transform(noisePCAd))
# plt.plot(ps,np.polynomial.legendre.legval(ps,noisePCAreconstructed[0]),label="PCA reconstruction")
# print("Reconstructed PCA cfts:",noisePCAreconstructed[0])

# # print("Original:     ",pWNlegfit)
# #Adding noise to pWNlegfit coefficients:
# legnoise = np.random.normal(1,0.0001,len(pWNlegfit))
# print(legnoise[:10])
# legnoiseCfts = pWNlegfit * legnoise
# plt.plot(ps,np.polynomial.legendre.legval(ps,legnoiseCfts),label="Noisy legendre coefficients")
# print("Noisy legendre:",legnoiseCfts)
# plt.plot(ps,scaledRawData[i],label="Original")
# # plt.plot(ps,propWithNoise,label="Propagator with noise")
# plt.title("Denoising with PCA on scaled data")
# plt.legend()




# plt.figure()
# Xax=X_pca[:,0]
# Yax=X_pca[:,1]
# plt.scatter(Xax,Yax,s=12,alpha=0.5)
# plt.title("Instance space of propagators")
# plt.xlabel("First principal component")
# plt.ylabel("Second principal component")


# print("Explained variance of the principal components", \
#       str(format(sum(pca.explained_variance_ratio_),".5f")))
# print(pca.explained_variance_ratio_)
# print(pca.singular_values_)
# print("Coefficients of each principal component")
# print(pca.components_)
# compList = []
# for axis in pca.components_:
#     axisList =[]
#     for elem in axis:
#         axisList.append(elem)
#     compList.append(axisList)
# print(compList)

SHOW_CFTS = False
if SHOW_CFTS:
    plt.matshow(pca.components_,cmap='RdYlGn')
    plt.yticks([0,1],['1st Comp','2nd Comp'],fontsize=10)
    plt.colorbar()
    plt.clim(-1,1)
    for (i, j), z in np.ndenumerate(pca.components_):
        plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
    # plt.title("Coefficients of the first 2 principal components",y=-0.4,fontsize=14)
    plt.show()





# #Analysis: MAE vs maximum degree of legendre fit
# maxdegrees = [5,10,20,25,35,50,64,100]
# residualsList = []
# MAEsList = []
# for maxdegree in maxdegrees:
#     residuals = []
#     MAEs = []
#     coeffs = np.polynomial.legendre.legfit(ps,alldata[i],maxdegree)
#     dls = np.polynomial.legendre.legval(ps,coeffs)
        
#     for j in range(len(dls)):
#         residuals.append(dls[j]-alldata[i][j])
#         MAEs.append(abs(dls[j]-alldata[i][j]))
        
#     residualsList.append(residuals)
#     MAEsList.append(sum(MAEs))
    
# for i in range(len(maxdegrees)):
#     print("Max Degree {},".format(maxdegrees[i]),"MAE:",MAEsList[i])
    
# plt.figure()
# plt.xlabel("Max degree of Legendre fit")
# plt.ylabel("MAE of fit")
# plt.plot(maxdegrees,MAEsList)
# plt.title("MAE vs maximum degree of Legendre fit")

#Splines
# from scipy import interpolate
# f = interpolate.splrep(ps,alldata[2],k=3)
# # print(f)
# ys = interpolate.splev(ps,f)
# plt.plot(ps,ys,label="Splines")

#Padé (or small attempt at padé)
# def rational(x, p, q):
#     """
#     The general rational function description.
#     p is a list with the polynomial coefficients in the numerator
#     q is a list with the polynomial coefficients (except the first one)
#     in the denominator
#     The zeroth order coefficient of the denominator polynomial is fixed at 1.
#     Numpy stores coefficients in [x**2 + x + 1] order, so the fixed
#     zeroth order denominator coefficent must comes last. (Edited.)
#     """
#     return np.polyval(p, x) / np.polyval(q + [1.0], x)

# def rational3_3(x, p0, p1, p2, q1, q2):
#     return rational(x, [p0, p1, p2], [q1, q2])

# from scipy.optimize import curve_fit
# x = alldata[2]
# # y = rational(ps, [-0.2, 0.3, 0.5], [-1.0, 2.0])
# # y = rational(ps)
# # ynoise = y * (1.0 + np.random.normal(scale=0.01, size=x.shape))
# popt,pcov = curve_fit(rational3_3, ps,alldata[2])
# print(popt)

# plt.figure()
# # plt.plot(x, y, label='original')
# # plt.plot(x, ynoise, '.', label='data')
# plt.plot(ps, rational3_3(ps, *popt), label='Rational 3/3 fit')
# plt.plot(ps,alldata[2],label="Original")
# plt.legend()

#1) Polynomial fit (roughly same as legendre)
# import numpy.polynomial.polynomial as poly
# coefs = poly.polyfit(ps,alldata[2],50)
# fit = poly.polyval(ps,coefs)
# plt.plot(ps,fit,label="poly fit")

#2) Pade approximation of polynomial fit, doesnt work great
# from scipy.interpolate import pade
# p, q = pade(coefs,4)
# for point in ps:
#     plt.plot(point,p(point)/q(point),"o",label="pade",color="C4")

#Chebyshev fit: same as legendre fit
# plt.figure()
# coefs = np.polynomial.chebyshev.chebfit(ps,alldata[2],30)
# chebfit = np.polynomial.chebyshev.chebval(ps,coefs)
# noise = np.random.normal(1,0.01,len(coefs))
# noisecfts = coefs*noise
# chebnoise = np.polynomial.chebyshev.chebval(ps,noisecfts)
# plt.plot(ps,alldata[2],label="Original")
# plt.plot(ps,chebfit,label="cheb fit")
# plt.plot(ps,chebnoise,label="Noisy cfts")
# plt.legend()

#Hermite fit: also roughly same as legendre fit
# coefs = np.polynomial.hermite.hermfit(ps,alldata[2],30)
# chebfit = np.polynomial.hermite.hermval(ps,coefs)
# plt.plot(ps,chebfit,label="hermite fit")

#Laguerre fit : also roughly same as legendre fit
# coefs = np.polynomial.laguerre.lagfit(ps,alldata[2],30)
# chebfit = np.polynomial.laguerre.lagval(ps,coefs)
# plt.plot(ps,chebfit,label="laguerre fit")

    
    
    
    