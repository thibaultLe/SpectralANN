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
from scipy import interpolate

#Load input parameters from inputParameters.py
sizeOfTraining = config.trainingPoints 
pstart = config.pstart
pend = config.pend
nbrPoints = config.nbrPoints

path = "C:/Users/Thibault/Documents/Universiteit/\
Honours/Deel 2, interdisciplinair/Code/NN/Datasets/"
train_data = Database(csv_target= path + "rhoTraining.csv",\
                      csv_input= path + "DTraining.csv",nb_data=sizeOfTraining).get_loader()

trainloader = DataLoader(train_data,batch_size=sizeOfTraining)
print("Input data loaded")

#get propagators from DTraining.csv
alldatatensors = list(trainloader)
alldata = alldatatensors[0][0].to("cpu").numpy()


print(len(alldata),"training points")
# print(alldata[i])

#Fit with legendre, build dataset of coefficients
#Run PCA
#Get Principal Components

# ps = np.geomspace(pstart,pend,nbrPoints)
#When using logspace, the fit is atrocious -> use linspace
ps = np.linspace(pstart,pend,nbrPoints)

# i = 23

# plt.figure()
# plt.title("Fitting of propagator data")
# # plt.xscale("log")
# plt.plot(ps,alldata[i],"o",label="Original")

#maxdegree of 35 = 36 values
maxdegree = 35

# #Legendre fit
# # maxdegree of 35 has the smallest MAE of almost all propagators (see below)
# coeffs = np.polynomial.legendre.legfit(ps,alldata[i],maxdegree)
# plt.plot(ps,np.polynomial.legendre.legval(ps,coeffs),label="Legendre fitted")
# print("Coefficients of legendre fit",coeffs)

# plt.legend()

#Legendre fit to all propagators in training set, keeps coefficients
coefficientsList = []
for i in range(len(alldata)):
    coefficientsList.append(np.polynomial.legendre.legfit(ps,alldata[i],maxdegree))

# print(coefficientsList)


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

x = coefficientsList

#Normalise the attributes
scaler=StandardScaler()#instantiate
scaler.fit(x) # compute the mean and standard which will be used in the next command
X_scaled=scaler.transform(x)
# print("Means:")
# compList = []
# for axis in scaler.mean_:
#     compList.append(axis)
# print(compList)
# print(scaler.mean_)
# print("Variances:")
# compList = []
# for axis in scaler.var_:
#     compList.append(axis)
# print(compList)
# print(scaler.var_)

#8 components is not enough: e.g. i = 10 diverges with only 8 components
pca=PCA(n_components=10) 
pca.fit(X_scaled) 
X_pca=pca.transform(X_scaled) 


#Restore after pca = inverse scaling after inverse PCA 
x_restore = scaler.inverse_transform(pca.inverse_transform(X_pca))

i = 1
#TODO: plot MAE vs nbr of PCA components over a bunch of propagators

# plt.figure()
# plt.plot(ps,np.polynomial.legendre.legval(ps,x[i]),label="Original Legendre fitted")
# plt.plot(ps,np.polynomial.legendre.legval(ps,x_restore[i]),label="PCA reconstruction")
# plt.title("PCA reconstruction")
# plt.legend()


#Noise removal test:
    
plt.figure()
# plt.plot(ps,alldata[i],label="Original propagator")
# plt.plot(ps,np.polynomial.legendre.legval(ps,x[i]),label="Propagator Legendre fit")
plt.xlim(ps[0]-0.5,ps[-1]+0.5)
plt.ylim(min(alldata[i])-1,max(alldata[i])+10)

np.random.seed(2)
noise = np.random.normal(1,0.01,len(alldata[0]))
#Possible alternative: adding instead of multiplying
propWithNoise = alldata[i] * noise

pWNlegfit = np.polynomial.legendre.legfit(ps,propWithNoise,maxdegree)
plt.plot(ps,np.polynomial.legendre.legval(ps,pWNlegfit),label="Legendre fit to noise")

pWNlegfitreshaped = pWNlegfit.reshape(1,-1)
noisePCAd = pca.transform(scaler.transform(pWNlegfitreshaped))
# noisePCAd = pca.transform(scaler.transform(pWNlegfit))
noisePCAreconstructed = scaler.inverse_transform(pca.inverse_transform(noisePCAd))
# plt.plot(ps,np.polynomial.legendre.legval(ps,noisePCAreconstructed[0]),label="PCA reconstruction")


print("Original:     ",pWNlegfit)
print("Reconstructed:",noisePCAreconstructed[0])

#Adding noise to pWNlegfit coefficients:
legnoise = np.random.normal(1,0.001,len(pWNlegfit))
legnoiseCfts = pWNlegfit * legnoise
plt.plot(ps,np.polynomial.legendre.legval(ps,legnoiseCfts),label="Noisy legendre fit")
print("Noisy legendre:",legnoiseCfts)

# plt.plot(ps,propWithNoise,label="Propagator with noise")
plt.title("Denoising with PCA")
plt.legend()





# plt.figure()
# Xax=X_pca[:,0]
# Yax=X_pca[:,1]
# plt.scatter(Xax,Yax,s=12,alpha=0.5)
# plt.title("Instance space of propagators")
# plt.xlabel("First principal component")
# plt.ylabel("Second principal component")


print("Explained variance of the principal components", \
      str(format(sum(pca.explained_variance_ratio_),".5f")))
print(pca.explained_variance_ratio_)
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
# y = rational(ps, [-0.2, 0.3, 0.5], [-1.0, 2.0])
# ynoise = y * (1.0 + np.random.normal(scale=0.1, size=x.shape))
# popt,pcov = curve_fit(rational3_3, ps,alldata[2])
# print(popt)

# plt.plot(x, y, label='original')
# plt.plot(x, ynoise, '.', label='data')
# plt.plot(ps, rational3_3(ps, *popt), label='Rational 3/3 fit')

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
# coefs = np.polynomial.chebyshev.chebfit(ps,alldata[2],30)
# chebfit = np.polynomial.chebyshev.chebval(ps,coefs)
# plt.plot(ps,chebfit,label="cheb fit")

#Hermite fit: also roughly same as legendre fit
# coefs = np.polynomial.hermite.hermfit(ps,alldata[2],30)
# chebfit = np.polynomial.hermite.hermval(ps,coefs)
# plt.plot(ps,chebfit,label="hermite fit")

#Laguerre fit : also roughly same as legendre fit
# coefs = np.polynomial.laguerre.lagfit(ps,alldata[2],30)
# chebfit = np.polynomial.laguerre.lagval(ps,coefs)
# plt.plot(ps,chebfit,label="laguerre fit")

    
    
    
    