# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 16:14:46 2021

@author: Thibault
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


#Testing spectral density functions

#Parameters:
#Z in [0,10]
#Impact: Large
Z = 1
# Z = 0
#m2 in [0,5]
#Small-medium
# m2 = 2
m2 = 5
#lam2 in [0,5]
#Large
lam2 = 4
#Extra constraint: w^2 + m^2 / lam2 > 1

#fixed gamma
gamma = 13/22

#N1,N2,N3 in [1,5] (int)
N1 = 1
N2 = 1
N3 = 1

#A,B,Cks in [-5,5] 
#Large
Aks = N1 * [0.3]
#Small
Bks = N1 * [3]
#Small
Cks = N1 * [5]

#alfa,beta in [0,5]
#gamma in [-5,5]
#Small
gaml = N2 *  [0.3]
#Small-Medium dependent on other params
alfal = N2 * [3]
#Small
betal = N2 * [5]

#Small
gami = N3 *  [0.3]
#Small
alfai = N3 * [0.2]
#Small
betai = N3 * [5]

#N in [0,3] (int)
N = 1

#a,b,c,d in [-5,5] 
#Large impact (poles + residues)
ajs = N * [1]
bjs = N * [1]
cjs = N * [0.3]
djs = N * [0.5]

sigma = 0

#Convert to different format
abcds = []
for i in range(N):
    abcds.append([ajs[i],bjs[i],cjs[i],djs[i]])
ABCs = []
for i in range(N1):
    ABCs.append([Aks[i],Bks[i],Cks[i]])
gabl = []
for i in range(N2):
    gabl.append([gaml[i],alfal[i],betal[i]])
gabi = []
for i in range(N3):
    gabi.append([gami[i],alfai[i],betai[i]])

#ABCs = [[A1,B1,C1],[A2,B2,C2]...[AN1,BN1,CN1]]
def rho1(w,Z,m2,lam2,N1,ABCs):
    gamma = 13/22
    w2 = w ** 2
    #Calculate the ksum
    ksum = 0
    for k in range(N1):
        ksum += (ABCs[k][0]*w2) / (ABCs[k][1]*w2 + ((ABCs[k][2] - w2) ** 2))
        
        # print((ABCs[k][1] * w2) + (ABCs[k][2] - w2)**2)
            
    #Multiply the ksum with -Z/ln(...)
    np.seterr('raise')
    ln = np.log((w2 + m2) / lam2) ** (1 + gamma)
    mult = -Z / ln
    

    
    return mult * ksum

def custompdf(w,mean,std):
    zlim = 10
    z = (w-mean)/std
    if z < zlim and z > -zlim:
        return np.exp(-(w-mean)**2/(std))
    else:
        return 0


#gabl = [[gam1,alfa1,beta1],[gam2,alfa2,beta2]...[gamN2,alfaN2,betaN2]]
def rho2(w,N2,gabl,N3,gabi):
    w2 = w ** 2

    lsum = 0
    #Sum of normal distributions
    #If w2 > ~5.2: underflow in exp
    for l in range(N2):       
        lsum += gabl[l][0] * custompdf(w2,gabl[l][1],gabl[l][2])

    
    isum = 0
    #Sum of derivatives of normal distributions
    for i in range(N3):           
        isum += gabi[i][0] * w2 * custompdf(w2,gabi[i][1],gabi[i][2])
    return lsum + isum


def rho(w,Z,m2,lam2,N1,ABCs,N2,gabl,N3,gabi):
    return rho1(w,Z,m2,lam2,N1,ABCs) + rho2(w,N2,gabl,N3,gabi)

#Correct form under the integral
def rhoint(w,Z,m2,lam2,N1,ABCs,N2,gabl,N3,gabi,p2):
    return rho(w,Z,m2,lam2,N1,ABCs,N2,gabl,N3,gabi)/(w**2 + p2)

    
def poles(p2,N,abcds):
    jsum = 0
    #Using alternate but equivalent form of pole sum (without i)
    for j in range(N):
        # nom = 2 * ( ajs[j] * (cjs[j] + p2) + bjs[j] * djs[j])
        nom = 2 * ( abcds[j][0] * (abcds[j][2] + p2) + abcds[j][1] * abcds[j][3])
        # denom = (cjs[j] ** 2) + (2 * cjs[j] * p2) + (djs[j] ** 2) + (p2 ** 2)
        denom = (abcds[j][2] ** 2) + (2 * abcds[j][2] * p2) + \
                (abcds[j][3] ** 2) + (p2 ** 2)
        #Extra constraint: not (d == 0 and p2 == -c)
        jsum += nom/denom
    return jsum

#Calculates the propagator out of the given parameters for the spectral
# density function and complex conjugate poles
def calcPropagator(Z,m2,lam2,N,abcds,N1,ABCs,N2,gabl,N3,gabi,sigma,p2):
    return integrate.quad(rhoint,0.01+sigma,10,args=(Z,m2,lam2,N1,ABCs,N2,gabl,N3,gabi,p2))[0] \
        + poles(p2,N,abcds)
        


#To plot spectral density function
ws = np.linspace(0.01,5,200)

# rho1s = []
# for w in ws:
#     rho1s.append(rho1(w))
# plt.figure()
# plt.plot(ws,rho1s)
# plt.xlabel("ω")
# plt.ylabel("ρ1(ω)")
# plt.title("First spectral density function")

# rho2s = []
# for w in ws:
#     rho2s.append(rho2(w))
# plt.figure()
# plt.plot(ws,rho2s)
# plt.xlabel("ω")
# plt.ylabel("ρ2(ω)")
# plt.title("Second spectral density function")

rhos = []
for w in ws:
    rhos.append(rho(w,Z,m2,lam2,N1,ABCs,N2,gabl,N3,gabi))
    
plt.figure()
plt.plot(ws,rhos)
plt.xlabel("ω")
plt.ylabel("ρ(ω)")
plt.title("Spectral density function ρ(ω)")


# ps = np.geomspace(0.001,100,100)
pstart = 0.1
pend = 7.7
nbrPoints = 100
# ps = np.geomspace(pstart,pend,nbrPoints)
ps = np.linspace(pstart,pend,nbrPoints)
# p = 0

# print(integrate.quad(rho,0.01,4))

#Without poles added
# dps = []
# for p in ps:
#     dps.append(integrate.quad(rhoint,0.01,4,p)[0])

#With poles added
#Note : Optimization possible by using LowLevelCallable in integration


dpswithpoles = []
for p in ps:
    #Assume p's are p^2's
    # dpswithpoles.append(integrate.quad(rhoint,0.01,5,p)[0] + poles(p))
    dpswithpoles.append(calcPropagator(Z,m2,lam2,N,abcds,N1,ABCs,N2,gabl,N3,gabi,sigma,p**2))


# rescaling = dpswithpoles[51] * 16.06
# print(ps[11])
# print(ps[12])
# rescaling = dpswithpoles[12] * 1
# for i in range(len(dpswithpoles)):
#     dpswithpoles[i] = dpswithpoles[i]/rescaling

# plt.figure()
# plt.plot(ps,dpswithpoles)
# plt.xlabel("p")
# plt.ylabel("D(p²)")
# plt.title("Propagator D(p²)")

plt.figure()
plt.plot(ps,dpswithpoles)
plt.title("Propagator with poles")
# plt.xscale("log")


# xs = [0.2,0.45,0.7]
# ys = [0.4,0.2,0.9]
# plt.figure()
# plt.scatter(xs,ys)
# plt.xlabel("Re(q)")
# plt.ylabel("Im(q)")
# plt.title("Poles")
# # plt.grid()

# xs = [0.2,0.45,0.7]
# ys = [0.7,0.6,0.4]
# plt.figure()
# plt.scatter(xs,ys)
# plt.xlabel("Re(R)")
# plt.ylabel("Im(R)")
# plt.title("Residues")
# plt.grid()










