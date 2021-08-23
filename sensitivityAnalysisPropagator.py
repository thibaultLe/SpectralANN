# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 15:23:13 2021

@author: Thibault
"""
# from SALib.sample import saltelli
# from SALib.analyze import sobol
from SALib.sample.morris import sample
from SALib.analyze.morris import analyze
from SALib.test_functions import Ishigami
import numpy as np
import matplotlib.pyplot as plt

import generatePropagators

# Define the model inputs
#Test with all Ns = 1
problem = {
    'groups' : ['rho1Zs','rho1Zs','rho1Zs',
                'rho1ABCs','rho1ABCs','rho1ABCs',
                'rho2gabls','rho2gabls','rho2gabls',
                'rho2gabis','rho2gabis','rho2gabis',
                'abcds','abcds','abcds','abcds'],
    #               3    +   3       + 3       +      3     +  4
    'num_vars': 16,
    'names': ['Z', 'm2', 'lam2','A','B','C','GL','AL','BL','GI','AI','BI',\
              'a','b','c','d'],
    'bounds': [#Zs:
               [1,10],
               [2,5],
               [1,4], #m2 > lam2, lam2 > 0
               #ABCs:
               [-3,3],
               [-3,3],
               [-3,3],
               #gabls:
               [-3,3],
               [1,5], 
               [1,5], #betas > 0
               #gabis:
               [-3,3],
               [1,5], 
               [1,5],#betas > 0
               #abcds:
               [-3,3],
               [-1,5],
               [-3,3],
               [-1,5]]
}

# Generate samples
# param_values = saltelli.sample(problem, 4)
param_values = sample(problem, 4)
print(len(param_values),"parameter combinations")
# print(param_values)

#Remove bad parameters: X raises runtimeError 
#                         Incorrect number of samples in model output file.
# def checkSingularity(w,B,C):
#     w2 = w**2
#     if B*w2 + C**2 -2*C*w2 + w2**2 <= 10**(-7):
#         return True
#     else:
#         return False
    
# ws = np.linspace(0.01,5,200)
# param_values_filtered = []
# for params in param_values:
#     singularityFound = False
#     for w in ws:
#         if checkSingularity(w,params[4],params[5]):
#             singularityFound = True
#     if not singularityFound:
#         param_values_filtered.append(params)

# Run model (example)
# evaluate
# pstart = 0.1
# pend = 10
# nbrPoints = 16
# ps = np.geomspace(pstart,pend,nbrPoints)
# ps = np.linspace(pstart,pend,nbrPoints)

listOfmuStars = []

maxP = 9

for evalAtP in range(1,maxP):
    def evalProp(Z,m2,lam2,A,B,C,GL,AL,BL,GI,AI,BI,a,b,c,d):
        N,N1,N2,N3 = (1,1,1,1)
        ABCs = ((A,B,C),)
        gabls = ((GL,AL,BL),)
        gabis = ((GI,AI,BI),)
        abcds = ((a,b,c,d),)
        # dps = []
        # for p in ps:
        #Evaluated at p = 1
        # evalAtP > 7.5 -> multiply error
        # evalAtP = 1
        return generatePropagators.calcPropagator(Z,m2,lam2,N,abcds,N1,ABCs,\
                                                  N2,gabls,N3,gabis,0,evalAtP)
        # return dps
            
    y = np.array([evalProp(*params) for params in param_values])
    # print(y.shape)
    # print(y[0])
    
    # # analyse
    # sobol_indices = [sobol.analyze(problem, Y) for Y in y.T]
    # sobol_indices = [analyze(problem,param_values, Y) for Y in y.T]
    
    # Y = Ishigami.evaluate(param_values)
    # print(Y)
    
    # # Perform analysis
    # Si = sobol.analyze(problem, Y, print_to_console=True)
    # print(param_values.shape)
    # print(y.shape)
    # print(Y.shape)
    Si = analyze(problem,param_values, y, print_to_console=False,seed=5)
    
    # print(Si.get("mu_star"))
    listOfmuStars.append(list(Si.get("mu_star")))

# print(listOfmuStars)

# ps = list(range(1,maxP))
ps = np.linspace(0,10,maxP-1)
mZs = []
mABCs = []
mgabls = []
mgabis = []
mabcds = []
for i in range(len(ps)):
    mZs.append(listOfmuStars[i][0])
    mABCs.append(listOfmuStars[i][1])
    mgabls.append(listOfmuStars[i][2])
    mgabis.append(listOfmuStars[i][3])
    mabcds.append(listOfmuStars[i][4])

# print(ps)
plt.figure()
# for i in range(len(ps)):
plt.plot(ps,mZs,color="C0",label="Z,m²,Lambda²")
plt.plot(ps,mABCs,color="C1",label="A,B,C")
plt.plot(ps,mgabls,color="C2",label="Gamma,alpha,beta")
plt.plot(ps,mgabis,color="C3",label="Gamma',alpha',beta'")
plt.plot(ps,mabcds,color="C4",label="Poles")
plt.legend()
plt.title("Sensitivity analysis of different parameters")
plt.xlabel("p²")
plt.ylabel("Mean elementary effect")

"""
TODO:
    for different evalAtP's, plot the mu_star of different groups
    
"""

# # Print the first-order sensitivity indices
# print(Si['S1'])
