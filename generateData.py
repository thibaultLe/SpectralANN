# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 14:44:28 2021

@author: Thibault
"""
import pandas as pd
import numpy as np

path = "C:/Users/Thibault/Documents/Universiteit/Honours/ \
    Deel 2, interdisciplinair/Code/NN/Datasets/"
    



DT = [[0.15,0.45,0.25],
      [0.2,0.5,0.3],
      [0.25,0.55,0.35]]
DV = [[0.3,0.6,0.4],
      [0.35,0.65,0.45]]
DTest = [[0.4,0.7,0.5],
      [0.45,0.75,0.55]]
      
amountOfTrainingSamples = 10
lengthOfSamples = 14
# rhoT = [[1.8,0.25,0.65],
#         [2,0.3,0.7],
#         [2.2,0.35,0.75]]
rhoT = np.random.rand(amountOfTrainingSamples,lengthOfSamples)
# rhoV = [[2.5,0.4,0.8],
#         [2.8,0.45,0.85]]
rhoV = np.random.rand(amountOfTrainingSamples,lengthOfSamples)
# rhoTest = [[3,0.5,0.9],
#         [3.2,0.55,0.95]]
rhoTest = np.random.rand(amountOfTrainingSamples,lengthOfSamples)

DTdf = pd.DataFrame(DT)
DVdf = pd.DataFrame(DV)
DTestdf = pd.DataFrame(DTest)
rhoTdf = pd.DataFrame(rhoT)
rhoVdf = pd.DataFrame(rhoV)
rhoTestdf = pd.DataFrame(rhoTest)
print("Example of dataframe:\n",DTdf)

DTdf.to_csv(path+"DTraining.csv",header=False,index=False)
DVdf.to_csv(path+"DValidation.csv",header=False,index=False)
DTestdf.to_csv(path+"DTest.csv",header=False,index=False)
rhoTdf.to_csv(path+"rhoTraining.csv",header=False,index=False)
rhoVdf.to_csv(path+"rhoValidation.csv",header=False,index=False)
rhoTestdf.to_csv(path+"rhoTest.csv",header=False,index=False)

print("Data generation succesfull")









