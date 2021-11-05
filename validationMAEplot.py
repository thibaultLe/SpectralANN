# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 18:58:08 2021

@author: Thibault
"""

#ValidationMAE comparison plot

import numpy as np
import matplotlib.pyplot as plt

#N = 10000:
    
#5*600 -> 0.107, 0.133 avg=0.120
#7*600 -> 0.116, 0.131 avg=0.124

#10/9/21 update: valMAE of 6x800 = 0.745

#100k: 0.26 on epoch 119 of 200 (6x800)
#0.198 epoch 179, 0.190 e190 nothing better at 275 (8x1000)
#0.27 epoch 96 (10x1500) 0.24 epoch 260
# x100k = [4,5,6,7,8,9,10,12]
# fivehundreds = [0.431,0.392,0.374,0.354,0.343,0.338,0.349,0.353]
# eighthundreds= [0.406,0.317,0.262,0.239,0.222,0.232,0.255,0.295]
# thousands =    [0.385,0.302,0.238,0.202,0.197,0.202,0.226,0.254]
# thousand500 =  [0.368,0.303,0.244,0.235,0.235,0.241,0.254,0.271]

x = [2,4,6,8,10]
xmore = [2,4,5,6,7,8,10]
twohundreds = np.array([0.403, 0.174,0.147,0.174,0.237])*6
fourhundreds = np.array([0.341,0.151,0.128,0.130,0.127,0.150,0.194])*6
sixhundreds = np.array([0.314,0.153,0.120,0.128,0.124,0.149,0.180])*6
eighthundreds = np.array([0.305,0.174,0.123,0.117,0.124,0.160,0.187])*6


plt.figure()

plt.plot(x,twohundreds,label="k = 400")
plt.plot(xmore,fourhundreds,label="k = 600")
plt.plot(xmore,eighthundreds,label="k = 800")
plt.plot(xmore,sixhundreds,label="k = 1000")

# plt.plot(x100k,fivehundreds,label="k = 500")
# plt.plot(x100k,eighthundreds,label="k = 750")
# plt.plot(x100k,thousands,label="k = 1000")
# plt.plot(x100k,thousand500,label="k = 1500")

plt.legend()
plt.xlabel("Amount of hidden layers")
plt.ylabel("Validation MAE")

# plt.title("Validation MAE for different amounts of hidden layers and neurons")



#Residues test
# plt.figure()

# plt.plot(-2,1,"^",color="blue",markersize=12)
# plt.plot(-2,2,"^",color="cyan",markersize=12)

# plt.plot(2.5,3.5,"o",color="blue",label="Original poles",markersize=12)
# plt.plot(2,3,"o",color="cyan",label="Reconstructed poles",markersize=12)

# plt.plot(-2,-3,"*",color="blue",markersize=12)
# plt.plot(-2.5,-4,"*",color="cyan",markersize=12)


# plt.plot(-2,1,marker="$1$",color="blue",markersize=12)
# plt.plot(-2,2,marker="$1$",color="cyan",markersize=12)

# plt.plot(2.5,3.5,marker="$2$",color="blue",label="Original poles",markersize=12)
# plt.plot(2,3,marker="$2$",color="cyan",label="Reconstructed poles",markersize=12)

# plt.plot(-2,-3,marker="$3$",color="blue",markersize=12)
# plt.plot(-2.5,-4,marker="$3$",color="cyan",markersize=12)


# plt.xlim(-6,6)
# plt.ylim(-6,6)
# plt.legend()
# plt.grid()
# plt.xlabel("Re(q)")
# plt.ylabel("Im(q)")
# plt.title("Reconstructed complex poles and residues")




