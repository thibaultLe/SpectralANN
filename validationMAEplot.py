# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 18:58:08 2021

@author: Thibault
"""

#ValidationMAE comparison plot


import matplotlib.pyplot as plt

#N = 10000:
    
#5*600 -> 0.107, 0.133 avg=0.120
#7*600 -> 0.116, 0.131 avg=0.124

# x = [2,4,6,8,10]
# xmore = [2,4,5,6,7,8,10]
# twohundreds = [0.403, 0.174,0.147,0.174,0.237]
# fourhundreds = [0.341,0.151,0.128,0.130,0.127,0.150,0.194]
# sixhundreds = [0.314,0.153,0.120,0.128,0.124,0.149,0.180]
# eighthundreds = [0.305,0.174,0.123,0.117,0.124,0.160,0.187]


# plt.figure()

# plt.plot(x,twohundreds,label="k = 200")
# plt.plot(xmore,fourhundreds,label="k = 400")
# plt.plot(xmore,sixhundreds,label="k = 600")
# plt.plot(xmore,eighthundreds,label="k = 800")

# plt.legend()
# plt.xlabel("Amount of hidden layers")
# plt.ylabel("Validation MAE")

# plt.title("Validation MAE for different amounts of hidden layers and neurons")



#Residues test
plt.figure()

plt.plot(-2,1,"^",color="blue",markersize=12)
plt.plot(-2,2,"^",color="cyan",markersize=12)

plt.plot(2.5,3.5,"o",color="blue",label="Original poles",markersize=12)
plt.plot(2,3,"o",color="cyan",label="Reconstructed poles",markersize=12)

plt.plot(-2,-3,"*",color="blue",markersize=12)
plt.plot(-2.5,-4,"*",color="cyan",markersize=12)


# plt.plot(-2,1,marker="$1$",color="blue",markersize=12)
# plt.plot(-2,2,marker="$1$",color="cyan",markersize=12)

# plt.plot(2.5,3.5,marker="$2$",color="blue",label="Original poles",markersize=12)
# plt.plot(2,3,marker="$2$",color="cyan",label="Reconstructed poles",markersize=12)

# plt.plot(-2,-3,marker="$3$",color="blue",markersize=12)
# plt.plot(-2.5,-4,marker="$3$",color="cyan",markersize=12)


plt.xlim(-6,6)
plt.ylim(-6,6)
plt.legend()
plt.grid()
plt.xlabel("Re(q)")
plt.ylabel("Im(q)")
plt.title("Reconstructed complex poles and residues")




