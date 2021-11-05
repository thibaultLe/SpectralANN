from ACANN import ACANN
from Database import Database
from torch.nn.modules.loss import KLDivLoss,L1Loss, SmoothL1Loss
from torch.optim import Adam,Rprop,Adamax, RMSprop,SGD,LBFGS,AdamW
from torch.utils.data import DataLoader
import torch
import numpy as np
import inputParameters as config
import math
import matplotlib.pyplot as plt


#Load input parameters from inputParameters.py
inputSize = config.nbrOfPCAcomponents
nbrWs = config.nbrWs
nbrOfPoles = config.nbrOfPoles
sizeOfTraining = config.trainingPoints 
sizeOfValidation = config.validationPoints
outputSize = nbrWs + (4 * nbrOfPoles) + 1
print("outputsize:",outputSize)
print("Input parameters loaded")

print("Starting ACANN")
# Create the network
# model = ACANN(inputSize,outputSize,[62,112,212],drop_p=0.09).double()
model = ACANN(inputSize,outputSize,8*[1000],drop_p=0.05).double()

epochs = 216
batch_size_train = 150

#100k: 0.26 on epoch 119 of 200 (6x800)
#0.198 epoch 179, 0.190 e190 nothing better at 275 (8x1000)
#0.27 epoch 96 (10x1500) 0.24 epoch 260




print("Model created")
# Import the data
path = "C:/Users/Thibault/Documents/Universiteit/Honours/Deel 2, interdisciplinair/Code/NN/Datasets/"
train_data = Database(csv_target= path + "rhoTraining.csv",csv_input= path + "DTraining.csv",nb_data=sizeOfTraining).get_loader()
validation_data=Database(csv_target= path + "rhoValidation.csv",csv_input= path + "DValidation.csv",nb_data=sizeOfValidation).get_loader()

trainloader = DataLoader(train_data,batch_size=batch_size_train,shuffle=True)
validationloader = DataLoader(validation_data,batch_size=100,shuffle=True)
# print("Training:",list(trainloader))
# print("Validation:",list(validationloader))
print("Data Loaded")

torch.pi = torch.tensor(math.pi)
normpdf = torch.sqrt(torch.pi * 2)
zlim = 10
ws = np.linspace(0.01,10,nbrWs)

#Calculates the pdf of a normal distribution
def custompdf(w,mean,std):
    z = (w-mean)/std
    if z < zlim and z > -zlim:
        #torch.exp(input), torch.sqrt(), 
        #torch.pi = torch.acos(torch.zeros(1)).item() * 2
        return torch.exp(-(w-mean)**2/(2.0*(std**2))) / normpdf
        # return np.exp(-(w-mean)**2/(2.0*(std**2))) / (_norm_pdf_C*std)
    else:
        return 0

def eval_output(predicData):
    nbrOfNormalDists = 66
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
    


# Define a function for computing the validation score
def validation_score(nn_model):
    #Turn evaluation mode on (turns off dropout and batch normalization):
    nn_model.eval()
    #Loss function
    val_error=L1Loss()
    #Turn off gradient computation
    with torch.no_grad():
        G_val,A_val=next(iter(validationloader))
        prediction=nn_model.forward(G_val)
        #Prediction and A_val are tensors with lists of size batch_size
        #The lists contain (outputSize) data values        
        score=val_error(prediction,A_val)
    #Turn training mode back on
    nn_model.train()
    return score.item()




#Define the loss
error = L1Loss()
#Define the optimizer
optimizer = Adam(model.parameters())

# Training parameters
step=-1
print_every = batch_size_train
print("##############################")
print("##############################")
print("Starting the training")

# Training
best_valscore = 10000
MAEs = []
for e in range(epochs):
    #Turn training mode on
    model.train()
    #  Load a minibatch
    for D,rho in trainloader:
        # print("D:",D)
        # print("A:",A)
        step+=1
        # restart the optimizer
        optimizer.zero_grad()
        # compute the loss
        prediction = model.forward(D)
        # print(prediction)
        # loss = error(eval_output(prediction),rho)
        loss = error(prediction,rho)
        # Compute the gradient and optimize
        loss.backward()
        optimizer.step()

        # Write the result
        if step % print_every == 0:
            step=0
            print("Epoch {}/{} : ".format(e+1,epochs),
                  "Training MAE = {} -".format(loss.item()),
                  "Validation MAE = {}".format(validation_score(model)))
            
            #Early stopping: only save the best validation MAE
            if validation_score(model) < best_valscore:
                best_valscore = validation_score(model)
                torch.save(model.state_dict(),'savedNNmodel.pth')
        
    MAEs.append(validation_score(model))
    
    
print("Saved model with validation MAE of", best_valscore)
    

plt.figure()
plt.plot(list(range(1,len(MAEs)+1)),MAEs)
plt.title("Validation loss")
plt.xlabel("Epochs")
plt.ylabel("MAE")

