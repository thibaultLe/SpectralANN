from ACANN import ACANN
from Database import Database
from torch.nn.modules.loss import KLDivLoss,L1Loss,MSELoss
from torch.optim import Adam,Rprop,Adamax, RMSprop,SGD,LBFGS,AdamW
from torch.utils.data import DataLoader
import torch
import inputParameters as config
import matplotlib.pyplot as plt


#Load input parameters from inputParameters.py
inputSize = config.nbrPoints
nbrWs = config.nbrWs
nbrOfPoles = config.nbrOfPoles
sizeOfTraining = config.trainingPoints 
sizeOfValidation = config.validationPoints
outputSize = nbrWs + (4 * nbrOfPoles) + 1
print("outputsize:",outputSize)
print("Input parameters loaded")


print("Starting ACANN")
# Create the network

MAEsplot = []
# layers = [2,4,6,8]
# neuronsPerLayer = [200,400,600,800]
layers = [6]
neuronsPerLayer = [800]

for layer in layers:
    for neuronNbr in neuronsPerLayer:
        print(layer,"layers",neuronNbr,"neurons per layer")
        model = ACANN(inputSize,outputSize,layer*[neuronNbr],drop_p=0.1).double()
        
        epochs = 48
        batch_size_train = 100
        
        #100k: 0.26 on epoch 119 of 200 (6x800)
        #0.198 epoch 179, 0.190 e190 nothing better at 275 (8x1000)
        #0.27 epoch 96 (10x1500) 0.24 epoch 260
        
        
        # print("Model created")
        # Import the data
        path = "C:/Users/Thibault/Documents/Universiteit/Honours/Deel 2, interdisciplinair/Code/NN/Datasets/"
        train_data = Database(csv_target= path + "rhoTraining.csv",csv_input= path + "DTraining.csv",nb_data=epochs*batch_size_train).get_loader()
        validation_data=Database(csv_target= path + "rhoValidation.csv",csv_input= path + "DValidation.csv",nb_data=sizeOfValidation).get_loader()
        
        trainloader = DataLoader(train_data,batch_size=batch_size_train,shuffle=True)
        validationloader = DataLoader(validation_data,batch_size=100,shuffle=True)
        # print("Training:",list(trainloader))
        # print("Validation:",list(validationloader))
        # print("Data Loaded")
        
        
        
        # Define a function for computing the validation score
        def validation_score(nn_model):
            #Turn evaluation mode on (turns off dropout and batch normalization):
            nn_model.eval()
            #Loss function
            val_error=MSELoss()
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
        error = MSELoss()
        #Define the optimizer
        optimizer = Adam(model.parameters())
        
        # Training parameters
        step=-1
        print_every = epochs
        # print("##############################")
        # print("##############################")
        # print("Starting the training")
        
        # Training
        best_valscore = 10000
        MAEs = []
        stop = False
        for e in range(epochs):
            if not stop:
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
                        # if validation_score(model) > 10:
                        #     stop = True
                            
                
                MAEs.append(validation_score(model))
            
            
        print("Saved model with validation MAE of", best_valscore)
            
        plt.figure()
        plt.plot(list(range(1,len(MAEs)+1)),MAEs)
        plt.title("Validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("MSE")
        
        MAEsplot.append(min(MAEs))
        
print(MAEsplot)
