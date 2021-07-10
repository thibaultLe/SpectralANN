from ACANN import ACANN
from Database import Database
from torch.nn.modules.loss import KLDivLoss,L1Loss, SmoothL1Loss
from torch.optim import Adam,Rprop,Adamax, RMSprop,SGD,LBFGS
from torch.utils.data import DataLoader
import torch
import inputParameters as config


#Load input parameters from inputParameters.py
nbrOfNormalDists = config.nbrOfNormalDists
nbrOfPoles = config.nbrOfPoles
sizeOfTraining = config.trainingPoints 
sizeOfValidation = config.validationPoints
outputSize = (3 * nbrOfNormalDists) + (4 * nbrOfPoles)
print("Input parameters loaded")

print("Starting ACANN")
# Create the network
model = ACANN(200,outputSize,[200,200,200],drop_p=0.09).double()

print("Model created")
# Import the data
path = "C:/Users/Thibault/Documents/Universiteit/Honours/Deel 2, interdisciplinair/Code/NN/Datasets/"
train_data = Database(csv_target= path + "rhoTraining.csv",csv_input= path + "DTraining.csv",nb_data=sizeOfTraining).get_loader()
validation_data=Database(csv_target= path + "rhoValidation.csv",csv_input= path + "DValidation.csv",nb_data=sizeOfValidation).get_loader()

trainloader = DataLoader(train_data,batch_size=10,shuffle=True)
validationloader = DataLoader(validation_data,batch_size=10)
# print("Training:",list(trainloader))
# print("Validation:",list(validationloader))
print("Data Loaded")


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
        
        #TODO: write validation loss function
        
        # print("rho_val",A_val)
        # print("prediction:",prediction)
        score=val_error(prediction,A_val)
    #Turn training mode back on
    nn_model.train()
    return score.item()


#Define the loss
error = L1Loss()
#Define the optimizer
optimizer = Adam(model.parameters())
#RMSPRO 10 - 2e-3
#ADAM 10 - 1.2e-3

# Training parameters
epochs = 100
step=-1
print_every = 30
print("##############################")
print("##############################")
print("Starting the training")

# Training
for e in range(epochs):
    #Turn training mode on
    model.train()
    #  Load a minibatch
    for D,A in trainloader:
        # print("D:",D)
        # print("A:",A)
        step+=1
        # restart the optimizer
        optimizer.zero_grad()
        # compute the loss
        prediction = model.forward(D)
        loss = error(prediction,A)
        # Compute the gradient and optimize
        loss.backward()
        optimizer.step()

        # Write the result
        if step % print_every == 0:
            step=0
            print("Epoch {}/{} : ".format(e+1,epochs),
                  "Training MAE = {} -".format(loss.item()),
                  "Validation MAE = {}".format(validation_score(model)))
torch.save(model.state_dict(),'savedNNmodel.pth')
print("Saved model")

