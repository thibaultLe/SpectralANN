# SpectralANN
The trained PyTorch neural network can be found under 'savedNNmodel.pth', which has a configuration of 6 hidden layers and 600 neurons per layer.

Our generated datasets of training, validation and testing can be found under Datasets


# Workflow to generate new data and train the network:
1) Run generatePropagators.py

   The datasets of training, validation and testing can be found under Datasets
3) Run propagatorNoise.py

   This converts the propagator data to the correct input format
5) Run train_ACANN.py

   The trained neural network is saved in 'savedNNmodel.pth'
5) Run test_ACANN.py

   The resulting neural network can be tested

Note: Update the "path" variable in most files to point to your current working directory
