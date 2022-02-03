# SpectralANN
Note: Update the "path" variable in most files to point to your current working directory

# Workflow:

1) Run generatePropagators.py

   The datasets of training, validation and testing can be found under Datasets
3) Run propagatorNoise.py

   This converts the propagator data to the correct input format
5) Run train_ACANN.py

   The trained neural network is saved in 'savedNNmodel.pth'
5) Run test_ACANN.py

   The resulting neural network can be tested
