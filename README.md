# SpectralANN: Neural network approach to reconstructing spectral functions and complex poles of confined particles
The trained PyTorch neural network can be found under 'savedNNmodel.pth', which has a configuration of 6 hidden layers and 600 neurons per layer.
Our generated datasets of training, validation and testing can be found under Datasets

Note: Make sure the "path" variable in each file points to your current working directory

# Testing the trained network:
1) Run test_ACANN.py for the available test data
2) Run MonteCarloTest.py for genuine lattice data (this data is too large for GitHub, but we can send it upon request)

# Workflow to generate new data and train a neural network:
1) Run generatePropagators.py

   The datasets of training, validation and testing can be found under Datasets
3) Run propagatorNoise.py

   This converts the propagator data to the correct input format (adds artificial noise to improve robustness)
5) Run train_ACANN.py

   The trained neural network is saved in 'savedNNmodel.pth'
5) Run test_ACANN.py

   The resulting neural network can be tested

If you use this dataset or code, please cite us (https://arxiv.org/abs/2203.03293):

    @Article{10.21468/SciPostPhys.13.4.097,
	title={{Neural network approach to reconstructing spectral functions and complex  poles of confined particles}},
	author={Thibault Lechien and David Dudal},
	journal={SciPost Phys.},
	volume={13},
	pages={097},
	year={2022},
	publisher={SciPost},
	doi={10.21468/SciPostPhys.13.4.097},
	url={https://scipost.org/10.21468/SciPostPhys.13.4.097},
}
