# GeONet

We propose a neural operator for the Wasserstein geodesic.

A neural operator is effectively a means of deducing a nonlinear operator between infinite-dimensional function spaces by constructing a parametric map over a finite-dimensional parameter space.

We learn the Wasserstein geodesic with a neural operator by learning the optimality conditions as a system of PDEs: a continuity equation and a Hamilton-Jacobi equation. We implement a physics-informed loss to solve the coupled PDE system. To account for numerous instances of data, we implement physics-informed DeepONets as an architecture.

GeONet is mesh-invariant, transmuting low-resolution images into high-resolution geodesics. Furthermore, it is instantaneous in the online setting, needing no retraining or recomputation for new input unlike existing geodesic algorithms.

![Geodesics Diagram JPG](https://user-images.githubusercontent.com/98125988/190829832-933d8a2e-f247-497b-bb7a-0f8e44b3b814.jpg)


## What does this code do?

This code performs the above description: learning a neural operator for the Wasserstein geodesic. We perform this among two datasets: synthetic data of Gaussian mixtures; and real data from the CIFAR-10 dataset. We implement all of our code using **PyTorch**.


## How do I use this code?

This code is primarily intended for use in Jupyter Notebook formats, where .py files can be run within the primary main notebook file. This code is also intended for all files belonging to the same directory. Our code is found in the **master branch** of this repository.

**Gaussian mixtures:** The primary file to run is GeONet_main_Gaussian_mixtures, while the other files are supplementary to run in conjunction with this file. Please ensure all files are in the same directory.

To run GeONet from pre-trained networks, one only needs the main file, the GeONet_models file, and the six .pth files containing the neural network parameters. One can read the comments in the main code to determine which code chunks to run to use the pre-trained model.

To train GeONet, one needs the main file, the 'GeONet_models.py' file, Physics_informed_loss.py, Gaussian_mixture_data_creation.py, and Training_algorithm.py. One can modify the experiment parameters, such as means and variances, if one wishes to test GeONet on new sets of training data.
