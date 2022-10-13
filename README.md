# GeONet

We propose a neural operator for the Wasserstein geodesic.

A neural operator is effectively a means of deducing a nonlinear operator between infinite-dimensional function spaces by constructing a parametric map over a finite-dimensional parameter space.

We learn the Wasserstein geodesic with a neural operator by learning the optimality conditions as a system of PDEs: a continuity equation and a Hamilton-Jacobi equation. We implement a physics-informed loss to solve the coupled PDE system. To account for numerous instances of data, we implement physics-informed DeepONets as an architecture.

GeONet is mesh-invariant, transmuting low-resolution images into high-resolution geodesics. Furthermore, it is instantaneous in the online setting, needing no retraining or recomputation for new input unlike existing geodesic algorithms.

![Geodesics Diagram JPG](https://user-images.githubusercontent.com/98125988/190829832-933d8a2e-f247-497b-bb7a-0f8e44b3b814.jpg)


## What does this code do?

This code performs the above description: learning a neural operator for the Wasserstein geodesic. We perform this among two datasets: synthetic data of Gaussian mixtures; and real data from the CIFAR-10 dataset. We implement all of our code using **PyTorch**.


## How do I use this code?

This code is primarily intended for use in Jupyter Notebook formats, where .py files can be run within the primary main notebook file. This code is also intended for all files belonging to the same directory.

Our code is found in the master branch of this repository.
