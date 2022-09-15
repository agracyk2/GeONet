# GeONet
We propose a neural operator for the Wasserstein geodesic.

A neural operator is effectively a means of deducing a nonlinear operator between infinite-dimensional functional spaces by constructing a parametric map over a finite-dimensional parameter space.

We learn the Wasserstein geodesic with a neural operator by learning the optimality conditions as a system of PDEs: a continuity equation and a Hamilton-Jacobi equation. We implement a physics-informed loss to solve the coupled PDE system. To account for numerous instances of data, we implement physics-informed DeepONets.

GeONet is mesh-invariant, transmuting low-resolution images into high-resolution geodesics. Furthermore, it is instantaneous in the online setting, unlike traditional geodesic algorithms.

![Geodesics Diagram JPG](https://user-images.githubusercontent.com/98125988/190309850-a7b9425c-86f9-4952-a7c4-d30f77181318.jpg)


## What does this code do?

This code performs the above: learning a neural operator for the Wasserstein geodesic. We perform this among two datasets: synthetic data of Gaussian mixtures; and real data of the CIFAR-10 dataset. We implement all of our code using **PyTorch**.

In the Gaussian mixture code, you will find: construction of DeepONets; physics-informed loss for DeepONets for continuity and HJ equations; a generation of Gaussian mixture data; our training algorithm to train GeONet.


## Dependencies

To compare the data to existing methods, we use POT, the optimal transport library. It may be necessary to run the following, or install by other means:

```
!pip install pot
```
