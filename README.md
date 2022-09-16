# GeONet

We propose a neural operator for the Wasserstein geodesic.

A neural operator is effectively a means of deducing a nonlinear operator between infinite-dimensional function spaces by constructing a parametric map over a finite-dimensional parameter space.

We learn the Wasserstein geodesic with a neural operator by learning the optimality conditions as a system of PDEs: a continuity equation and a Hamilton-Jacobi equation. We implement a physics-informed loss to solve the coupled PDE system. To account for numerous instances of data, we implement physics-informed DeepONets as an architecture.

GeONet is mesh-invariant, transmuting low-resolution images into high-resolution geodesics. Furthermore, it is instantaneous in the online setting, needing no retraining or recomputation for new input unlike existing geodesic algorithms.

![Geodesics Diagram JPG](https://user-images.githubusercontent.com/98125988/190828970-aa3f231f-51b2-4bde-ba0c-c88949e1f54f.jpg)

## What does this code do?

This code performs the above description: learning a neural operator for the Wasserstein geodesic. We perform this among two datasets: synthetic data of Gaussian mixtures; and real data from the CIFAR-10 dataset. We implement all of our code using **PyTorch**.

![Gaussians 6 jpg](https://user-images.githubusercontent.com/98125988/190829170-5bd7a566-1e6e-476d-bcae-72fbde16caa6.jpg)
<img src=["https://user-images.githubusercontent.com/16319829/81180309-2b51f000-8fee-11ea-8a78-ddfe8c3412a7.png](https://user-images.githubusercontent.com/98125988/190829170-5bd7a566-1e6e-476d-bcae-72fbde16caa6.jpg)" width=50% height=50%>
