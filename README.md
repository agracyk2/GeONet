# GeONet
A neural operator for the Wasserstein geodesic

We propose a neural operator for the Wasserstein geodesic.

A neural operator is effectively a means of deducing a nonlinear operator between infinite-dimensional function spaces by constructing a parametric map over a finite-dimensional parameter space.

We learn the Wasserstein geodesic with a neural operator by learning the optimality conditions as a system of PDEs: a continuity equation and a Hamilton-Jacobi equation. We implement a physics-informed loss to solve the coupled PDE system. To account for numerous instances of data, we implement physics-informed DeepONets as an architecture.

GeONet is mesh-invariant, transmuting low-resolution images into high-resolution geodesics. Furthermore, it is instantaneous in the online setting, needing no retraining or recomputation for new input unlike existing geodesic algorithms.
