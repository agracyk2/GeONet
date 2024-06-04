We present GeONet, a neural operator for the Wasserstein geodesic. It is known the quadratic Wasserstein distance minimization optimization problem can be expressed as minimal kinetic energy flow problem. Using the method of Lagrange multipliers, we obtain the Karush-Kuhn-Tucker (KKT) optimality conditions, which are joint continuity and Hamilton-Jacobi (HJ) equations. GeONet behaves as a physics-informed enhanced deep operator network by solving the continuity and HJ partial differential equations simultaneously, the continuity equation solution yielding the geodesic.

<p align="center">
<img src="https://github.com/agracyk2/GeONet/assets/98125988/1ec8613c-cbbf-4649-b688-f38802940bf1" width = 700>
</p>


In the offline stage, GeONet trains using such a physics-informed approach. GeONet is instantaneous in the online setting, and can be deployed for real-time predictions of Wasserstein geodesics with a significant reduction in computation cost with considerable accuracy. GeONet is trained using a uniform collocation procedure, which allows output mesh-invariance, meaning GeONet is suitable for zero-shot super resolution, and low-resolution geodesics can be adapted to high-resolution with no extra computational cost; traditional OT solvers cannot do this.
