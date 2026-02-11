### Forewords 
This code is a proof of concept designed to solve a minimal-time transfer problem on the special unitary group $SU(d)$ applied to the control of driftless closed quantum systems, as detailled in [Janković et al. (2025)](https://arxiv.org/abs/2505.21203). 

This code is written in `Julia` version `1.11.6`. The solver is based on a natural gradient descent in the **optimize-then-discretize** paradigm, also known as **indirect** method. We were able to optimize gates for qudits of dimension less than or equal sixteen.

### Getting started 
Open the terminal and run
```
git clone https://github.com/killianlutz/Magicarp.git
```

Specify the desired hyperparameters in `./scripts/parameters.jl` and then run the file `./scripts/natgrad.jl` for various target gates.

### Mathematical details
We are given $m$ orthonormal control Hamiltonians $H_1, \ldots, H_m$ generating the Lie algebra of $SU(d)$ and a target special unitary gate $U_{\mathrm{target}}$. We then optimize for the self-adjoint traceless matrix $g$ minimizing

$$T = \sqrt{\sum_{j=1}^m \mathrm{Tr}\left(H_{j}g\right)^2}$$

subject to the constraint that the solution $U_g(\cdot)$ of

$$\dot{U} = -\mathrm{i}\sum_{j=1}^m \mathrm{Re} \mathrm{Tr}\left(U^\dagger H_{j}U g\right)H_j U, \quad 0 < t < 1$$

starting at $U(0) = I$ reaches the target at time $t = 1$, that is $U_g(1) = U_{\mathrm{target}}$.

**Caveats**: at the moment, the solver only solves for a control $g$ such that $U_g(1) = U_{\mathrm{target}}$.

### Troubleshooting
Feel free to reach out to me: [Killian Lutz](https://killianlutz.github.io/).
