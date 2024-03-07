# RAD (Relativistic Adaptive Gradient Descent)
## Description
Training deep reinforcement learning (RL) agents necessitates overcoming the highly unstable nonconvex stochastic optimization inherent in the trial-and-error mechanism. To tackle this challenge, we propose a physics-inspired optimization algorithm called relativistic adaptive gradient descent (RAD), which enhances long-term training stability. By conceptualizing neural network (NN) training as the evolution of a conformal Hamiltonian system, we present a universal framework for transferring long-term stability from conformal symplectic integrators to iterative NN updating rules, where the choice of kinetic energy governs the dynamical properties of resulting optimization algorithms. By utilizing relativistic kinetic energy, RAD incorporates principles from special relativity and limits parameter updates below a finite speed, effectively mitigating abnormal gradient influences. Additionally, RAD models a multi-particle system where each trainable parameter acts as an independent particle with an individual adaptive learning rate, facilitating rapid convergence. We establish RAD's convergence under general nonconvex stochastic settings, enabling its broad applicability beyond RL. Notably, RAD generalizes the well-known adaptive moment estimation algorithm while providing insights into the intrinsic dynamics of other adaptive gradient algorithms. Experimental results on RL benchmarks demonstrate that RAD achieves state-of-the-art performance compared to mainstream optimizers, highlighting its potential for stabilizing RL training.

## Requirement
1. Linux is preferred.
2. Python 3.6 or greater. We recommend using Python 3.7.
3. Pytorch installed.

## Quick Start
1. All optimizers have been implemented in the Python file "optimizers.py", including RAD, ADAM, SGD (equaling HB when momentum is not 0), DLPF, NAG, RGD.
2. Include the following code in your training script to utilize the corresponding optimizer.
```bash
from optimizers import RAD, Adam, SGD, DLPF, RGD, NAG
rad_optim = RAD(net.parameters(), lr=0.001, betas=(0.9, 0.999), delta=1, order=1, eps=1, eps_annealing=1e6)
adam_optim = Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-16)
hb_optim = SGD(net.parameters(), lr=0.001, momentum=0.9)
dlpf_optim = DLPF(net.parameters(), lr=0.001, momentum=0.9)
rgd_optim = RGD(net.parameters(), lr=0.001, momentum=0.9, delta=1, order=1)
nag_optim = NAG(net.parameters(), lr=0.001, momentum=0.9)
sgd_optim = SGD(actor.parameters(), lr=0.001, momentum=0)
```

## Supplementary materials
Any user can find the supplementary in the "Supplementary materials" folder.