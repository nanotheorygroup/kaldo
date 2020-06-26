supplements

# Supplementary Reading
## Boltzmann Transport Equation

In quantum mechanics, the Bose Einstein statistic describes the distribution for atomic vibrations at temperature T by

$$
n_{\mu} = n(\omega_{\mu}) = \frac{1}{e^{\frac{\hbar \omega_{\mu} }{k_B  T} }- 1}
$$

where $k_B$ is the Boltzmann constant and we use $\mu =(k,s)$ to make the notation general where $k$ is a wave vector and s is a position.

We consider a small temperature gradient applied along the $\alpha$-axis of a crystalline material. If the phonons population depends on the position only through the temperature,  $\frac{\partial n_{\mu\alpha}}{\partial x_\alpha} =  \frac{\partial n_{\mu\alpha}}{\partial T}\nabla_\alpha T$, we can Taylor expand it
$$
\tilde n_{\mu\alpha} \simeq n_\mu + \lambda_{\mu\alpha} \frac{\partial n_\mu}{\partial x_\alpha} \simeq  n_\mu + \psi_{\mu\alpha}\nabla_\alpha T
$$
with $\psi_{\mu\alpha}=\lambda_{\mu\alpha} \frac{\partial n_\mu}{\partial T}$, where $\lambda_{\mu\alpha}$ is the phonons mean free path.