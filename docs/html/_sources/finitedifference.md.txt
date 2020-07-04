finitediff_v2

# Finite Difference Theory

## Dynamical Matrices

In solids, electronic and vibrational dynamics often involve different time scales, and can thus be decoupled using the Born Oppenheimer approximation. Under this assumption, the potential $\phi$ of a system made of $N_{atoms}$ atoms, where $i$ and $\alpha$ refer to the atomic and Cartesian indices respectively. Near thermal equilibrium, the potential energy can be Taylor expanded in the atomic displacements, $\mathbf{u}=\mathbf x-\mathbf{x}_{\rm equilibrium}$, 
$$
    \phi^{\prime\prime}_{i\alpha i'\alpha '}=\frac{\partial^{2} \phi}{\partial u_{i\alpha } \partial u_{i'\alpha '}   },\qquad
    \phi^{\prime\prime\prime}_{i\alpha i'\alpha 'i''\alpha ''}=\frac{\partial^{3} \phi}{\partial u_{i\alpha } \partial u_{i'\alpha '} \partial u_{i''\alpha ''}}
$$

are the second and third order interatomic force constants (IFC). The term $\phi_0$ can be discarded, and the forces $F = - \phi^{\prime}$ are zero at equilibrium.

The IFCs can be evaluated by finite difference. This technique consists of calculating the difference between the forces of the system when one of the atoms displaced of a small finite shift along a Cartesian direction. The second and third order IFCs need respectively, $2N_{atoms}$, and $4N_{atoms}^2$ force calculations.

The dynamical matrix is the second order IFC rescaled by the masses,
$$
D_{i\alpha i'\alpha}=\frac{\phi''_{i\alpha i'\alpha'}}{\sqrt{m_im_{i'}}}c
$$
where c=9648.53 and is a constant used to convert $eV$ to $10\ J$ per mol. The dynamical matrix is diagonal in the phonons basis
$$
\sum_{i'\alpha'} D_{i\alpha i'\alpha'}\eta_{i'\alpha'\mu} =\eta_{i\alpha\mu} \omega_\mu^2
$$
and $\omega_\mu/(2\pi)$ are the frequencies of the normal modes of the system. The units of the dynamical matrix are returned in THz according to

$$
[V] =\frac{10J}{mol}\frac{1}{A^2 g/mol}
= 10^{24}\frac{J}{m^2 kg} = THz^2
$$


# Force Constants for Periodic Systems

## Crystals

For crystals, where there is long range order due to the periodicity, the dimensionality of the problem can be reduced. The Fourier transfom maps the large direct space onto a compact volume in the reciprocal space: the Brillouin zone. More precisely we adopt a supercell approach, where we calculate the dynamical matrix on $N_{\rm replicas}$ replicas of a unit cell of $N_{\rm unit}$ atoms, at positions $\mathbf R_l$, and calculate
$$
 D_{i \alpha k i' \alpha'}=\sum_l \chi_{kl}  D_{i \alpha l i' \alpha'},\quad \chi_{kl} = \mathrm{e}^{-i \mathbf{q_k}\cdot \mathbf{R}_{l} },
$$
where $\mathbf q_k$ is a grid of size $N_k$ indexed by $k$ and the eigenvalue equation becomes
$$
\sum_{i'\alpha'} D_{i \alpha k i' \alpha'} \eta_{i' \alpha'k s}=\omega_{k m}^{2} \eta_{i \alpha k s }
$$
which now depends on the quasi-momentum index, $k$. The $k$ indices are found with
$$
\mathbf k = 2 \pi C^{-1}\mathbf q
$$
where $C$ are unit cell vectors.

## Folding

*Under Construction*



