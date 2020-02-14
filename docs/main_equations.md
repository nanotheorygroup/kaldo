# Structure
![engine](_resources/engine.png)
Here is a draft of the main equations implemented in the code with a reference to the units.



# Input Units
Potential derivatives are in
$eV/A^2$ or $eV/A^3$. Distances are in $A$. Masses are in $g/N_A$

# Density
$$
\mathrm{kelvintothz} = 10^{-12} \frac{k_B}{ 2 \pi \hbar J} =0.02083
$$
$k_B$ is in $eV/K$ and $J$ means $J/eV=1/e$.
Define the tilde-temperature in $THz$
$$
\tilde T =  T\mathrm{kelvintothz}
$$
the Bose Einstein distribution for each phonon mode $\mu$

$$
n_\mu= \frac{1}{e^{\nu_\mu/\tilde T}-1}
$$

## Heat capacity

If we define
$$
\mathrm{kelvintojoule} = \frac{k_B}{J}=1.38064852\cdot 10^{-23}
$$

The classical heat capacity is defined as

$$
{c}_\mu = \mathrm{kelvintojoule}
$$

while the quantum

$$
{c}_\mu= \frac{\nu_\mu^2}{\tilde T^2}n_\mu (n_\mu + 1)\mathrm{kelvintojoule}
$$

# Dynamical matrix

If $C$ are the cell vectors
$$
\mathbf k = 2 \pi C^{-1}\mathbf q
$$

We introduce the Bloch waves unitary tranformation

$$
\chi(\mathbf k)_l = e^{i \mathbf R_l\dot\mathbf k}
$$

where $l$ is the index of the cell replica.

We rewrite the second derivative of the potential
$$
V_{lil'i'} =\frac{V_{lil'i'}}{\sqrt{m_im_i'}}\mathrm{evtotenjovermol}
$$
where $i$ is the index of the atom in the unit cell (replica). $\mathrm{evtotenjovermol}=9648.53$
Units:

$$
[V] =\frac{10J}{mol}\frac{1}{A^2 g/mol}
= 10^24\frac{J}{m^2 kg} = THz^2
$$

In order to calculate frequencies and velocities, we create two new tensors
$$
D(\mathbf k)_{ii'} = \sum_l V_{0ili'}\chi(\mathbf k)_l
$$
$$
E(\mathbf k)_{ii'\alpha} = i R_{l \alpha} V_{0ili'}\chi(\mathbf k)_l
$$
And diagonalize them
$$
\omega(\mathbf k)^2_\mu = \sum_{ij}(\eta)^{*T}_{\mu i}D(\mathbf k)_{ii'}\eta_{i'\mu} =
$$
$$
v(\mathbf k)_{\mu\alpha} = \frac{1}{2\omega(\mathbf k)_\mu}\sum_{ij}(\eta)^{*T}_{\mu i}E(\mathbf k)_{ii'\alpha}\eta_{i'\mu}
$$
Units
$$
[\omega] = \sqrt{THz^2} = THz
$$
$$
[v] = \frac{1}{THz}A THz^2 = A\cdot THz = 100m/s
$$
# Lifetime
$$
\tilde{\eta}_{i\mu} = \frac{{\eta}_{i\mu}}{\sqrt{m_\mu}}
$$
$$
V^{\pm}_{\mu\mu'\mu''}=\sum_{i}V_{0il'i'l''i''}\tilde\eta_{\mu i}\tilde\eta_{\mu i}\tilde\eta_{\mu i}\chi^\pm(\mathbf k')_{l'}\chi^-(\mathbf k'')_{l''}
$$
where $\chi^+ = \chi$ and $\chi^- = \chi^*$.
$$
\delta^\pm_{\mu\mu'\mu''} = \frac{1}{\sqrt{\pi\sigma^2}} e^{-\frac{(\omega_\mu \pm \omega_{\mu'} - \omega_{\mu''})^2}{\sigma^2}}
$$
Units:
$$
[\delta] = \frac{1}{[\omega]} = \frac{1}{2 \pi THz}
$$

$$
g^+_{\mu\mu'\mu''} =\frac{(n_{\mu'}-n_{\mu''})}{\omega_\mu\omega_{\mu'}\omega_{\mu''}} \delta^+_{\mu\mu'\mu''}
$$
$$
g^-_{\mu\mu'\mu''} =\frac{(1 + n_{\mu'}+n_{\mu''})}{\omega_\mu\omega_{\mu'}\omega_{\mu''}} \delta^-_{\mu\mu'\mu''}
$$
units:
$$
[g]=  \frac{1}{[\omega]^4}
$$
$$
\mathrm{gammatothz} = 10^{11}N_A\mathrm{evtotenjovermol}^2
$$
$$
\gamma^\pm_{\mu\mu'\mu''} = \frac{\hbar\pi}{4}\frac{1}{N_k}|V^\pm_{\mu\mu'\mu''}|^2 g^\pm_{\mu\mu'\mu''} \mathrm{gammatothz}
$$
Units
$$
[\gamma^\pm]=THz
$$

## Conductivity
$$
\lambda_{\mu\alpha} ={\gamma_\mu}{v_{\mu\alpha}}
$$
$$
\kappa_{\mu\alpha\alpha'}=\frac{1}{N_k\mathrm{det}C}(10^{22}c_\mu) {v_{\mu\alpha}}\lambda_{\mu\alpha'}
$$
Units
$$
[\kappa] = W/m/K
$$
