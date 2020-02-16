# Phonons Object
## Input
- `is_classic`
- `temperature`
- `kpts`
- `min_frequency`
- `max_frequency`
- `broadening_shape`
- `is_nw`
- `third_bandwidth`
- `diffusivity_bandwidth`
- `diffusivity_threshold`
- `is_tf_backend`
- `folder`
- `storage`

## Output
- `frequency`
- `velocity`
- `heat_capacity`
- `population`
- `bandwidth`
- `lifetime`
- `phase_space`
- `diffusivity`
- `flux`
- `eigenvalues`
- `eigenvectors`



## Equations  (draft)

### Density and heat capacity
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

### Heat capacity

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


### Lifetime
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