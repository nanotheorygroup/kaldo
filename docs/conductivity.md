# Conductivity Object
## Input
- `method`
- `length`
- `finite_length_method`
- `n_iterations`
- `tolerance`
- `storage`

## Output
- `conductivity`
- `mean_free_path`


# Conductivity Theory (draft)
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

## Finite Size models

### Mckelvey Schockley method

Heat

$$
q_{\mu}=\frac{1}{v_{\mu}^{+}}(j_{\mu}^{+}+j_{\mu}^{-})
$$

with

$$
v^\pm=\theta(\pm v_\mu)v_\mu
$$

the current is constant

$$
j_\mu = (j_\mu^+-j_\mu^-)=\tilde j_\mu
$$

A simple BTE in which the forward and reverse fluxes have been integrated over angle. This particular discretization is especially effective in handling the correct physical boundary conditions, where a flux is injected from each side. Inside the device, the carrier distribution can be very far from equilibrium, but each half of the distribution is at equilibrium with its originating contact as it is injected and scattering grad- ually mixes both flux components. In the limiting case of purely ballistic transport, each half of the distribution is in equilibrium with its originating contact.

$$
\frac{dn_\mu^{+}}{dt}= v_\mu^{+}\frac{dn_\mu^{+}}{dx} = -\gamma_{\mu}(n_{\mu}^+ - n_{\mu}^-) =  -\gamma_{\mu}\tilde n_{\mu}
$$

$$
\frac{dn_\mu^{-}}{dt}= v_\mu^{+}\frac{dn_\mu^{-}}{dx} = -\gamma_{\mu}(n_{\mu}^+ - n_{\mu}^-) =  -\gamma_{\mu}\tilde n_{\mu}
$$

If we introduce $j^+/j^−$, the forward/backward phonon fluxes,

$$
\frac{dj_^+\mu}{dx} = -\frac{1}{\lambda_\mu}(j_\mu^+ - j_\mu^-) =  -\frac{1}{\lambda_\mu}\tilde j_\mu
$$

$$
\frac{dj_\mu^-}{dx} = -\frac{1}{\lambda_\mu}(j_\mu^+ - j_\mu^-) =  -\frac{1}{\lambda_\mu}\tilde j_\mu
$$

$\lambda_\mu=\lambda( \omega_\mu)$ is the mean-free-path for backscattering and $\hbar \omega_\mu$ is the phonon energy. The above coupled equations describe the evolution of each flux type, which can scatter to/from the opposite flux component. $\lambda_mu$ governs the scattering, and is defined as the average distance travelled along $x$ by a phonon with energy $\hbar \omega_\mu$ before scattering into an opposite moving state

Solving the differential equations

$$
j_\mu^+(x) = j_{\mu, -L/2}^+ -\frac{(x + L/2)}{\lambda_\mu} \tilde{j}_\mu
$$

$$
j_\mu^-(x) = j_{\mu, +L/2}^- -\frac{(x - L/2)}{\lambda_\mu} \tilde{j}_\mu
$$

Finally the heat flux

$$
j_\mu = \tilde j_\mu = j_\mu^+(x) - j_\mu^-(x)
= j_\mu^+(L/2)- j_^+\mu(L/2) 
-\frac{1}{\lambda_\mu}\tilde{j}_\mu (x + L/2-  (x - L/2))
$$

$$
j_\mu^+(L/2)- j_^+\mu(L/2) =
j_\mu( 1
+L/\lambda_\mu)
$$

$$
j_{\mu, -L/2}^{+}=\hbar \omega_\mu v_\mu f_{\mu}^{BE}\left(T_{ -L/2}\right)\\
$$

$$
j_{\mu, L/2}^{-}=\hbar \omega_\mu v_\mu f_{\mu}^{BE}\left(T_{+ L/2}\right)
$$

$$
j_{\mu}=\hbar \omega_\mu v_\mu\left(\frac{\lambda_\mu}{\lambda_\mu+L}\right)
\left(f_\mu^{BE}\left(T_{-L/2}\right)-f_{\mu}^{BE}\left(T_{ +L/2}\right)\right)
$$

#### Currents 
$$
2 \delta j_\mu^+ =-\left.\kappa_\mu \frac{\mathrm{d}(\delta T)}{\mathrm{d} x}\right|_{-L/2}+c_{\mu} v_{\mu}^{+} \delta T\left(-\frac{L}{2}\right) 
$$

$$
2 \delta j_\mu^- =+\left.\kappa_\mu \frac{\mathrm{d}(\delta T)}{\mathrm{d} x}\right|_{+L/2}+c_{\mu} v_{\mu}^{+} \delta T\left(+\frac{L}{2}\right)
$$

![ff5ccc2f0de5d6ab2783d05264047152.png](_resources/ffb896b9bb7245bbab9f5e0c45f985ff.png)

$$
T(x)=\delta T(x)+T_{+ L/2}
$$

$$
\delta T(x)=\left(\delta T^{+}(x)+\delta T^{-}(x)\right) / 2
$$

for each $\mu$

$$
\delta T^{\pm}(x)=\frac{2 \delta j_{\mu}^{\pm}(x)}{c_{\mu} v_{\mu}^{+}}
$$

#### Geralization to scattering tensor
$$
\frac{dn_\mu^{+}}{dt} = v^{+}_\mu\frac{dn^{+}_\mu}{dx} =v_\mu^{+} \frac{dT_\mu}{dx}\frac{dn_\mu}{dT} = -\sum_\mu'\Gamma_{\mu\mu'}(n_{\mu'}^+ - n_{\mu’}^-)
$$


We can rewrite it in terms of currents
$$
 v_\mu^{+}\frac{dn_\mu^{+}}{dx} 
 = -\sum_{\mu'}\Gamma_{\mu\mu'}(n_{\mu'}^+ - n_{\mu’}^-) = 
{n_\mu}\delta n_{\mu'}
$$

$$
\frac{dj_\mu^{+}}{dx} 
 = -\sum_{\mu'}\frac{\omega_\mu}{\omega_{\mu’}}\frac{1}{v_{\mu’}}\Gamma_{\mu\mu'}(j_{\mu'}^+ - j_{\mu’}^-)
 = -\sum_{\mu'}\frac{1}{v_{\mu’}} \tilde\Gamma_{\mu\mu'}(j_{\mu'}^+ - j_{\mu’}^-)
$$

$$
\frac{dj_\mu^{+}}{dx} 
 = -\sum_{\mu'}\frac{1}{v_{\mu’}} \tilde\Gamma_{\mu\mu'}(j_{\mu'}^+ - j_{\mu’}^-)
$$

$$
\frac{dj_\mu^{-}}{dx} 
 = -\sum_{\mu'}\frac{1}{v_{\mu’}} \tilde\Gamma_{\mu\mu'}(j_{\mu'}^+ - j_{\mu’}^-)
$$

### Caltech method

$$
v_{\mu} \frac{\partial n_{\mu}}{\partial x}=-\frac{n_{\mu}-n_{\mu}^{0}(\delta T_\mu(x))}{\tau_{\mu}}
$$

If the boundaries are $n_{\mu,\pm\frac{L}{2}}^{\mp}=n_{\mu}^{\mp}(x=\pm L/2)$, and
$\delta T_\mu$ is the temperature change per mode
$$
n_{\mu}^{+}(x)=n_{\mu, -\frac{L}{2}}^+ e^{-\frac{x + L/2}{\lambda_{\mu}} }+  \frac{c_{\mu}}{\hbar \omega_\mu\lambda_{\mu}} \int_{-L/2}^{x}\delta T_\mu^+\left(x^{\prime}\right) e^{-\frac{x-x^{\prime}}{\lambda_{\mu}}} d x^{\prime}
$$

$$
n_{\mu}^{-}(x)=n_{\mu, +\frac{L}{2}}^-e^{-\frac{x-L/2}{\lambda_{\mu}}} - \frac{c_{\mu}}{\hbar \omega_\mu\lambda_{\mu}}\int_{x}^{L/2} \delta T_\mu^-\left(x^{\prime}\right) e^{-\frac{x-x^{\prime}}{\lambda_{\mu}}} d x^{\prime}
$$


$$
\delta T_\mu^{\pm}(x)=\frac{2\hbar \omega_\mu n_{\mu}^{\pm}(x)}{c_{\mu}}
$$

$$
n_{\mu}^{+}(x)=n_{\mu, -\frac{L}{2}}^+ e^{-\frac{x + L/2}{\lambda_{\mu}} }+  \frac{2}{\lambda_{\mu}} \int_{-L/2}^{x}
n_{\mu}^{+}
\left(x^{\prime}\right) e^{-\frac{x-x^{\prime}}{\lambda_{\mu}}} d x^{\prime}
$$

$$
n_{\mu}^{-}(x)=n_{\mu, +\frac{L}{2}}^-e^{-\frac{x-L/2}{\lambda_{\mu}}} - \frac{2}{\lambda_{\mu}}\int_{x}^{L/2}
n_{\mu}^{-}
\left(x^{\prime}\right) e^{-\frac{x-x^{\prime}}{\lambda_{\mu}}} d x^{\prime}
$$


## Allen-Feldman and Quasi-Harmonic-Green-Kubo

### Allen-Feldman Model

### Quasi-Harmonic-Green-Kubo Model