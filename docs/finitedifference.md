# FiniteDifference Object
## Input
- `supercell`
- `calculator`
- `second_order`
- `third_order`
- `third_order_symmerty_inputs`
- `is_reduced_second`
- `delta_shift`
- `calculator`
- `calculator_inputs`
- `is_optimizing`
- `distance_threshold`
- `folder`

## Output
- `dynamical_matrix`
- `second_order`
- `third_order`
- `replicated_atoms`


## Equations  (draft)

### Input
Potential derivatives are in
$eV/A^2$ or $eV/A^3$. Distances are in $A$. Masses are in $g/N_A$

### Dynamical matrix
If $C$ are the cell vectors
$$
\mathbf k = 2 \pi C^{-1}\mathbf q
$$
We introduce the Bloch waves unitary tranformation
$$
\chi_l(\mathbf k) = e^{i \mathbf R_l\dot\mathbf k}
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
\omega(\mathbf k)_\mu^2 = \sum_{ij}\eta_{\mu i}^{*T}D_{ii'}(\mathbf k)\eta_{i'\mu} =
$$
$$
v_{\mu\alpha}(\mathbf k) = \frac{1}{2\omega_\mu(\mathbf k)}\sum_{ij}(\eta)^{*T}_{\mu i}E(\mathbf k)_{ii'\alpha}\eta_{i'\mu}
$$
Units
$$
[\omega] = \sqrt{THz^2} = THz
$$
$$
[v] = \frac{1}{THz}A THz^2 = A\cdot THz = 100m/s
$$