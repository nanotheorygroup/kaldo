# Output


# Default storage
When using the default storage option `default`, Ballistico stores the following dense tensor as formatted, human readable, files:
- `frequency` $(n_{kpoints}, n_{modes})$. mode changes first k changes after
- `velocity_alpha` $(n_{kpoints}, n_{modes})$
- `physical_modes_<min_freq>_<max_freq>_<is_nw>` $(n_{kpoints}, n_{modes})$

- `<temperature>/<statistics>/heat_capacity` $(n_{kpoints}, n_{modes})$
- `<temperature>/<statistics>/population` $(n_{kpoints}, n_{modes})$

- `<temperature>/<statistics>/<method>/<third_bandwidth>/<diffusivity_bandwidth>/conductivity_<alpha>_<beta>` $(n_{kpoints}, n_{modes})$ where the `<third_bandwidth>/<diffusivity_bandwidth>` folder is created only if those values are defined
- `<temperature>/<statistics>/<diffusivity_bandwidth>/diffusivity` $(n_{kpoints}, n_{modes})$
- `<temperature>/<statistics>/<third_bandwidth>/<method>/mean_free_path` $(n_{kpoints}, n_{modes})$
- `<temperature>/<statistics>/<third_bandwidth>/lifetime` $(n_{kpoints}, n_{modes})$
- `<temperature>/<statistics>/<third_bandwidth>/bandwidth` $(n_{kpoints}, n_{modes})$
- `<temperature>/<statistics>/<third_bandwidth>/phase_space` $(n_{kpoints}, n_{modes})$
- `<diffusivity_bandwidth>flux_dense` $(n_{kpoints}, n_{modes}, n_{kpoints}, n_{modes})$, when `diffusivity_threshold` is not specified.
- `<diffusivity_bandwidth>/<diffusivity_threshold>/flux_sparse` $(n_{kpoints}, n_{modes}, n_{kpoints}, n_{modes})$. Sparse only when  `diffusivity_threshold` is specified.


The folder structure depends on the input parameters and in parentesis is the shape of the tensor. All of the above observables are stored in a dense format, except for `flux_alpha` which is stored as formatted file in a `index value` format.

The following tensors are stored in raw binary format and help saving time when performing different simulations on the same sample.
- `_eigensystem (eigenvalues and eigenvectors)`
- `_dynmat_derivatives`
- `<temperature>/<statistics>/<diffusivity_bandwidth>/_generalized_diffusivity` 
- `<temperature>/<statistics>/<third_bandwidth>/_ps_and_gamma_tensor`
- `<temperature>/<statistics>/<third_bandwidth>/_ps_and_gamma`, when only RTA conductivity is required

# Alternative storages
Other options available are `numpy` and `hdf5` where all the files are saved as one of those formats.

