# QE 7.6 q2r/matdyn NAC references

These fourteen directories cover every three-dimensional Bravais class. Each
case retains the plain-text Quantum ESPRESSO inputs used for SCF, dielectric
and Born-charge calculation, the `2x2x2` phonon grid, q2r, and matched matdyn
runs. The NAC-on and NAC-off matdyn inputs differ only in whether the q2r
macroscopic header is present.

The checked-in `<case>.fc`, `<case>.on.dyn`, and `<case>.off.dyn` files are
reference outputs, not scratch data. Tests generate `<case>.without-nac.fc`
from the polar q2r file with `kaldo.interfaces.qe_io.strip_q2r_nac`, verify that
the IFC body is unchanged, and compare the Python rigid-ion tensor directly
with QE's readable matdyn matrix difference. `run_case.sh` performs the same
deterministic stripping step before reproducing the NAC-off `matdyn.x` output.

`manifest.json` records structures, pseudopotentials, and campaign settings.
The pseudopotential files themselves, wavefunctions, charge-density files,
temporary dynamical matrices, and QE `.save` directories are intentionally not
committed. Copy the listed UPF files into a case's `pseudo/` directory and run:

```bash
QE_BIN=/path/to/qe-7.6/bin MPI_COMMAND="mpirun -n 2" ./run_case.sh CASE_DIRECTORY
```

The uranium case is retained for lattice/kernel coverage, as documented in
the manifest; it is not a material-accuracy reference without an independently
validated magnetic and DFT+U protocol.
