# `kaldo.cumulant` examples

Runnable scripts demonstrating the cumulant free-energy correction
pipeline on TDEP-format inputs.

| Script | System | Highlights |
|---|---|---|
| [`quickstart.py`](quickstart.py) | Stillinger-Weber Si @ 100 K | n_uc=2 multi-atom primitive, mesh-convergence sweep, comparison vs Julia LDT |
| [`lj_argon.py`](lj_argon.py) | Lennard-Jones Ar @ 80 K | n_uc=1 single-atom primitive (centrosymmetric), large supercell (det M=256) |
| [`gate6_ne_25cubed_from_cache.py`](gate6_ne_25cubed_from_cache.py) | Solid Neon @ 24 K (cached) | Full Gate-6 cumulant thermodynamics from pre-computed Phase 5 samples |

All three scripts assume the Julia `LatticeDynamicsToolkit.jl` package
(Ethan Meitz, CMU) is installed — it ships TDEP fixtures under
`~/.julia/packages/LatticeDynamicsToolkit/*/data/`. Replace the input
paths with your own TDEP output folder to run the same analysis on
arbitrary materials.

## Minimal usage pattern

```python
from kaldo.forceconstants import ForceConstants
from kaldo.cumulant import F1_from_fc, F2_from_fc

# 3x3x3 conventional cubic supercell of fcc Si rhombo primitive (det M = 108)
import numpy as np
M = np.array([[3, -3, 3], [3, 3, -3], [-3, 3, 3]], dtype=int)

fc = ForceConstants.from_folder(
    folder="path/to/tdep/output",
    supercell_matrix=M,        # required for non-diagonal tilings
    format="tdep",
    include_fourth=True,        # set False if no IFC4 file is available
)

r1 = F1_from_fc(fc, masses_amu=np.full(2, 28.0855),
                kmesh=(10, 10, 10), T_K=300.0,
                use_q_symmetry=True)
r2 = F2_from_fc(fc, masses_amu=np.full(2, 28.0855),
                kmesh=(10, 10, 10), T_K=300.0,
                sigma_THz=None,             # adaptive sigma
                use_q_symmetry=True)
print(r1["F1"], r2["F2"])    # in eV/atom
```

For diagonal tilings (e.g. when ssposcar is N×N×N copies of the
primitive without basis rotation), pass the scalar `supercell=(N, N, N)`
tuple instead of `supercell_matrix=`.

## Implementation reference

The implementation closely follows
[`CumulantAnalysis.jl`](https://github.com/ejmeitz/CumulantAnalysis.jl)
(reference paper code in
[`paper/`](https://github.com/ejmeitz/CumulantAnalysis.jl/tree/main/paper))
and `LatticeDynamicsToolkit.jl` by Ethan Meitz (CMU). On the
Stillinger-Weber Si test fixture our F1 and F2 values agree with Julia
LDT to ~5e-5 relative across mesh sizes 2³–5³.
