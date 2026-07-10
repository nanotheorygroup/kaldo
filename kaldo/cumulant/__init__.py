"""
kaldo.cumulant - anharmonic free-energy corrections via phonon cumulants.

Recommended (kaldo-native) entry points:

    from kaldo.forceconstants import ForceConstants
    from kaldo.cumulant import F1_from_fc, F2_from_fc, cumulant_thermo

    fc = ForceConstants.from_folder(
        folder, supercell_matrix=M, format="tdep", include_fourth=True,
    )
    F1 = F1_from_fc(fc, masses_amu, kmesh=(10, 10, 10), T_K=300.0,
                    use_q_symmetry=True)["F1"]
    F2 = F2_from_fc(fc, masses_amu, kmesh=(10, 10, 10), T_K=300.0,
                    use_q_symmetry=True)["F2"]

Legacy list-based API (still supported for back-compat):

    from kaldo.cumulant import (
        read_tdep_pair_fcs, read_tdep_ifc3, read_tdep_ifc4,
        dynmat_and_eigs,
        F1_vectorized, F2_vectorized,
        load_tdep_folder,   # one-shot reader returning the list format
    )

Implements Ethan Meitz's `CumulantAnalysis.jl` + `LatticeDynamicsToolkit.jl`
algorithms as a vectorized Python port, validated bit-for-bit against
Julia LDT on Ne (n_uc=1) and Si diamond (n_uc=2).

Entry-point modules:
  * :mod:`kaldo.cumulant.constants` - constants, TDEP IFC parsers, dynmat
  * :mod:`kaldo.cumulant.harmonic` - F_H / U_H / S_H / Cv_H
  * :mod:`kaldo.cumulant.free_energy` - F1 (quartic) and F2 (cubic);
    preferred entry points are ``F1_from_fc`` / ``F2_from_fc`` which
    consume a ``kaldo.ForceConstants`` object and work on both diagonal
    and non-diagonal TDEP supercells.
  * :mod:`kaldo.cumulant.sampler` - supercell-Gamma canonical sampler
  * :mod:`kaldo.cumulant.taylor` - SC-remapped V_2/V_3/V_4 contractors
  * :mod:`kaldo.cumulant.estimator` - calculate_cumulants (F_offset etc.)
  * :mod:`kaldo.cumulant.bootstrap` - bootstrap SEs
  * :mod:`kaldo.cumulant.api` - ``cumulant_thermo`` top-level runner
"""
from __future__ import annotations

from .constants import (
    HBAR, KB, EV, AMU, ANG, NE_MASS_AMU, FREQ_TOL_THZ, KB_eV_per_K,
)
from .common import (
    dynmat_and_eigs, load_tdep_folder,
    read_tdep_pair_fcs, read_tdep_ifc3, read_tdep_ifc4,
)

from .harmonic import (
    harmonic_thermo_quantum,
    monkhorst_pack_qcart,
    compute_all_frequencies_THz,
    harmonic_thermo_from_ifc2,
)

from .free_energy import (
    flatten_quartets, build_psi4_realspace_v, F1_vectorized,
    flatten_triplets, build_psi3_realspace, F2_vectorized,
    F1_from_fc, F2_from_fc,
    planck_and_derivs,
    compute_group_velocity, compute_group_velocity_analytic,
    compute_default_smearing, adaptive_sigma,
)

from .sampler import SCSampler
from .taylor import SCContractors
from .estimator import calculate_cumulants, dA_dT, d2A_dT2
from .bootstrap import bootstrap_corrections
from .api import cumulant_thermo, CumulantResult, print_ethan_table

# Public surface — kept minimal. Internal helpers (constants like HBAR,
# parsers like read_tdep_*, kernel helpers like flatten_* and build_psi*,
# adaptive-sigma shims, etc.) remain importable from their submodules
# (e.g. ``from kaldo.cumulant.common import HBAR``) but are not part of
# the documented top-level API.
__all__ = [
    # cumulant-specific constants
    "NE_MASS_AMU", "FREQ_TOL_THZ",
    # one-shot TDEP folder reader (still useful for non-diagonal SNF)
    "load_tdep_folder",
    # kaldo-native entry points (recommended)
    "F1_from_fc", "F2_from_fc",
    # legacy list-based math kernels (kept as a cross-check; tests import them)
    "F1_vectorized", "F2_vectorized",
    # MC pipeline
    "SCSampler", "SCContractors",
    "calculate_cumulants",
    "bootstrap_corrections",
    # top-level API
    "cumulant_thermo", "CumulantResult", "print_ethan_table",
]
