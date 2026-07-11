"""
Phase-5 atom-frame consistency for the cumulant constant-correction sampler.

The 0th-order (constant) correction draws harmonic displacements ``u`` from
``SCSampler`` and evaluates the Taylor potentials with ``SCContractors``. The
two must speak the SAME supercell atom order or every term is silently wrong:

  * ``SCContractors`` index atoms in infile.ssposcar file order.
  * ``ForceConstants.irred_to_full`` produces the supercell IFC2 in
    replica-major order (replica_id * n_uc + atom_uc).

Phase 5 reindexes the IFC2 into ssposcar order before building the sampler,
so the drawn ``u`` is already in the contractor frame. This test pins that:
``contractors.V2(u)`` must equal the exact quadratic form ``0.5 uᵀ Φ u`` of
the sampler's own (file-frame) IFC2 to machine precision.

Regression: with the un-permuted (replica-major) IFC2 the two disagree by
order 100 % (measured ~7.2 eV vs ~2.8 eV on the SW 3×3×3 fixture).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import ase.io

FIX = Path(__file__).parent / "cumulant_fixtures" / "SW" / "100K_3UC"


@pytest.mark.skipif(not (FIX / "infile.ssposcar").exists(), reason="SW cumulant fixture missing")
def test_sampler_and_contractors_share_ssposcar_frame():
    from kaldo.forceconstants import ForceConstants
    from kaldo.cumulant.sampler import SCSampler
    from kaldo.cumulant.contractors import SCContractors
    from kaldo.interfaces.tdep_io import build_supercell_replica_mapping

    uc = ase.io.read(str(FIX / "infile.ucposcar"), format="vasp")
    sc = ase.io.read(str(FIX / "infile.ssposcar"), format="vasp")
    n_uc, n_sc = len(uc), len(sc)

    fc = ForceConstants.from_folder(str(FIX), supercell=(3, 3, 3), format="tdep", only_second=True)

    # Replica-major -> ssposcar (file) reindex, exactly as Phase 5 does it.
    mapping = build_supercell_replica_mapping(uc, sc)
    replica_major_of_file = mapping["replica_id_of_sc"] * n_uc + mapping["atom_of_sc"]
    dof_perm = (replica_major_of_file[:, None] * 3 + np.arange(3)).reshape(-1)
    ifc2_sc = fc.irred_to_full(order=2).reshape((3 * n_sc, 3 * n_sc))[np.ix_(dof_perm, dof_perm)]

    sampler = SCSampler(ifc2_sc, sc.get_masses(), T_K=100.0, is_classic=True, seed=1234)
    contractors = SCContractors.from_tdep_folder(str(FIX), include_fourth=False)

    # For several draws, contractors.V2(u) must equal 0.5 uᵀ Φ_file u.
    max_abs_err = 0.0
    v2_scale = 0.0
    for _ in range(25):
        u, _z = sampler.draw_with_z()
        uf = u.reshape(-1)
        v2_exact = 0.5 * uf @ ifc2_sc @ uf
        v2_contract = contractors.V2(u)
        max_abs_err = max(max_abs_err, abs(v2_contract - v2_exact))
        v2_scale = max(v2_scale, abs(v2_exact))

    assert v2_scale > 0.0
    # Machine precision relative to the ~eV scale of V2.
    assert max_abs_err < 1e-9 * max(v2_scale, 1.0), (
        f"contractors.V2 disagrees with the sampler's file-frame quadratic form "
        f"by {max_abs_err:.3e} eV (V2 scale {v2_scale:.3f} eV): the Phase-5 atom "
        f"frames are out of sync."
    )


@pytest.mark.skipif(not (FIX / "infile.ssposcar").exists(), reason="SW cumulant fixture missing")
def test_unpermuted_ifc2_is_detectably_wrong():
    """Guard rail: the OLD replica-major IFC2 (no reindex) DOES disagree with
    the contractors, so this test would have caught the original bug."""
    from kaldo.forceconstants import ForceConstants
    from kaldo.cumulant.sampler import SCSampler
    from kaldo.cumulant.contractors import SCContractors

    sc = ase.io.read(str(FIX / "infile.ssposcar"), format="vasp")
    n_sc = len(sc)
    fc = ForceConstants.from_folder(str(FIX), supercell=(3, 3, 3), format="tdep", only_second=True)

    ifc2_replica_major = fc.irred_to_full(order=2).reshape((3 * n_sc, 3 * n_sc))
    sampler = SCSampler(ifc2_replica_major, sc.get_masses(), T_K=100.0, is_classic=True, seed=1234)
    contractors = SCContractors.from_tdep_folder(str(FIX), include_fourth=False)

    u, _z = sampler.draw_with_z()
    uf = u.reshape(-1)
    v2_exact = 0.5 * uf @ ifc2_replica_major @ uf
    v2_contract = contractors.V2(u)  # contractors are in file order, u is replica-major
    rel = abs(v2_contract - v2_exact) / max(abs(v2_exact), 1e-30)
    assert rel > 0.05, "expected the un-reindexed IFC2 to visibly disagree with the contractors"
