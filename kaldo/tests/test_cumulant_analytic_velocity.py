"""
Analytic (Hellmann-Feynman) velocity vs finite-difference equivalence.

kaldo.cumulant.compute_group_velocity currently uses central finite differences
(6 extra dynmat diagonalizations per q). Julia LDT uses analytic Hellmann-
Feynman: ∂ω²/∂q_α = <e|∂D/∂q_α|e>, then ∂ω/∂q_α = ∂ω²/(2ω).

The analytic path should match FD to well under 1e-4 absolute (FD noise floor
from dq=1e-4), and be 6× fewer diagonalizations. These tests pin that match
and allow a future PR to replace the FD implementation with the analytic one
without silently changing numbers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

SI_TDEP_DIR = Path(__file__).parent / "si-tdep"
SI_MASS_AMU = 28.0855


@pytest.mark.skipif(not SI_TDEP_DIR.exists(), reason="si-tdep fixture missing")
def test_analytic_velocity_matches_finite_difference_si():
    """compute_group_velocity_analytic == compute_group_velocity on Si 3^3.

    The analytic routine uses Hellmann-Feynman: ∂ω²/∂q_α = <e|∂D/∂q_α|e>;
    the FD routine samples D at ±dq and differences ω. Both should agree to
    roughly the FD noise floor ~ ω × dq² ~ 1e-4 rad/s absolute.
    """
    import ase.io
    from kaldo.cumulant import (
        read_tdep_pair_fcs, dynmat_and_eigs, AMU,
    )
    from kaldo.cumulant.free_energy import (
        compute_group_velocity,
        compute_group_velocity_analytic,
    )
    from kaldo.cumulant.harmonic import monkhorst_pack_qcart

    uc = ase.io.read(str(SI_TDEP_DIR / "infile.ucposcar"), format="vasp")
    uc_pos = np.asarray(uc.get_positions())
    uc_cell = np.asarray(uc.get_cell())
    n_uc = len(uc)
    masses_kg = np.full(n_uc, SI_MASS_AMU * AMU)
    neighbors = read_tdep_pair_fcs(
        str(SI_TDEP_DIR / "infile.forceconstant"), uc_pos, uc_cell,
    )
    qs = monkhorst_pack_qcart((3, 3, 3), uc_cell)

    for iq, q in enumerate(qs):
        om, egv = dynmat_and_eigs(neighbors, uc_pos, masses_kg, q)
        v_fd = compute_group_velocity(neighbors, uc_pos, masses_kg, q, om, egv)
        v_an = compute_group_velocity_analytic(neighbors, uc_pos, masses_kg, q, om, egv)
        mask = np.abs(om) > 1e11
        if not mask.any():
            continue
        # Adaptive sigma uses |v|, which is gauge-invariant under degenerate
        # subspace mixing. Compare |v| per band — values must agree to FD
        # noise floor (~1e-4 relative, 2nd-order central difference error).
        norm_fd = np.linalg.norm(v_fd[:, mask], axis=0)
        norm_an = np.linalg.norm(v_an[:, mask], axis=0)
        scale = max(1.0, norm_fd.max())
        diff = np.max(np.abs(norm_fd - norm_an))
        assert diff / scale < 5e-4, (
            f"iq={iq}: analytic vs FD |v| differs by {diff:.3e} "
            f"(scale {scale:.3e}, rel {diff/scale:.2e})"
        )
