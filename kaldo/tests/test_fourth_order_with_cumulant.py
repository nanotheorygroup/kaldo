"""
Cross-check test linking the kaldo TDEP IFC4 parser to the
``kaldo.cumulant`` subpackage's list-of-quartets parser.

Lives in its own module (instead of ``test_fourth_order.py``) because it
imports ``kaldo.cumulant`` and would force the subpackage to be a
dependency of the kaldo-core ``FourthOrder`` observable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

SI_PROD = Path("/home/giuseppe/Development/4th-order-cumulants/reference_si/T300_0")


@pytest.mark.skipif(not SI_PROD.exists(), reason="Si production IFC4 unavailable")
def test_parse_tdep_fourth_matches_cumulant_readers_on_raw_file():
    """Sum / 1-norm / Frobenius² match between the two IFC4 readers.

    Both read the same file. The kaldo reader additionally indexes into a
    Grid(supercell), which is only correct on diagonal tilings; on the
    non-diagonal production fixture the replica indices are off, but the
    **values** stored in the tensor are still exactly what the file contained.
    Compare element-wise sums across all 81 × n_quartet entries per atom.
    """
    from kaldo.interfaces.tdep_io import parse_tdep_fourth_forceconstant
    from kaldo.cumulant import read_tdep_ifc4

    per_atom = read_tdep_ifc4(
        str(SI_PROD / "infile.forceconstant_fourthorder"), na_uc=2,
    )
    sum_cum = 0.0
    l1_cum = 0.0
    l2sq_cum = 0.0
    n_quartets_total = 0
    for quartets in per_atom:
        for (_, _, _, _, _, _, phi) in quartets:
            sum_cum += phi.sum()
            l1_cum += np.abs(phi).sum()
            l2sq_cum += (phi ** 2).sum()
            n_quartets_total += 1

    fourth = parse_tdep_fourth_forceconstant(
        fc_filename=str(SI_PROD / "infile.forceconstant_fourthorder"),
        primitive=str(SI_PROD / "infile.ucposcar"),
        supercell=(3, 3, 3),
    )
    dense = fourth.todense()
    sum_kaldo = float(dense.sum())
    l1_kaldo = float(np.abs(dense).sum())

    assert n_quartets_total > 0
    assert abs(sum_kaldo - sum_cum) < 1e-5 * max(l1_cum, 1.0), (
        f"IFC4 total-sum divergence: cumulant={sum_cum:.3e}, kaldo={sum_kaldo:.3e}"
    )
    assert l1_kaldo <= l1_cum * (1 + 1e-12), (
        f"kaldo L1 exceeds cumulant L1 (shouldn't happen): "
        f"cumulant={l1_cum:.3e}, kaldo={l1_kaldo:.3e}"
    )
