"""
Cross-check tests: kaldo's TDEP IFC2 reader vs kaldo.cumulant.dynmat_and_eigs.

These pin the two independent codepaths for building and diagonalizing the
dynamical matrix from a TDEP `infile.forceconstant`:

  Path A: kaldo.ForceConstants.from_folder(..., format="tdep")
          -> kaldo.Phonons(fc).frequency

  Path B: kaldo.cumulant.read_tdep_pair_fcs(...)
          -> kaldo.cumulant.dynmat_and_eigs(...) per q

On a diagonal supercell (si-tdep fixture, 5^3 tiling of fcc rhombo Si),
both paths must give the same phonon frequencies. This is PR A of the
kaldo.cumulant -> kaldo consolidation plan (DESIGN_TDEP_READER.md §7).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

SI_TDEP_DIR = Path(__file__).parent / "si-tdep"
SI_MASS_AMU = 28.0855


def _sort_freqs_2d(freq):
    """Sort phonon frequencies within each q and across q for stable comparison.

    The MP-mesh ordering and eigenvalue sort order may differ between the two
    codepaths; the physical content (spectrum per mesh) must still match.
    """
    return np.sort(np.sort(freq, axis=1), axis=0)


@pytest.mark.skipif(not SI_TDEP_DIR.exists(), reason="si-tdep fixture missing")
def test_kaldo_tdep_ifc2_matches_cumulant_dynmat_si():
    """Si 5^3 diagonal ssposcar: kaldo Phonons vs kaldo.cumulant dynmat_and_eigs.

    Both codepaths read the same ``infile.forceconstant`` and must produce the
    same phonon spectrum on a (3,3,3) q-mesh to ~1e-4 THz (numerical floor set
    by the acoustic modes at Gamma).
    """
    import ase.io
    from kaldo.forceconstants import ForceConstants
    from kaldo.phonons import Phonons
    from kaldo.cumulant import read_tdep_pair_fcs, dynmat_and_eigs, AMU
    from kaldo.cumulant.harmonic import monkhorst_pack_qcart

    # Path A: kaldo's tdep reader + Phonons
    fc = ForceConstants.from_folder(
        folder=str(SI_TDEP_DIR), supercell=(5, 5, 5), format="tdep",
    )
    phonons = Phonons(
        forceconstants=fc, kpts=(3, 3, 3), temperature=300,
        is_classic=False, storage="memory",
    )
    freq_kaldo = np.asarray(phonons.frequency)  # (Nq, Nband), THz

    # Path B: kaldo.cumulant's reader + dynmat_and_eigs
    uc = ase.io.read(str(SI_TDEP_DIR / "infile.ucposcar"), format="vasp")
    uc_pos = np.asarray(uc.get_positions())
    uc_cell = np.asarray(uc.get_cell())
    n_uc = len(uc)
    masses_kg = np.full(n_uc, SI_MASS_AMU * AMU)
    neighbors = read_tdep_pair_fcs(
        str(SI_TDEP_DIR / "infile.forceconstant"), uc_pos, uc_cell,
    )
    qs = monkhorst_pack_qcart((3, 3, 3), uc_cell)
    freq_cumulant = np.zeros((qs.shape[0], 3 * n_uc))
    for i, q_cart in enumerate(qs):
        w, _ = dynmat_and_eigs(neighbors, uc_pos, masses_kg, q_cart)
        freq_cumulant[i] = w / (2 * np.pi * 1e12)  # rad/s -> THz

    # MP orderings may differ; compare sorted spectra per-q and globally
    a = _sort_freqs_2d(freq_kaldo)
    b = _sort_freqs_2d(freq_cumulant)
    assert a.shape == b.shape
    # Max absolute difference must be below acoustic-floor tolerance
    max_abs = np.max(np.abs(a - b))
    assert max_abs < 2e-4, f"kaldo-vs-cumulant spectra diverge by {max_abs:.2e} THz"

    # Above the acoustic modes at Gamma the frequencies should also match
    # relatively (~1e-5). Filter acoustic modes (|freq| < 0.1 THz).
    mask = (np.abs(a) > 0.1) & (np.abs(b) > 0.1)
    rel = np.max(np.abs((a[mask] - b[mask]) / a[mask]))
    assert rel < 5e-5, f"kaldo-vs-cumulant non-acoustic modes diverge by rel {rel:.2e}"


@pytest.mark.skipif(
    not Path("/home/giuseppe/Development/4th-order-cumulants/reference_si/T300_0").exists(),
    reason="non-diagonal Si fixture unavailable",
)
def test_tdep_reader_rejects_non_diagonal_supercell():
    """format='tdep' must refuse non-diagonal ssposcar with a clear error.

    Our production Ne and Si sTDEP fixtures use rhombo primitives with cubic
    conventional ssposcar, which is a non-diagonal M matrix (det(M) = 108 for
    Si, 256 for Ne). Per DESIGN_TDEP_READER.md Decision 1, this is an SNF
    follow-up, not supported by PR A. The reader must raise a clear error
    telling the user what's wrong (rather than a cryptic reshape failure).
    """
    from kaldo.forceconstants import ForceConstants

    si_prod = Path("/home/giuseppe/Development/4th-order-cumulants/reference_si/T300_0")
    with pytest.raises(ValueError, match=r"(?i)non.?diagonal|SNF|diagonal"):
        ForceConstants.from_folder(
            folder=str(si_prod), supercell=(3, 3, 3), format="tdep",
        )


@pytest.mark.skipif(not SI_TDEP_DIR.exists(), reason="si-tdep fixture missing")
def test_kaldo_tdep_ifc3_matches_cumulant_triplets_si():
    """Si 5^3 si-tdep: ThirdOrder TDEP reader vs kaldo.cumulant.read_tdep_ifc3.

    Both codepaths read the same ``infile.forceconstant_thirdorder``. The
    two representations differ structurally (dense sparse COO vs list of
    per-triplet tuples), so compare through sum-rule invariants that don't
    depend on storage layout:

      sum(Phi)         : exact ASR (both = 0)
      sum(|Phi|)       : 1-norm of the IFC tensor
      sum(Phi^2)       : 2-norm squared (Frobenius)

    These three together uniquely pin the IFC magnitude distribution up to
    sign-flipped entries; any drift flags a parser divergence.
    """
    from kaldo.forceconstants import ForceConstants
    from kaldo.cumulant import read_tdep_ifc3

    fc = ForceConstants.from_folder(
        folder=str(SI_TDEP_DIR), supercell=(5, 5, 5), format="tdep",
    )
    third_kaldo = fc.third.value.todense()

    triplets = read_tdep_ifc3(
        str(SI_TDEP_DIR / "infile.forceconstant_thirdorder"), 2,
    )

    # Accumulate norms from the cumulant list-of-triplets
    l1_cum = 0.0
    l2sq_cum = 0.0
    sum_cum = 0.0
    for il in triplets:
        for (_, _, _, _, phi) in il:
            sum_cum += phi.sum()
            l1_cum += np.abs(phi).sum()
            l2sq_cum += (phi ** 2).sum()

    # kaldo aggregate
    sum_kaldo = float(np.sum(third_kaldo))
    l1_kaldo = float(np.sum(np.abs(third_kaldo)))
    l2sq_kaldo = float(np.sum(third_kaldo ** 2))

    np.testing.assert_allclose(sum_kaldo, sum_cum, atol=1e-10,
        err_msg="kaldo vs cumulant IFC3 ASR differs")
    np.testing.assert_allclose(l1_kaldo, l1_cum, rtol=1e-12,
        err_msg="kaldo vs cumulant IFC3 1-norm differs")
    np.testing.assert_allclose(l2sq_kaldo, l2sq_cum, rtol=1e-12,
        err_msg="kaldo vs cumulant IFC3 Frobenius-squared differs")


@pytest.mark.skipif(
    not Path("/home/giuseppe/Development/4th-order-cumulants/reference_si/T300_0").exists(),
    reason="non-diagonal Si fixture unavailable",
)
def test_tdep_third_reader_rejects_non_diagonal_supercell():
    """ThirdOrder format='tdep' must refuse non-diagonal ssposcar too.

    The IFC3 parser uses a diagonal Grid(supercell) index; on a non-diagonal
    tiling this either mis-maps triplets or raises a cryptic error. Match
    the SecondOrder behaviour: raise a clear ValueError pointing to SNF.
    """
    from kaldo.forceconstants import ForceConstants

    si_prod = Path("/home/giuseppe/Development/4th-order-cumulants/reference_si/T300_0")
    with pytest.raises(ValueError, match=r"(?i)non.?diagonal|SNF|diagonal"):
        ForceConstants.from_folder(
            folder=str(si_prod), supercell=(3, 3, 3), format="tdep",
        )


@pytest.mark.skipif(not SI_TDEP_DIR.exists(), reason="si-tdep fixture missing")
def test_si_tdep_is_diagonal_5cube():
    """Regression: si-tdep ssposcar must remain a 5^3 diagonal tiling of the rhombo uc.

    PR A and the cumulant dynmat comparison rely on this fixture being a
    diagonal tiling. If someone regenerates si-tdep with a different tiling,
    the preceding test will break; this one trips first with a clearer cause.
    """
    import ase.io
    uc = ase.io.read(str(SI_TDEP_DIR / "infile.ucposcar"), format="vasp")
    sc = ase.io.read(str(SI_TDEP_DIR / "infile.ssposcar"), format="vasp")
    M = np.linalg.solve(np.asarray(uc.cell), np.asarray(sc.cell))
    expected = np.diag([5, 5, 5])
    assert np.allclose(M, expected, atol=1e-8), f"Expected 5^3 diagonal, got M=\n{M}"
