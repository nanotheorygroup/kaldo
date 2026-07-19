"""
Regression test for the non-analytic-correction (NAC) Ewald phase on non-FCC
polar cells.

Background
----------
``HarmonicWithQ.nac_dynmat`` builds the reciprocal-space dipole phase in a
normalized unit system and historically applied a prefactor of ``pi``. That
reproduces the physical phase ``exp(i G.r)`` only when ``c_recip * a_max == 0.5``
-- an identity that holds for an FCC primitive cell and for no other Bravais
lattice. On any non-FCC polar cell the phase was applied at half its correct
value, which (among other things) *splits symmetry-required degeneracies* that a
physically correct NAC must preserve.

These tests are deliberately reference-light: they assert physical invariants
(a symmetry-mandated degeneracy that the half-phase breaks, and FCC agreement
with the trusted MgO matdyn.x reference) rather than absolute hexagonal
frequencies, because kaldo's hexagonal Wigner-Seitz interpolation at
incommensurate q has separate, pre-existing differences vs matdyn.x (see the
tolerance note in ``test_gan_hex_qe.py``).

The hexagonal degeneracy test FAILS on the unpatched code (TA split ~36 cm^-1)
and PASSES once the phase is evaluated in the physical ``exp(i G.r)`` convention.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from kaldo.forceconstants import ForceConstants
from kaldo.observables.harmonic_with_q import HarmonicWithQ

THZ_TO_CM = 33.3564095198152

# Generic, physically reasonable wurtzite Born/dielectric constants. The GaN
# fixture ships with NAC flag 'F' (no Born block); we inject a block so the
# hexagonal NAC path is exercised. Exact values are immaterial to the symmetry
# test -- the transverse-acoustic pair along Gamma-A is degenerate for *any*
# C6v-consistent choice.
GAN_EPS = [[5.35, 0.0, 0.0], [0.0, 5.35, 0.0], [0.0, 0.0, 5.48]]
GAN_Z_GA = [[2.7, 0.0, 0.0], [0.0, 2.7, 0.0], [0.0, 0.0, 2.8]]
GAN_Z_N = [[-2.7, 0.0, 0.0], [0.0, -2.7, 0.0], [0.0, 0.0, -2.8]]

# MgO (FCC) matdyn.x reference (ships with an in-file NAC block); this is the
# trusted control that must stay bit-for-bit unchanged by the fix.
MGO_Q = [(0.3, 0.0, 0.3), (0.1, 0.0, 0.0), (0.15, 0.15, 0.15), (0.3, 0.1, 0.0)]
MGO_MATDYN = [
    [239.7640, 239.7640, 367.6916, 422.9322, 422.9322, 582.6500],
    [77.0501, 77.0501, 135.2487, 389.4064, 389.4064, 691.1982],
    [113.7407, 113.7407, 200.8237, 387.8639, 387.8639, 684.7630],
    [203.9083, 210.4709, 344.9263, 386.7525, 394.9880, 643.5431],
]


def _inject_gan_born(src_dir, dst_dir):
    """Copy the GaN QE fixture and turn on NAC by injecting a Born block."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(src_dir) / "POSCAR", dst_dir / "POSCAR")
    lines = (Path(src_dir) / "espresso.ifc2").read_text().splitlines()
    flag_idx = next(i for i, ln in enumerate(lines[:12]) if ln.strip() in ("F", "T"))
    fmt3 = lambda row: "".join("%18.10f" % x for x in row)
    zeff = [GAN_Z_GA, GAN_Z_GA, GAN_Z_N, GAN_Z_N]
    block = [" T"] + [fmt3(r) for r in GAN_EPS]
    for ia in range(4):
        block.append("%5d" % (ia + 1))
        block.extend(fmt3(r) for r in zeff[ia])
    new = lines[:flag_idx] + block + lines[flag_idx + 1:]
    (dst_dir / "espresso.ifc2").write_text("\n".join(new) + "\n")


@pytest.fixture(scope="module")
def gan_nac_second(tmp_path_factory):
    src = Path(__file__).resolve().parent / "gan"
    dst = tmp_path_factory.mktemp("gan_nac") / "gan"
    _inject_gan_born(src, dst)
    fc = ForceConstants.from_folder(
        folder=str(dst), supercell=[5, 5, 5], only_second=True, format="qe-d3q"
    )
    assert "dielectric" in fc.second.atoms.info, "Born block was not read"
    return fc.second


def _freqs_cm(second, q_point, is_unfolding=True):
    hwq = HarmonicWithQ(
        q_point=np.array(q_point, float), second=second,
        is_unfolding=is_unfolding, storage="memory",
    )
    return np.sort(np.array(hwq.frequency).flatten()) * THZ_TO_CM


def test_hexagonal_TA_degeneracy_along_gamma_A(gan_nac_second):
    """Along Gamma-A (q || c) the two transverse-acoustic branches are
    C6v-symmetry-required to be degenerate. The FCC-only half-phase splits
    them by tens of cm^-1; the physical phase restores the degeneracy."""
    f = _freqs_cm(gan_nac_second, (0.0, 0.0, 0.3), is_unfolding=True)
    assert abs(f[0] - f[1]) < 1.0, f"TA pair split by {abs(f[0]-f[1]):.2f} cm^-1"
    # the next (low optical) pair is likewise degenerate along Gamma-A
    assert abs(f[2] - f[3]) < 1.0


def test_hexagonal_NAC_is_active_and_shifts_spectrum(gan_nac_second):
    """Sanity guard: with a Born block the NAC must actually move the spectrum
    away from the NAC-off short-range result (otherwise the phase test is
    vacuous)."""
    on = _freqs_cm(gan_nac_second, (0.0, 0.0, 0.1), is_unfolding=True)
    # NAC-off reference
    import tensorflow as tf
    orig = HarmonicWithQ.nac_dynmat
    HarmonicWithQ.nac_dynmat = (
        lambda self, qpoint=None, gmax=None, Lambda=None:
        tf.zeros([len(self.second.atoms) * 3] * 2, dtype=tf.complex64)
    )
    try:
        off = _freqs_cm(gan_nac_second, (0.0, 0.0, 0.1), is_unfolding=True)
    finally:
        HarmonicWithQ.nac_dynmat = orig
    assert np.abs(on - off).max() > 5.0


@pytest.fixture(scope="module")
def mgo_second():
    src = Path(__file__).resolve().parent / "mgo"
    fc = ForceConstants.from_folder(
        folder=str(src), supercell=[5, 5, 5], only_second=True, format="qe-d3q"
    )
    return fc.second


@pytest.mark.parametrize("qi", range(len(MGO_Q)))
def test_fcc_mgo_nac_unchanged_vs_matdyn(mgo_second, qi):
    """FCC control: the fix must leave the FCC primitive cell bit-for-bit
    unchanged (c_recip*a_max == 0.5 -> prefactor == pi), so MgO keeps matching
    the trusted matdyn.x reference exactly."""
    f = _freqs_cm(mgo_second, MGO_Q[qi], is_unfolding=True)
    np.testing.assert_allclose(f, np.sort(MGO_MATDYN[qi]), atol=0.5)
