"""
Regression test for the non-analytic-correction (NAC) Ewald-parameter (Lambda)
consistency between ``HarmonicWithQ.nac_dynmat`` (frequencies) and
``HarmonicWithQ.nac_derivatives`` (group velocities).

Background
----------
The NAC group-velocity operator ``nac_derivatives`` must be the exact q-gradient
of the NAC dynamical matrix ``nac_dynmat`` that is added to the frequencies. Both
evaluate a reciprocal-only Ewald dipole sum, whose value depends on the Ewald
splitting parameter ``Lambda`` (it sets the Gaussian decay ``exp(-geg/(4*Lambda))``,
the reciprocal cutoff ``geg/(4*Lambda) < gmax`` and the reciprocal-grid extent).
Historically the two routines used *different* ``Lambda``:

* ``nac_dynmat``      : normalises its reciprocal grid by ``c_recip = |inv(cell)[0,0]|``
                       and uses ``Lambda = 1``  (validated: MgO matches matdyn.x to 0.00);
* ``nac_derivatives`` : ``Lambda = (2*pi*Bohr/|cell[0,:]|)**2``.

These give the same physical Gaussian width only when
``|cell[0,:]| * |inv(cell)[0,0]| == 1`` (true for a cell whose first lattice
vector is axis-aligned, e.g. wurtzite). For a primitive cell whose first lattice
vector is *not* axis-aligned (e.g. an FCC primitive cell such as MgO) they differ,
so ``nac_derivatives`` is NOT the gradient of ``nac_dynmat`` and the NAC group
velocities are wrong (~8% for MgO).

This is a REFERENCE-FREE test: it checks the internal identity
``nac_derivatives == 1j * d/dq[nac_dynmat]`` by finite differences, against no
external data. It FAILS on the unpatched code (MgO ~8%) and PASSES once the two
routines share one Ewald parameter (``Lambda`` derived from ``c_recip`` in both).

The FCC primitive cell (MgO) is the discriminating fixture; it ships with an
in-file NAC block so no Born injection is needed.
"""

from pathlib import Path

import numpy as np
import pytest

from kaldo.forceconstants import ForceConstants
from kaldo.observables.harmonic_with_q import HarmonicWithQ

# central-difference step in cartesian q (1/Angstrom); small enough that the
# O(delta^2) truncation error sits well below the ~8% defect, large enough to
# stay above the complex64 accumulation floor of nac_dynmat.
FD_DELTA = 1e-3
# gradient-consistency tolerance (%). unpatched MgO ~8.3%; patched ~0.01%.
TOL_PCT = 1.0


def _best_fit_reldiff(analytic, fd):
    """Relative residual of ``analytic`` against ``fd`` after the best-fit complex
    scale c (= <fd, analytic> / <fd, fd>). For a true gradient c == 1j and the
    residual is the finite-difference floor."""
    a, b = analytic.ravel(), fd.ravel()
    c = np.vdot(b, a) / np.vdot(b, b)
    resid = np.abs(analytic - c * fd).max() / (np.abs(analytic).max() + 1e-30)
    return float(resid), c


def _grad_consistency(second, cell, q=(0.04, 0.02, 0.0)):
    q = np.array(q, float)
    hq = HarmonicWithQ(q_point=q, second=second, storage="memory")

    def nac_dyn(qf):
        return np.array(hq.nac_dynmat(qpoint=np.array(qf, float), Lambda=None))

    worst = 0.0
    scale = None
    for a in range(3):
        # step cartesian q along axis a by FD_DELTA:  dq_frac = (delta/2pi) * cell[:, a]
        dq = (FD_DELTA / (2 * np.pi)) * cell[:, a]
        fd = (nac_dyn(q + dq) - nac_dyn(q - dq)) / (2 * FD_DELTA)
        analytic = np.array(hq.nac_derivatives(direction=a, Lambda=None))
        resid, c = _best_fit_reldiff(analytic, fd)
        if resid > worst:
            worst, scale = resid, c
    return 100.0 * worst, scale


@pytest.fixture(scope="module")
def mgo_second():
    src = Path(__file__).resolve().parent / "mgo"
    fc = ForceConstants.from_folder(
        folder=str(src), supercell=[5, 5, 5], only_second=True, format="qe-d3q"
    )
    assert "dielectric" in fc.second.atoms.info, "MgO NAC block was not read"
    return fc.second


def test_nac_velocity_operator_is_gradient_of_dynmat_fcc(mgo_second):
    """FCC primitive (MgO): nac_derivatives must equal 1j * d/dq[nac_dynmat].

    Fails on the unpatched code (the two routines use different Ewald Lambda ->
    ~8% mismatch); passes once they share one Ewald parameter."""
    cell = np.array(mgo_second.atoms.cell)
    # the defect is triggered precisely when this product != 1
    prod = float(np.linalg.norm(cell[0, :]) * abs(np.round(np.linalg.inv(cell), 12)[0, 0]))
    assert abs(prod - 1.0) > 0.05, (
        "MgO's first lattice vector is unexpectedly axis-aligned; this fixture "
        "would not exercise the Lambda defect.")
    reldiff_pct, scale = _grad_consistency(mgo_second, cell)
    assert reldiff_pct < TOL_PCT, (
        f"nac_derivatives is not the q-gradient of nac_dynmat: "
        f"residual {reldiff_pct:.2f}% (best-fit scale {scale:.3f}). The two NAC "
        f"routines use inconsistent Ewald Lambda.")


def test_nac_gradient_scale_is_imaginary_unit(mgo_second):
    """The best-fit scale between nac_derivatives and the finite-difference of
    nac_dynmat must be ~1j (a genuine gradient, correct units), not merely
    proportional."""
    cell = np.array(mgo_second.atoms.cell)
    _, scale = _grad_consistency(mgo_second, cell)
    assert abs(scale - 1j) < 0.05, f"best-fit scale {scale:.3f} != 1j"
