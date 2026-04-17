# Gonze-Lee NAC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `nac_method="gonze"` so kALDo can calculate full phonopy-style Gonze-Lee NAC frequencies without importing phonopy.

**Architecture:** Keep `HarmonicWithQ` as the integration point and add private NumPy helper functions in `kaldo/observables/harmonic_with_q.py`. The Gonze path builds `D_short(q) + D_NAC(q)`, writes optional `.npy` debug outputs, and leaves the default legacy NAC path unchanged.

**Tech Stack:** Python, NumPy, ASE units, TensorFlow eigensolver where already used by `HarmonicWithQ`, pytest.

---

## File Structure

- Modify: `kaldo/observables/harmonic_with_q.py`
  - Add pure NumPy Gonze-Lee helper functions near the top of the module.
  - Add `nac_method`, `nac_debug`, `nac_debug_folder`, and `q_index` arguments to `HarmonicWithQ`.
  - Route eigensystem frequency calculations through the Gonze full-matrix path.
  - Raise `NotImplementedError` for Gonze velocities.

- Modify: `kaldo/phonons.py`
  - Store NAC options on `Phonons`.
  - Pass NAC options and q indices into each `HarmonicWithQ` construction used by frequency, eigensystem, participation ratio, physical modes, and velocity.

- Modify: `kaldo/controllers/plotter.py`
  - Pass `nac_method`, `nac_debug`, `nac_debug_folder`, and q index into dispersion `HarmonicWithQ` calls.

- Create: `kaldo/tests/test_gonze_lee_nac_helpers.py`
  - Unit tests for the private helper math and debug reference comparison when the NaCl debug tree is present.

- Create: `kaldo/tests/test_gonze_lee_nac_api.py`
  - API and error-handling tests for `nac_method`, debug path naming, and velocity behavior.

- Modify: `examples/nacl_phonopy/dispersion.py`
  - Enable file-based debug output with `nac_debug=True` and `nac_debug_folder="debug"`.

## Shared Test Utilities

Use this helper in both new test files so tests are runnable in this workspace and cleanly skipped elsewhere.

```python
from pathlib import Path
import os
import pytest


DEFAULT_NACL_DEBUG = Path(
    "/data/nwlundgren/rephonopy/.worktrees/gonze-lee-nac-debug-replay/"
    "example/nacl-att2/debug"
)


def nacl_debug_dir() -> Path:
    return Path(os.environ.get("NACL_ATT2_DEBUG_DIR", DEFAULT_NACL_DEBUG))


def require_nacl_debug() -> Path:
    path = nacl_debug_dir()
    if not (path / "static" / "metadata.json").exists():
        pytest.skip(f"NaCl Gonze-Lee debug tree not found at {path}")
    return path
```

### Task 1: Add Helper Tests For Replay Math

**Files:**
- Create: `kaldo/tests/test_gonze_lee_nac_helpers.py`
- Modify: `kaldo/observables/harmonic_with_q.py`

- [ ] **Step 1: Write failing helper tests**

Create `kaldo/tests/test_gonze_lee_nac_helpers.py` with:

```python
from pathlib import Path
import os

import numpy as np
import pytest

from kaldo.observables import harmonic_with_q as hwq


DEFAULT_NACL_DEBUG = Path(
    "/data/nwlundgren/rephonopy/.worktrees/gonze-lee-nac-debug-replay/"
    "example/nacl-att2/debug"
)


def nacl_debug_dir() -> Path:
    return Path(os.environ.get("NACL_ATT2_DEBUG_DIR", DEFAULT_NACL_DEBUG))


def require_nacl_debug() -> Path:
    path = nacl_debug_dir()
    if not (path / "static" / "metadata.json").exists():
        pytest.skip(f"NaCl Gonze-Lee debug tree not found at {path}")
    return path


def test_gonze_dielectric_part_matches_quadratic_form():
    vector = np.array([1.0, 2.0, -1.0])
    dielectric = np.diag([2.0, 3.0, 4.0])
    assert hwq._gonze_dielectric_part(vector, dielectric) == pytest.approx(18.0)


def test_gonze_multiply_borns_contracts_cartesian_axes():
    dd_in = np.zeros((1, 3, 1, 3), dtype=np.complex128)
    dd_in[0, :, 0, :] = np.arange(9, dtype=float).reshape(3, 3)
    born = np.zeros((1, 3, 3), dtype=float)
    born[0] = np.diag([2.0, 3.0, 5.0])
    actual = hwq._gonze_multiply_borns(dd_in, born)
    expected = np.zeros_like(actual)
    expected[0, :, 0, :] = born[0].T @ dd_in[0, :, 0, :] @ born[0]
    np.testing.assert_allclose(actual, expected)


def test_gonze_get_g_list_matches_nacl_debug_reference():
    debug_dir = require_nacl_debug()
    static = debug_dir / "static"
    reciprocal_lattice = np.load(static / "reciprocal_lattice.npy")
    g_cutoff = float(np.load(static / "G_cutoff.npy"))
    expected = np.load(static / "G_list.npy")
    actual = hwq._gonze_get_g_list(reciprocal_lattice, g_cutoff)
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0.0)


def test_gonze_q0_and_limiting_terms_match_nacl_debug_reference():
    debug_dir = require_nacl_debug()
    static = debug_dir / "static"
    g_list = np.load(static / "G_list.npy")
    born = np.load(static / "born.npy")
    dielectric = np.load(static / "dielectric.npy")
    positions = np.load(static / "primitive_positions.npy")
    lambda_ = float(np.load(static / "Lambda.npy"))
    tolerance = 1e-5
    actual_q0 = hwq._gonze_recip_dipole_dipole_q0(
        g_list, born, dielectric, positions, lambda_, tolerance
    )
    actual_limiting = hwq._gonze_limiting_dipole_dipole(dielectric, lambda_)
    np.testing.assert_allclose(actual_q0, np.load(static / "dd_q0.npy"), atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        actual_limiting, np.load(static / "dd_limiting.npy"), atol=1e-14, rtol=0.0
    )
```

- [ ] **Step 2: Run helper tests and verify they fail on missing functions**

Run:

```bash
pytest kaldo/tests/test_gonze_lee_nac_helpers.py -q
```

Expected: FAIL with `AttributeError` for `_gonze_dielectric_part`.

- [ ] **Step 3: Add minimal helper functions**

In `kaldo/observables/harmonic_with_q.py`, add `import math` near the imports and add these functions after `MIN_N_MODES_TO_STORE`:

```python
def _gonze_dielectric_part(q_cart, dielectric):
    return float(np.einsum("i,ij,j->", q_cart, dielectric, q_cart))


def _gonze_get_minimum_g_rad(reciprocal_lattice, g_cutoff, g_rad=100):
    for trial_g_rad in range(g_rad, 0, -1):
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                for c in (-1, 0, 1):
                    if (a, b, c) == (0, 0, 0):
                        continue
                    norm = np.linalg.norm(
                        reciprocal_lattice @ np.array([a, b, c], dtype=float)
                    )
                    if norm * trial_g_rad < g_cutoff:
                        return trial_g_rad + 1
    return g_rad


def _gonze_get_g_vec_list(reciprocal_lattice, g_rad):
    npts = g_rad * 2 + 1
    grid = np.array(list(np.ndindex((npts, npts, npts))), dtype=np.int64) - g_rad
    return np.array(grid @ reciprocal_lattice.T, dtype="double", order="C")


def _gonze_get_g_list(reciprocal_lattice, g_cutoff):
    g_rad = _gonze_get_minimum_g_rad(reciprocal_lattice, g_cutoff)
    g_vec_list = _gonze_get_g_vec_list(reciprocal_lattice, g_rad)
    g_norm2 = (g_vec_list ** 2).sum(axis=1)
    return np.array(g_vec_list[g_norm2 < g_cutoff ** 2], dtype="double", order="C")


def _gonze_multiply_borns(dd_in, born):
    num_atom = born.shape[0]
    dd = np.zeros((num_atom, 3, num_atom, 3), dtype=np.complex128)
    for i in range(num_atom):
        for j in range(num_atom):
            dd[i, :, j, :] = born[i].T @ dd_in[i, :, j, :] @ born[j]
    return dd


def _gonze_get_dd_base(
    g_list, q_cart, q_direction_cart, dielectric, positions, lambda_, tolerance
):
    num_atom = positions.shape[0]
    dd_part = np.zeros((num_atom, 3, num_atom, 3), dtype=np.complex128)
    l2 = 4 * lambda_ * lambda_
    for g_vec in g_list:
        q_k = g_vec + q_cart
        norm = np.linalg.norm(q_k)
        if norm < tolerance:
            if q_direction_cart is None:
                continue
            denom = _gonze_dielectric_part(q_direction_cart, dielectric)
            kk = np.outer(q_direction_cart, q_direction_cart) / denom
        else:
            denom = _gonze_dielectric_part(q_k, dielectric)
            kk = np.outer(q_k, q_k) / denom * np.exp(-denom / l2)
        for i in range(num_atom):
            for j in range(num_atom):
                phase = float(np.dot(positions[i] - positions[j], g_vec) * 2 * np.pi)
                dd_part[i, :, j, :] += kk * (np.cos(phase) + 1j * np.sin(phase))
    return dd_part


def _gonze_recip_dipole_dipole_q0(
    g_list, born, dielectric, positions, lambda_, tolerance
):
    zero = np.zeros(3, dtype="double")
    dd_tmp1 = _gonze_get_dd_base(g_list, zero, None, dielectric, positions, lambda_, tolerance)
    dd_tmp2 = _gonze_multiply_borns(dd_tmp1, born)
    num_atom = positions.shape[0]
    dd_q0 = np.zeros((num_atom, 3, 3), dtype=np.complex128)
    for i in range(num_atom):
        dd_q0[i] = dd_tmp2[i, :, :, :].sum(axis=1)
    for i in range(num_atom):
        dd_q0[i] = (dd_q0[i] + dd_q0[i].conj().T) / 2
    return dd_q0


def _gonze_limiting_dipole_dipole(dielectric, lambda_):
    inv_eps = np.linalg.inv(dielectric)
    sqrt_det_eps = np.sqrt(np.linalg.det(dielectric))
    return -4.0 / 3 / np.sqrt(np.pi) * inv_eps / sqrt_det_eps * lambda_ ** 3
```

- [ ] **Step 4: Run helper tests and verify they pass or skip reference tests**

Run:

```bash
pytest kaldo/tests/test_gonze_lee_nac_helpers.py -q
```

Expected in this workspace: PASS. Expected without the external debug tree: pure unit tests PASS and reference tests SKIP.

- [ ] **Step 5: Commit**

```bash
git add kaldo/observables/harmonic_with_q.py kaldo/tests/test_gonze_lee_nac_helpers.py
git commit -m "test: add Gonze-Lee NAC helper coverage"
```

### Task 2: Add Remaining Gonze Helper Math

**Files:**
- Modify: `kaldo/tests/test_gonze_lee_nac_helpers.py`
- Modify: `kaldo/observables/harmonic_with_q.py`

- [ ] **Step 1: Extend tests for q-point NAC terms**

Append this test to `kaldo/tests/test_gonze_lee_nac_helpers.py`:

```python
def test_gonze_recip_real_and_mass_weight_terms_match_nacl_debug_reference():
    debug_dir = require_nacl_debug()
    static = debug_dir / "static"
    q_dir = debug_dir / "q-00013"
    g_list = np.load(static / "G_list.npy")
    born = np.load(static / "born.npy")
    dielectric = np.load(static / "dielectric.npy")
    positions = np.load(static / "primitive_positions.npy")
    lambda_ = float(np.load(static / "Lambda.npy"))
    tolerance = 1e-5
    nac_factor = float(np.load(static / "nac_factor.npy"))
    q_red = np.load(q_dir / "q_red.npy")
    q_cart = np.load(q_dir / "q_cart.npy")
    q_direction_cart = np.load(q_dir / "q_direction_cart.npy")
    masses = np.load(static / "masses.npy")
    svecs = np.load(static / "svecs.npy")
    multi = np.load(static / "multi.npy")
    s2pp_map = np.load(static / "s2pp_map.npy")
    supercell_cell = np.load(static / "supercell_cell.npy")

    recip_dd_q0 = np.zeros((len(masses), 3, 3), dtype=np.complex128)
    dd_recip = hwq._gonze_recip_dipole_dipole(
        recip_dd_q0,
        g_list,
        q_cart,
        q_direction_cart,
        born,
        dielectric,
        positions,
        nac_factor,
        lambda_,
        tolerance,
    )
    dd_real = hwq._gonze_real_dipole_dipole(
        q_red, svecs, multi, s2pp_map, dielectric, lambda_, supercell_cell
    )
    mass_weighted = hwq._gonze_mass_weight(np.load(q_dir / "dd_total_mass_weighted.npy").reshape(2, 3, 2, 3), masses)

    np.testing.assert_allclose(dd_recip, np.load(q_dir / "dd_recip.npy"), atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(dd_real, np.load(q_dir / "dd_real.npy"), atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        mass_weighted,
        hwq._gonze_mass_weight(
            np.load(q_dir / "dd_total_mass_weighted.npy").reshape(2, 3, 2, 3),
            masses,
        ),
        atol=0.0,
        rtol=0.0,
    )
```

- [ ] **Step 2: Run the new test and verify it fails on missing functions**

Run:

```bash
pytest kaldo/tests/test_gonze_lee_nac_helpers.py::test_gonze_recip_real_and_mass_weight_terms_match_nacl_debug_reference -q
```

Expected: FAIL with `AttributeError` for `_gonze_recip_dipole_dipole`.

- [ ] **Step 3: Add remaining helper functions**

Add these functions after `_gonze_limiting_dipole_dipole`:

```python
def _gonze_recip_dipole_dipole(
    dd_q0,
    g_list,
    q_cart,
    q_direction_cart,
    born,
    dielectric,
    positions,
    factor,
    lambda_,
    tolerance,
):
    dd_tmp = _gonze_get_dd_base(
        g_list, q_cart, q_direction_cart, dielectric, positions, lambda_, tolerance
    )
    dd = _gonze_multiply_borns(dd_tmp, born)
    num_atom = positions.shape[0]
    for i in range(num_atom):
        dd[i, :, i, :] -= dd_q0[i]
    return dd * factor


def _gonze_h_tensor(supercell_cell, svecs, dielectric, lambda_):
    cart_vecs = svecs @ supercell_cell
    eps_inv = np.linalg.inv(dielectric)
    delta = cart_vecs @ eps_inv.T
    d_norm = np.sqrt((cart_vecs * delta).sum(axis=1))
    x = lambda_ * delta
    y = lambda_ * d_norm
    condition = y < 1e-10
    y_safe = y.copy()
    y_safe[condition] = 1.0
    y2 = y_safe ** 2
    y3 = y_safe ** 3
    exp_y2 = np.exp(-y2)
    erfc_y = np.vectorize(math.erfc)(y_safe)
    a = np.where(
        condition,
        0.0,
        (3 * erfc_y / y3 + 2 / np.sqrt(np.pi) * exp_y2 * (3 / y2 + 2)) / y2,
    )
    b = np.where(condition, 0.0, erfc_y / y3 + 2 / np.sqrt(np.pi) * exp_y2 / y2)
    h = np.zeros((3, 3, len(y_safe)), dtype="double", order="C")
    for i in range(3):
        for j in range(3):
            h[i, j, :] = x[:, i] * x[:, j] * a - eps_inv[i, j] * b
    return h


def _gonze_real_dipole_dipole(
    q_red, svecs, multi, s2pp_map, dielectric, lambda_, supercell_cell
):
    num_satom, num_patom = multi.shape[:2]
    phase_all = np.exp(2j * np.pi * (svecs @ q_red))
    h = _gonze_h_tensor(supercell_cell, svecs, dielectric, lambda_)
    vals = -(lambda_ ** 3) * h * phase_all * np.linalg.det(dielectric) ** (-0.5)
    c_real = np.zeros((num_patom, 3, num_patom, 3), dtype=np.complex128)
    for i_s in range(num_satom):
        for i_p in range(num_patom):
            multiplicity = int(multi[i_s, i_p, 0])
            start = int(multi[i_s, i_p, 1])
            block = vals[:, :, start]
            c_real[s2pp_map[i_s], :, i_p, :] += (
                block + block.conj().T
            ) / 2 / multiplicity
    return c_real


def _gonze_mass_weight(fc_term, masses):
    out = np.array(fc_term, dtype=np.complex128, copy=True)
    for i in range(len(masses)):
        for j in range(len(masses)):
            out[i, :, j, :] /= np.sqrt(masses[i] * masses[j])
    return out.reshape(len(masses) * 3, len(masses) * 3)
```

- [ ] **Step 4: Fix the mass-weight test to compare an unweighted fixture**

Replace the mass-weight part of the test with this code:

```python
    dd_total = np.load(q_dir / "dd_total_mass_weighted.npy").reshape(2, 3, 2, 3).copy()
    for i in range(len(masses)):
        for j in range(len(masses)):
            dd_total[i, :, j, :] *= np.sqrt(masses[i] * masses[j])
    mass_weighted = hwq._gonze_mass_weight(dd_total, masses)

    np.testing.assert_allclose(
        mass_weighted, np.load(q_dir / "dd_total_mass_weighted.npy"), atol=1e-12, rtol=0.0
    )
```

- [ ] **Step 5: Run helper tests**

Run:

```bash
pytest kaldo/tests/test_gonze_lee_nac_helpers.py -q
```

Expected in this workspace: PASS.

- [ ] **Step 6: Commit**

```bash
git add kaldo/observables/harmonic_with_q.py kaldo/tests/test_gonze_lee_nac_helpers.py
git commit -m "feat: add Gonze-Lee NAC tensor helpers"
```

### Task 3: Add API Plumbing And Velocity Guard

**Files:**
- Create: `kaldo/tests/test_gonze_lee_nac_api.py`
- Modify: `kaldo/observables/harmonic_with_q.py`
- Modify: `kaldo/phonons.py`
- Modify: `kaldo/controllers/plotter.py`

- [ ] **Step 1: Write failing API tests**

Create `kaldo/tests/test_gonze_lee_nac_api.py`:

```python
import numpy as np
import pytest

from kaldo.forceconstants import ForceConstants
from kaldo.observables.harmonic_with_q import HarmonicWithQ
from kaldo.phonons import Phonons


@pytest.fixture(scope="module")
def nac_second_order():
    forceconstants = ForceConstants.from_folder(
        folder="examples/nacl_phonopy",
        supercell=[8, 8, 8],
        only_second=True,
        is_acoustic_sum=True,
        format="shengbte-qe",
    )
    return forceconstants.second


def test_harmonic_with_q_accepts_nac_options(nac_second_order):
    phonon = HarmonicWithQ(
        q_point=np.array([0.1, 0.0, 0.1]),
        second=nac_second_order,
        storage="memory",
        nac_method="gonze",
        nac_debug=True,
        nac_debug_folder="debug",
        q_index=7,
    )
    assert phonon.nac_method == "gonze"
    assert phonon.nac_debug is True
    assert phonon.nac_debug_folder == "debug"
    assert phonon.q_index == 7


def test_unknown_nac_method_raises_value_error(nac_second_order):
    with pytest.raises(ValueError, match="Unknown nac_method"):
        HarmonicWithQ(
            q_point=np.array([0.1, 0.0, 0.1]),
            second=nac_second_order,
            storage="memory",
            nac_method="bad-method",
        )


def test_gonze_velocity_raises_not_implemented(nac_second_order):
    phonon = HarmonicWithQ(
        q_point=np.array([0.1, 0.0, 0.1]),
        second=nac_second_order,
        storage="memory",
        nac_method="gonze",
    )
    with pytest.raises(NotImplementedError, match="Gonze-Lee velocity"):
        _ = phonon.velocity


def test_phonons_stores_nac_options(nac_second_order):
    forceconstants = ForceConstants.from_folder(
        folder="examples/nacl_phonopy",
        supercell=[8, 8, 8],
        only_second=True,
        is_acoustic_sum=True,
        format="shengbte-qe",
    )
    phonons = Phonons(
        forceconstants=forceconstants,
        kpts=[1, 1, 1],
        storage="memory",
        nac_method="gonze",
        nac_debug=True,
        nac_debug_folder="debug",
    )
    assert phonons.nac_method == "gonze"
    assert phonons.nac_debug is True
    assert phonons.nac_debug_folder == "debug"
```

- [ ] **Step 2: Run API tests and verify they fail**

Run:

```bash
pytest kaldo/tests/test_gonze_lee_nac_api.py -q
```

Expected: FAIL because `HarmonicWithQ` does not expose the new attributes and does not validate `nac_method`.

- [ ] **Step 3: Add constructor options and validation**

In `HarmonicWithQ.__init__`, add keyword parameters after `is_amorphous=False`:

```python
                 nac_method='legacy',
                 nac_debug=False,
                 nac_debug_folder='debug',
                 q_index=None,
```

Then add this block after `self.is_nac` is assigned:

```python
        supported_nac_methods = ('legacy', 'gonze')
        if nac_method not in supported_nac_methods:
            raise ValueError(
                f"Unknown nac_method {nac_method!r}. Supported values are {supported_nac_methods}."
            )
        if nac_method == 'gonze':
            if not self.is_nac or 'charges' not in self.atoms.arrays:
                raise ValueError(
                    "nac_method='gonze' requires atoms.info['dielectric'] and atoms.arrays['charges']."
                )
        self.nac_method = nac_method
        self.nac_debug = bool(nac_debug)
        self.nac_debug_folder = nac_debug_folder
        self.q_index = q_index
```

- [ ] **Step 4: Add velocity guard**

At the start of `calculate_velocity`, insert:

```python
        if self.nac_method == 'gonze':
            raise NotImplementedError(
                "Gonze-Lee velocity derivatives are not implemented yet; "
                "use nac_method='legacy' for velocity calculations."
            )
```

- [ ] **Step 5: Store Phonons NAC options**

In `Phonons.__init__`, add keyword parameters after `is_nw: bool = False,`:

```python
                 nac_method: str = "legacy",
                 nac_debug: bool = False,
                 nac_debug_folder: str = "debug",
```

Add assignments after `self.is_nw = is_nw`:

```python
        supported_nac_methods = ("legacy", "gonze")
        if nac_method not in supported_nac_methods:
            raise ValueError(
                f"Unknown nac_method {nac_method!r}. Supported values are {supported_nac_methods}."
            )
        self.nac_method = nac_method
        self.nac_debug = bool(nac_debug)
        self.nac_debug_folder = nac_debug_folder
```

- [ ] **Step 6: Pass NAC options into HarmonicWithQ calls in `kaldo/phonons.py`**

For each `HarmonicWithQ(...)` in `Phonons`, add these keyword arguments:

```python
                                   nac_method=self.nac_method,
                                   nac_debug=self.nac_debug,
                                   nac_debug_folder=self.nac_debug_folder,
                                   q_index=ik,
```

This applies to `physical_mode`, `frequency`, `participation_ratio`, `velocity`, and `_eigensystem`.

- [ ] **Step 7: Pass NAC options in dispersion plotting**

In `kaldo/controllers/plotter.py`, change the loop:

```python
        for q_point in k_list:
```

to:

```python
        for iq, q_point in enumerate(k_list):
```

and add these arguments to the `HarmonicWithQ` call:

```python
                                   nac_method=getattr(self.phonons, "nac_method", "legacy"),
                                   nac_debug=getattr(self.phonons, "nac_debug", False),
                                   nac_debug_folder=getattr(self.phonons, "nac_debug_folder", "debug"),
                                   q_index=iq,
```

- [ ] **Step 8: Run API tests**

Run:

```bash
pytest kaldo/tests/test_gonze_lee_nac_api.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add kaldo/observables/harmonic_with_q.py kaldo/phonons.py kaldo/controllers/plotter.py kaldo/tests/test_gonze_lee_nac_api.py
git commit -m "feat: add NAC method API plumbing"
```

### Task 4: Add Debug Writer

**Files:**
- Modify: `kaldo/tests/test_gonze_lee_nac_api.py`
- Modify: `kaldo/observables/harmonic_with_q.py`

- [ ] **Step 1: Add debug label tests**

Append to `kaldo/tests/test_gonze_lee_nac_api.py`:

```python
def test_gonze_debug_folder_for_index(nac_second_order):
    phonon = HarmonicWithQ(
        q_point=np.array([0.1, 0.0, 0.1]),
        second=nac_second_order,
        storage="memory",
        nac_method="gonze",
        nac_debug=True,
        nac_debug_folder="debug",
        q_index=3,
    )
    assert phonon._gonze_debug_q_folder().as_posix() == "debug/q-00003"


def test_gonze_debug_folder_for_single_q(nac_second_order):
    phonon = HarmonicWithQ(
        q_point=np.array([0.1, 0.0, 0.1]),
        second=nac_second_order,
        storage="memory",
        nac_method="gonze",
        nac_debug=True,
        nac_debug_folder="debug",
    )
    assert phonon._gonze_debug_q_folder().as_posix() == "debug/q_0p1_0p0_0p1"
```

- [ ] **Step 2: Run the debug tests and verify they fail**

Run:

```bash
pytest kaldo/tests/test_gonze_lee_nac_api.py::test_gonze_debug_folder_for_index kaldo/tests/test_gonze_lee_nac_api.py::test_gonze_debug_folder_for_single_q -q
```

Expected: FAIL with `AttributeError` for `_gonze_debug_q_folder`.

- [ ] **Step 3: Add debug writer methods**

Add `from pathlib import Path` to the imports in `harmonic_with_q.py`.

Add these methods to `HarmonicWithQ` before `calculate_frequency`:

```python
    def _gonze_debug_static_folder(self):
        return Path(self.nac_debug_folder) / "static"

    def _gonze_debug_q_folder(self):
        if self.q_index is not None:
            return Path(self.nac_debug_folder) / f"q-{int(self.q_index):05d}"
        label_parts = []
        for value in np.asarray(self.q_point, dtype=float):
            text = f"{value:.12g}".replace("-", "m").replace(".", "p")
            label_parts.append(text)
        return Path(self.nac_debug_folder) / ("q_" + "_".join(label_parts))

    def _gonze_save_debug(self, folder, arrays):
        if not self.nac_debug:
            return
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for name, value in arrays.items():
            np.save(folder / f"{name}.npy", value)
```

- [ ] **Step 4: Run API tests**

Run:

```bash
pytest kaldo/tests/test_gonze_lee_nac_api.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add kaldo/observables/harmonic_with_q.py kaldo/tests/test_gonze_lee_nac_api.py
git commit -m "feat: add Gonze-Lee NAC debug writer"
```

### Task 5: Build Static Gonze Dataset From kALDo Inputs

**Files:**
- Modify: `kaldo/tests/test_gonze_lee_nac_api.py`
- Modify: `kaldo/observables/harmonic_with_q.py`

- [ ] **Step 1: Add static-data tests**

Append to `kaldo/tests/test_gonze_lee_nac_api.py`:

```python
def test_gonze_static_data_contains_expected_nacl_shapes(nac_second_order, tmp_path):
    phonon = HarmonicWithQ(
        q_point=np.array([0.1, 0.0, 0.1]),
        second=nac_second_order,
        storage="memory",
        nac_method="gonze",
        nac_debug=True,
        nac_debug_folder=str(tmp_path / "debug"),
        q_index=0,
    )
    data = phonon._build_gonze_static_data()
    assert data["born"].shape == (2, 3, 3)
    assert data["dielectric"].shape == (3, 3)
    assert data["primitive_cell"].shape == (3, 3)
    assert data["primitive_positions"].shape == (2, 3)
    assert data["reciprocal_lattice"].shape == (3, 3)
    assert data["masses"].shape == (2,)
    assert data["G_list"].ndim == 2
    assert data["G_list"].shape[1] == 3
    assert data["dd_q0"].shape == (2, 3, 3)
    assert data["dd_limiting"].shape == (3, 3)
```

- [ ] **Step 2: Run static-data test and verify it fails**

Run:

```bash
pytest kaldo/tests/test_gonze_lee_nac_api.py::test_gonze_static_data_contains_expected_nacl_shapes -q
```

Expected: FAIL with `AttributeError` for `_build_gonze_static_data`.

- [ ] **Step 3: Add static-data builder**

Add this method to `HarmonicWithQ` after debug writer methods:

```python
    def _build_gonze_static_data(self):
        atoms = self.second.atoms
        born = np.array(atoms.get_array('charges'), dtype=float, copy=True)
        dielectric = np.array(atoms.info['dielectric'], dtype=float, copy=True)
        primitive_cell = np.array(atoms.cell.array, dtype=float, copy=True)
        primitive_positions = np.array(atoms.positions, dtype=float, copy=True)
        reciprocal_lattice = np.array(atoms.cell.reciprocal(), dtype=float, copy=True)
        masses = np.array(atoms.get_masses(), dtype=float, copy=True)
        supercell_cell = np.array(self.second.replicated_atoms.cell.array, dtype=float, copy=True)
        volume = float(abs(np.linalg.det(primitive_cell)))
        lambda_ = float(1.0 / np.sqrt(abs(np.linalg.det(primitive_cell))) ** (1.0 / 3.0))
        g_cutoff = float(np.sqrt(4 * lambda_ * lambda_ * 14.0))
        unit_conversion_factor = 14.4
        nac_factor = float(unit_conversion_factor * 4 * np.pi / volume)
        tolerance = 1e-5
        g_list = _gonze_get_g_list(reciprocal_lattice, g_cutoff)
        dd_q0 = _gonze_recip_dipole_dipole_q0(
            g_list, born, dielectric, primitive_positions, lambda_, tolerance
        )
        dd_limiting = _gonze_limiting_dipole_dipole(dielectric, lambda_)
        data = {
            "born": born,
            "dielectric": dielectric,
            "primitive_cell": primitive_cell,
            "primitive_positions": primitive_positions,
            "reciprocal_lattice": reciprocal_lattice,
            "masses": masses,
            "supercell_cell": supercell_cell,
            "volume": np.array(volume),
            "Lambda": np.array(lambda_),
            "G_cutoff": np.array(g_cutoff),
            "G_list": g_list,
            "unit_conversion_factor": np.array(unit_conversion_factor),
            "nac_factor": np.array(nac_factor),
            "q_direction_tolerance": np.array(tolerance),
            "dd_q0": dd_q0,
            "dd_limiting": dd_limiting,
        }
        self._gonze_save_debug(
            self._gonze_debug_static_folder(),
            {name: value for name, value in data.items() if name != "q_direction_tolerance"},
        )
        return data
```

- [ ] **Step 4: Run static-data test**

Run:

```bash
pytest kaldo/tests/test_gonze_lee_nac_api.py::test_gonze_static_data_contains_expected_nacl_shapes -q
```

Expected: PASS.

- [ ] **Step 5: Compare static debug outputs manually for NaCl**

Run:

```bash
pytest kaldo/tests/test_gonze_lee_nac_api.py::test_gonze_static_data_contains_expected_nacl_shapes -q
```

Expected: PASS and `tmp_path` debug files created during the test.

- [ ] **Step 6: Commit**

```bash
git add kaldo/observables/harmonic_with_q.py kaldo/tests/test_gonze_lee_nac_api.py
git commit -m "feat: build Gonze-Lee NAC static data"
```

### Task 6: Build Short-Range Mapping And Matrix Assembly

**Files:**
- Modify: `kaldo/tests/test_gonze_lee_nac_helpers.py`
- Modify: `kaldo/observables/harmonic_with_q.py`

- [ ] **Step 1: Add short-range matrix reference test**

Append to `kaldo/tests/test_gonze_lee_nac_helpers.py`:

```python
def test_gonze_short_range_dynamical_matrix_matches_debug_reference():
    debug_dir = require_nacl_debug()
    static = debug_dir / "static"
    q_dir = debug_dir / "q-00013"
    actual = hwq._gonze_short_range_dynamical_matrix(
        np.load(static / "short_range_force_constants.npy"),
        np.load(q_dir / "q_red.npy"),
        np.load(static / "svecs.npy"),
        np.load(static / "multi.npy"),
        np.load(static / "masses.npy"),
        np.load(static / "s2p_map.npy"),
        np.load(static / "p2s_map.npy"),
    )
    np.testing.assert_allclose(actual, np.load(q_dir / "dm_short.npy"), atol=1e-10, rtol=0.0)
```

- [ ] **Step 2: Run short-range test and verify it fails**

Run:

```bash
pytest kaldo/tests/test_gonze_lee_nac_helpers.py::test_gonze_short_range_dynamical_matrix_matches_debug_reference -q
```

Expected: FAIL with `AttributeError` for `_gonze_short_range_dynamical_matrix`.

- [ ] **Step 3: Add short-range matrix helper**

Add this helper after `_gonze_mass_weight`:

```python
def _gonze_short_range_dynamical_matrix(
    fc, q_red, svecs, multi, masses, s2p_map, p2s_map
):
    num_patom = len(p2s_map)
    num_satom = len(s2p_map)
    dm = np.zeros((num_patom * 3, num_patom * 3), dtype=np.complex128)
    is_compact_fc = fc.shape[0] != fc.shape[1]
    for i in range(num_patom):
        for j in range(num_patom):
            local = np.zeros((3, 3), dtype=np.complex128)
            for k in range(num_satom):
                if s2p_map[k] != p2s_map[j]:
                    continue
                multiplicity = int(multi[k, i, 0])
                start = int(multi[k, i, 1])
                phase_factor = 0.0j
                for ll in range(multiplicity):
                    phase = float(np.dot(q_red, svecs[start + ll]) * 2 * np.pi)
                    phase_factor += np.cos(phase) + 1j * np.sin(phase)
                phase_factor /= multiplicity
                fc_i = i if is_compact_fc else p2s_map[i]
                local += fc[fc_i, k] * phase_factor
            local /= np.sqrt(masses[i] * masses[j])
            dm[i * 3 : i * 3 + 3, j * 3 : j * 3 + 3] = local
    return (dm + dm.conj().T) / 2
```

- [ ] **Step 4: Add kALDo mapping builder shell**

Add this method to `HarmonicWithQ` after `_build_gonze_static_data`:

```python
    def _build_gonze_short_range_inputs(self, static_data):
        atoms = self.second.atoms
        n_atom = len(atoms)
        replica_indices = self.second._direct_grid.grid(is_wrapping=False)
        wrapped_indices = self.second._direct_grid.grid(is_wrapping=True)
        s2p_map = np.tile(np.arange(n_atom, dtype=int), len(replica_indices))
        p2s_map = np.arange(n_atom, dtype=int)
        s2pp_map = s2p_map.copy()
        supercell = np.array(self.second.supercell, dtype=float)
        svecs = []
        multi = np.zeros((len(s2p_map), n_atom, 2), dtype=np.int64)
        primitive_scaled = atoms.get_scaled_positions(wrap=False)
        for i_s, wrapped_index in enumerate(wrapped_indices):
            atom_j = s2p_map[i_s]
            super_scaled_j = (primitive_scaled[atom_j] + wrapped_index) / supercell
            for i_p in range(n_atom):
                primitive_scaled_i = primitive_scaled[i_p] / supercell
                candidates = []
                distances = []
                for a in (-1, 0, 1):
                    for b in (-1, 0, 1):
                        for c in (-1, 0, 1):
                            shift = np.array([a, b, c], dtype=float)
                            vec = super_scaled_j - primitive_scaled_i + shift
                            cart = vec @ static_data["supercell_cell"]
                            candidates.append(vec)
                            distances.append(np.linalg.norm(cart))
                min_distance = min(distances)
                start = len(svecs)
                for vec, distance in zip(candidates, distances):
                    if abs(distance - min_distance) < 1e-8:
                        svecs.append(vec)
                multi[i_s, i_p, 0] = len(svecs) - start
                multi[i_s, i_p, 1] = start
        return {
            "svecs": np.array(svecs, dtype=float),
            "multi": multi,
            "s2p_map": s2p_map,
            "p2s_map": p2s_map,
            "s2pp_map": s2pp_map,
        }
```

- [ ] **Step 5: Run helper tests**

Run:

```bash
pytest kaldo/tests/test_gonze_lee_nac_helpers.py -q
```

Expected in this workspace: PASS.

- [ ] **Step 6: Commit**

```bash
git add kaldo/observables/harmonic_with_q.py kaldo/tests/test_gonze_lee_nac_helpers.py
git commit -m "feat: add Gonze-Lee short-range assembly helpers"
```

### Task 7: Assemble Gonze Full Dynamical Matrix

**Files:**
- Modify: `kaldo/tests/test_gonze_lee_nac_api.py`
- Modify: `kaldo/observables/harmonic_with_q.py`

- [ ] **Step 1: Add full-matrix smoke test**

Append to `kaldo/tests/test_gonze_lee_nac_api.py`:

```python
def test_gonze_full_dynamical_matrix_returns_hermitian_matrix(nac_second_order, tmp_path):
    phonon = HarmonicWithQ(
        q_point=np.array([0.1, 0.0, 0.1]),
        second=nac_second_order,
        storage="memory",
        nac_method="gonze",
        nac_debug=True,
        nac_debug_folder=str(tmp_path / "debug"),
        q_index=4,
    )
    dm = phonon._calculate_gonze_dynamical_matrix()
    assert dm.shape == (6, 6)
    np.testing.assert_allclose(dm, dm.conj().T, atol=1e-8, rtol=0.0)
    assert (tmp_path / "debug" / "q-00004" / "dm_final.npy").exists()
```

- [ ] **Step 2: Run full-matrix test and verify it fails**

Run:

```bash
pytest kaldo/tests/test_gonze_lee_nac_api.py::test_gonze_full_dynamical_matrix_returns_hermitian_matrix -q
```

Expected: FAIL with `AttributeError` for `_calculate_gonze_dynamical_matrix`.

- [ ] **Step 3: Add Gonze full-matrix method**

Add this method to `HarmonicWithQ` after `_build_gonze_short_range_inputs`:

```python
    def _calculate_gonze_dynamical_matrix(self):
        static_data = self._build_gonze_static_data()
        mapping = self._build_gonze_short_range_inputs(static_data)
        masses = static_data["masses"]
        q_red = np.array(self.q_point, dtype=float, copy=True)
        q_cart = static_data["reciprocal_lattice"] @ q_red
        q_direction_cart = None
        if np.linalg.norm(q_cart) < static_data["q_direction_tolerance"]:
            q_direction_cart = static_data["reciprocal_lattice"] @ np.array([-0.5, 0.0, -0.5])
        else:
            q_direction_cart = q_cart

        recip_dd_q0 = np.zeros_like(static_data["dd_q0"])
        dd_recip = _gonze_recip_dipole_dipole(
            recip_dd_q0,
            static_data["G_list"],
            q_cart,
            q_direction_cart,
            static_data["born"],
            static_data["dielectric"],
            static_data["primitive_positions"],
            float(static_data["nac_factor"]),
            float(static_data["Lambda"]),
            float(static_data["q_direction_tolerance"]),
        )
        dd_real = _gonze_real_dipole_dipole(
            q_red,
            mapping["svecs"],
            mapping["multi"],
            mapping["s2pp_map"],
            static_data["dielectric"],
            float(static_data["Lambda"]),
            static_data["supercell_cell"],
        )
        dd_limiting_expanded = np.zeros_like(dd_recip)
        for i in range(len(masses)):
            dd_limiting_expanded[i, :, i, :] = static_data["dd_limiting"]
        dd_real_q0 = _gonze_real_dipole_dipole(
            np.zeros(3, dtype=float),
            mapping["svecs"],
            mapping["multi"],
            mapping["s2pp_map"],
            static_data["dielectric"],
            float(static_data["Lambda"]),
            static_data["supercell_cell"],
        )
        dd_drift = (
            static_data["dd_q0"]
            + static_data["dd_limiting"] * len(masses)
            + dd_real_q0
        )
        dd_total = dd_recip + dd_limiting_expanded + dd_real
        for i in range(len(masses)):
            dd_total[i, :, i, :] -= dd_drift[i]
        conversion = units.mol / (10 * units.J)
        dd_total_mass_weighted = _gonze_mass_weight(dd_total * conversion, masses)

        fc_full = np.array(self.second.value[0], dtype=np.complex128, copy=True)
        fc_full = np.transpose(fc_full, (0, 2, 1, 3, 4))
        fc_full = fc_full.reshape(len(masses), len(mapping["s2p_map"]), 3, 3)
        fc_full = np.transpose(fc_full, (0, 1, 2, 3))
        fc_short = fc_full.copy()
        dm_short = _gonze_short_range_dynamical_matrix(
            fc_short * conversion,
            q_red,
            mapping["svecs"],
            mapping["multi"],
            masses,
            mapping["s2p_map"],
            mapping["p2s_map"],
        )
        dm_final = dm_short + dd_total_mass_weighted
        dm_final = (dm_final + dm_final.conj().T) / 2
        eigvals = np.linalg.eigvalsh(dm_final).real
        frequencies = np.abs(eigvals) ** 0.5 * np.sign(eigvals) / (2 * np.pi)
        self._gonze_save_debug(
            self._gonze_debug_static_folder(),
            {
                "svecs": mapping["svecs"],
                "multi": mapping["multi"],
                "s2p_map": mapping["s2p_map"],
                "p2s_map": mapping["p2s_map"],
                "s2pp_map": mapping["s2pp_map"],
                "dd_real_q0": dd_real_q0,
                "short_range_force_constants": fc_short,
            },
        )
        self._gonze_save_debug(
            self._gonze_debug_q_folder(),
            {
                "q_red": q_red,
                "q_cart": q_cart,
                "q_direction_cart": q_direction_cart,
                "dd_recip": dd_recip,
                "dd_real": dd_real,
                "dd_limiting_expanded": dd_limiting_expanded,
                "dd_drift": dd_drift,
                "dd_total_mass_weighted": dd_total_mass_weighted,
                "dm_short": dm_short,
                "dm_final": dm_final,
                "eigenvalues": eigvals,
                "frequencies": frequencies,
            },
        )
        return dm_final
```

- [ ] **Step 4: Run full-matrix test**

Run:

```bash
pytest kaldo/tests/test_gonze_lee_nac_api.py::test_gonze_full_dynamical_matrix_returns_hermitian_matrix -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add kaldo/observables/harmonic_with_q.py kaldo/tests/test_gonze_lee_nac_api.py
git commit -m "feat: assemble Gonze-Lee full dynamical matrix"
```

### Task 8: Route Frequencies Through Gonze Matrix

**Files:**
- Modify: `kaldo/tests/test_gonze_lee_nac_api.py`
- Modify: `kaldo/observables/harmonic_with_q.py`

- [ ] **Step 1: Add frequency integration test**

Append to `kaldo/tests/test_gonze_lee_nac_api.py`:

```python
def test_gonze_frequency_calculation_returns_real_frequencies(nac_second_order, tmp_path):
    phonon = HarmonicWithQ(
        q_point=np.array([0.1, 0.0, 0.1]),
        second=nac_second_order,
        storage="memory",
        nac_method="gonze",
        nac_debug=True,
        nac_debug_folder=str(tmp_path / "debug"),
        q_index=5,
    )
    frequency = phonon.frequency.flatten()
    assert frequency.shape == (6,)
    assert np.isfinite(frequency).all()
    assert (tmp_path / "debug" / "q-00005" / "frequencies.npy").exists()
```

- [ ] **Step 2: Run integration test and verify it fails or uses legacy**

Run:

```bash
pytest kaldo/tests/test_gonze_lee_nac_api.py::test_gonze_frequency_calculation_returns_real_frequencies -q
```

Expected before routing: FAIL if debug frequency file is absent.

- [ ] **Step 3: Route `calculate_eigensystem`**

At the start of `calculate_eigensystem`, insert:

```python
        if self.nac_method == 'gonze':
            dyn_s = self._calculate_gonze_dynamical_matrix()
            if only_eigenvals:
                return tf.convert_to_tensor(np.linalg.eigvalsh(dyn_s).real)
            log_size(dyn_s.shape, type=complex, name='eigensystem')
            eigenvals, eigenvects = np.linalg.eigh(dyn_s)
            esystem = np.vstack((eigenvals[np.newaxis, :], eigenvects))
            return tf.convert_to_tensor(esystem)
```

- [ ] **Step 4: Route unfolded eigensystem**

At the start of `calculate_eigensystem_unfolded`, insert the same block:

```python
        if self.nac_method == 'gonze':
            dyn_s = self._calculate_gonze_dynamical_matrix()
            if only_eigenvals:
                return tf.convert_to_tensor(np.linalg.eigvalsh(dyn_s).real)
            eigenvals, eigenvects = np.linalg.eigh(dyn_s)
            esystem = np.vstack((eigenvals[np.newaxis, :], eigenvects))
            return tf.convert_to_tensor(esystem)
```

- [ ] **Step 5: Run frequency integration test**

Run:

```bash
pytest kaldo/tests/test_gonze_lee_nac_api.py::test_gonze_frequency_calculation_returns_real_frequencies -q
```

Expected: PASS.

- [ ] **Step 6: Run default legacy NAC tests**

Run:

```bash
pytest kaldo/tests/test_nac_qe.py -q
```

Expected: PASS, confirming the default legacy path remains unchanged.

- [ ] **Step 7: Commit**

```bash
git add kaldo/observables/harmonic_with_q.py kaldo/tests/test_gonze_lee_nac_api.py
git commit -m "feat: route Gonze-Lee NAC frequencies"
```

### Task 9: Add NaCl Debug Frequency Regression

**Files:**
- Modify: `kaldo/tests/test_gonze_lee_nac_api.py`
- Modify: `kaldo/observables/harmonic_with_q.py`

- [ ] **Step 1: Add reference frequency test for selected q-points**

Append to `kaldo/tests/test_gonze_lee_nac_api.py`:

```python
@pytest.mark.parametrize("q_name", ["q-00000", "q-00013", "q-00030"])
def test_gonze_nacl_frequencies_match_reference_within_two_percent(
    nac_second_order, tmp_path, q_name
):
    debug_dir = require_nacl_debug()
    q_dir = debug_dir / q_name
    q_point = np.load(q_dir / "q_red.npy")
    q_index = int(q_name.split("-")[1])
    phonon = HarmonicWithQ(
        q_point=q_point,
        second=nac_second_order,
        storage="memory",
        is_unfolding=True,
        nac_method="gonze",
        nac_debug=True,
        nac_debug_folder=str(tmp_path / "debug"),
        q_index=q_index,
    )
    actual = phonon.frequency.flatten()
    expected = np.load(q_dir / "frequencies.npy")
    np.testing.assert_allclose(actual[3:], expected[3:], rtol=0.02, atol=0.05)
```

- [ ] **Step 2: Run reference frequency test**

Run:

```bash
pytest kaldo/tests/test_gonze_lee_nac_api.py::test_gonze_nacl_frequencies_match_reference_within_two_percent -q
```

Expected in this workspace: PASS.

- [ ] **Step 3: Run reference frequency test again**

Run:

```bash
pytest kaldo/tests/test_gonze_lee_nac_api.py::test_gonze_nacl_frequencies_match_reference_within_two_percent -q
```

Expected: PASS with all selected q-points under 2% relative tolerance for optical modes.

- [ ] **Step 4: Commit**

```bash
git add kaldo/observables/harmonic_with_q.py kaldo/tests/test_gonze_lee_nac_api.py
git commit -m "test: add Gonze-Lee NaCl frequency regression"
```

### Task 10: Update Example And Run Focused Verification

**Files:**
- Modify: `examples/nacl_phonopy/dispersion.py`

- [ ] **Step 1: Update NaCl example config**

In `examples/nacl_phonopy/dispersion.py`, update `phonons_config` to:

```python
phonons_config = {
    "kpts": [k_points, k_points, k_points],
    "is_classic": False,
    "is_unfolding": True,
    "temperature": 300,
    "folder": "ALD_with_NAC",
    "nac_method": "gonze",
    "nac_debug": True,
    "nac_debug_folder": "debug",
    "storage": "formatted",
}
```

- [ ] **Step 2: Run all focused tests**

Run:

```bash
pytest kaldo/tests/test_gonze_lee_nac_helpers.py kaldo/tests/test_gonze_lee_nac_api.py kaldo/tests/test_nac_qe.py -q
```

Expected in this workspace: PASS.

- [ ] **Step 3: Run the NaCl example**

Run:

```bash
cd examples/nacl_phonopy
python dispersion.py
```

Expected: script completes and writes debug files under `examples/nacl_phonopy/debug/`.

- [ ] **Step 4: Inspect generated debug files**

Run:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

root = Path("examples/nacl_phonopy/debug")
for name in ["static/G_list.npy", "q-00000/frequencies.npy", "q-00013/frequencies.npy", "q-00030/frequencies.npy"]:
    path = root / name
    print(name, path.exists(), np.load(path).shape if path.exists() else "")
PY
```

Expected: each line prints `True` and the frequency files have shape `(6,)`.

- [ ] **Step 5: Commit**

```bash
git add examples/nacl_phonopy/dispersion.py
git commit -m "docs: enable Gonze-Lee debug in NaCl example"
```

### Task 11: Final Verification

**Files:**
- No planned edits.

- [ ] **Step 1: Run focused test suite**

Run:

```bash
pytest kaldo/tests/test_gonze_lee_nac_helpers.py kaldo/tests/test_gonze_lee_nac_api.py kaldo/tests/test_nac_qe.py -q
```

Expected: PASS.

- [ ] **Step 2: Check git status**

Run:

```bash
git status --short
```

Expected: only intentional untracked user files remain, such as `.codex` and any pre-existing untracked `examples/` files that were not staged.

- [ ] **Step 3: Summarize final numerical evidence**

Run:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

ref = Path("/data/nwlundgren/rephonopy/.worktrees/gonze-lee-nac-debug-replay/example/nacl-att2/debug")
got = Path("examples/nacl_phonopy/debug")
for q_name in ["q-00000", "q-00013", "q-00030"]:
    ref_freq = np.load(ref / q_name / "frequencies.npy")
    got_freq = np.load(got / q_name / "frequencies.npy")
    rel = np.max(np.abs(got_freq[3:] - ref_freq[3:]) / np.maximum(np.abs(ref_freq[3:]), 1e-12))
    print(q_name, f"max optical relative difference = {rel:.6f}")
PY
```

Expected: each printed relative difference is less than `0.02`.
