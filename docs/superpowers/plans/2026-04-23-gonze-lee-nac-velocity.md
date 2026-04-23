# Gonze-Lee NAC Velocity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add production-ready Gonze-Lee NAC group velocities in kALDo by matching phonopy's finite-difference velocity debug output on the four diagnostic att3 q-points and then routing the public `HarmonicWithQ.velocity` API through that same implementation.

**Architecture:** Keep `kaldo/observables/harmonic_with_q.py` as the single implementation point. Add private Gonze velocity helpers that reuse the existing Gonze dynamical-matrix path for the central q-point and finite-difference offsets, emit phonopy-shaped debug artifacts, and make the parity tests fail at the earliest wrong intermediate. Keep the reference-loading and parity harness in the test tree, with a dedicated velocity test module separate from the existing NAC API tests.

**Tech Stack:** Python, NumPy, ASE units, pytest, existing Gonze NAC helpers in `HarmonicWithQ`, phonopy `debug-velocity` reference data.

---

## File Structure

- Modify: `kaldo/tests/gonze_debug_reference.py`
  - Add velocity-debug reference paths and loaders for top-level q-point tensors, per-direction tensors, and JSON payloads.

- Modify: `kaldo/tests/test_gonze_lee_nac_helpers.py`
  - Add narrow unit tests for the new velocity reference helpers.

- Create: `kaldo/tests/test_gonze_lee_nac_velocity.py`
  - Add the dedicated four-point Gonze velocity parity harness and layered parity tests.

- Modify: `kaldo/observables/harmonic_with_q.py`
  - Add private Gonze velocity constants and helper functions.
  - Add a reusable Gonze velocity debug-data orchestrator.
  - Reuse the existing Gonze dynamical-matrix path for finite-difference plus/minus evaluations.
  - Route `calculate_velocity()` through the Gonze path when `nac_method="gonze"`.

- Modify: `kaldo/tests/test_gonze_lee_nac_api.py`
  - Replace the current `NotImplementedError` expectation with public Gonze velocity parity and smoke tests.

## Reference Inputs

- Phonopy velocity debug ground truth:
  - `/home/nwlundgren/data/rephonopy/.worktrees/gonze-velocity-debug-dump/example/nacl-att3/debug-velocity`

- kALDo force constants:
  - `examples/nacl_phonopy_v2`

- Injected NAC metadata:
  - `examples/nacl_phonopy/espresso.ifc2`

- Diagnostic q-points:
  - `q-00000`
  - `q-00013`
  - `q-00020`
  - `q-00030`

## Task 1: Add Velocity Reference Helpers

**Files:**
- Modify: `kaldo/tests/gonze_debug_reference.py`
- Modify: `kaldo/tests/test_gonze_lee_nac_helpers.py`
- Test: `kaldo/tests/test_gonze_lee_nac_helpers.py`

- [ ] **Step 1: Write the failing helper tests**

Add these imports and tests near the top of `kaldo/tests/test_gonze_lee_nac_helpers.py`:

```python
from kaldo.tests.gonze_debug_reference import (
    load_velocity_direction_tensor,
    load_velocity_json,
    load_velocity_q_tensor,
    require_nacl_att3_velocity_debug,
)


def test_att3_velocity_debug_tree_is_loadable():
    root = require_nacl_att3_velocity_debug()
    assert (root / "q-00013" / "gv_scaled.npy").exists()


def test_load_velocity_direction_tensor_reads_per_direction_matrix():
    root = require_nacl_att3_velocity_debug()
    actual = load_velocity_direction_tensor(root, "q-00013", "d0", "ddm_fd")
    assert actual.shape == (6, 6)
    assert np.iscomplexobj(actual)


def test_load_velocity_json_reads_named_payload():
    root = require_nacl_att3_velocity_debug()
    payload = load_velocity_json(root, "q-00013", "degenerate_sets")
    assert payload["sets"] == [[0, 1], [2], [3, 4], [5]]
    np.testing.assert_allclose(
        load_velocity_q_tensor(root, "q-00013", "gv_cutoff_mask"),
        np.array([1, 1, 1, 1, 1, 1]),
        atol=0,
        rtol=0,
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest kaldo/tests/test_gonze_lee_nac_helpers.py -k velocity_debug -q
```

Expected: FAIL with `ImportError` for `require_nacl_att3_velocity_debug` and the other new helper names.

- [ ] **Step 3: Add the helper implementation**

In `kaldo/tests/gonze_debug_reference.py`, add `import json` with the imports and add these helpers below `require_nacl_att3_debug()`:

```python
DEFAULT_NACL_ATT3_VELOCITY_DEBUG = Path(
    "/home/nwlundgren/data/rephonopy/.worktrees/gonze-velocity-debug-dump/"
    "example/nacl-att3/debug-velocity"
)


def nacl_att3_velocity_debug_dir() -> Path:
    return Path(
        os.environ.get(
            "NACL_ATT3_VELOCITY_DEBUG_DIR",
            DEFAULT_NACL_ATT3_VELOCITY_DEBUG,
        )
    )


def require_nacl_att3_velocity_debug() -> Path:
    path = nacl_att3_velocity_debug_dir()
    if not (path / "q-00013" / "gv_scaled.npy").exists():
        pytest.skip(f"NaCl att3 velocity debug tree not found at {path}")
    return path


def load_velocity_q_tensor(root: Path, q_name: str, name: str) -> np.ndarray:
    return np.load(root / q_name / f"{name}.npy", allow_pickle=False)


def load_velocity_direction_tensor(
    root: Path, q_name: str, direction_name: str, name: str
) -> np.ndarray:
    return np.load(root / q_name / direction_name / f"{name}.npy", allow_pickle=False)


def load_velocity_json(root: Path, q_name: str, name: str) -> dict:
    return json.loads((root / q_name / f"{name}.json").read_text())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python -m pytest kaldo/tests/test_gonze_lee_nac_helpers.py -k velocity_debug -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add kaldo/tests/gonze_debug_reference.py kaldo/tests/test_gonze_lee_nac_helpers.py
git commit -m "test: add Gonze velocity reference helpers"
```

## Task 2: Create The Dedicated Velocity Parity Harness

**Files:**
- Create: `kaldo/tests/test_gonze_lee_nac_velocity.py`
- Modify: `kaldo/observables/harmonic_with_q.py`
- Test: `kaldo/tests/test_gonze_lee_nac_velocity.py`

- [ ] **Step 1: Write the failing harness test**

Create `kaldo/tests/test_gonze_lee_nac_velocity.py` with this initial content:

```python
import numpy as np
import pytest

from kaldo.forceconstants import ForceConstants
from kaldo.interfaces import shengbte_io
from kaldo.observables.harmonic_with_q import HarmonicWithQ


def attach_reference_nac(second_order, nac_file="examples/nacl_phonopy/espresso.ifc2"):
    _, _, charges = shengbte_io.read_second_order_qe_matrix(nac_file)
    second_order.atoms.info["dielectric"] = charges[0, :, :]
    second_order.atoms.set_array("charges", charges[1:, :, :], shape=(3, 3))
    return second_order


@pytest.fixture(scope="module")
def nac_second_order(tmp_path_factory):
    forceconstants = ForceConstants.from_folder(
        folder="examples/nacl_phonopy_v2",
        supercell=[8, 8, 8],
        only_second=True,
        is_acoustic_sum=True,
        format="shengbte-qe",
    )
    forceconstants.second.folder = str(tmp_path_factory.mktemp("gonze_velocity_cache"))
    return attach_reference_nac(forceconstants.second)


def test_gonze_velocity_debug_data_contains_top_level_fields(nac_second_order):
    phonon = HarmonicWithQ(
        q_point=np.array([0.1, 0.0, 0.1]),
        second=nac_second_order,
        storage="memory",
        is_unfolding=True,
        nac_method="gonze",
    )
    data = phonon._calculate_gonze_velocity_debug_data()
    assert data["dm_q"].shape == (6, 6)
    assert data["eigenvectors"].shape == (6, 6)
    assert sorted(data["directions"]) == ["d0", "d1", "d2", "d3"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest kaldo/tests/test_gonze_lee_nac_velocity.py -k top_level_fields -q
```

Expected: FAIL with `AttributeError` because `_calculate_gonze_velocity_debug_data` does not exist yet.

- [ ] **Step 3: Add the minimal private velocity skeleton**

In `kaldo/observables/harmonic_with_q.py`, add `import json` with the imports, add these module-level constants after `MIN_N_MODES_TO_STORE`, and add the helper methods inside `HarmonicWithQ` just below `_gonze_save_debug()`:

```python
GONZE_VELOCITY_Q_LENGTH = 1e-5
GONZE_VELOCITY_DEGENERACY_TOLERANCE = 1e-4
GONZE_VELOCITY_CUTOFF_FREQUENCY = 1e-4
GONZE_VELOCITY_DIRECTIONS_CART = np.array(
    [
        np.array([1.0, 2.0, 3.0], dtype=float) / np.sqrt(14.0),
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([0.0, 0.0, 1.0], dtype=float),
    ],
    dtype=float,
)


def _gonze_degenerate_sets(frequencies, tolerance=GONZE_VELOCITY_DEGENERACY_TOLERANCE):
    sets = []
    current = [0]
    for index in range(1, len(frequencies)):
        if abs(frequencies[index] - frequencies[current[-1]]) < tolerance:
            current.append(index)
        else:
            sets.append(current)
            current = [index]
    sets.append(current)
    return sets
```

```python
def _gonze_save_debug_json(self, folder, payloads):
    if not self.nac_debug:
        return
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        (folder / f"{name}.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


def _calculate_gonze_velocity_debug_data(self):
    static_data = self._build_gonze_static_data()
    q_red = np.array(self.q_point, dtype=float, copy=True)
    q_cart = static_data["reciprocal_lattice"] @ q_red
    dm_q = self._calculate_gonze_dynamical_matrix()
    eigenvalues, eigenvectors = np.linalg.eigh(dm_q)
    eigenvalues = eigenvalues.real
    frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)
    return {
        "q_red": q_red,
        "q_cart": q_cart,
        "dm_q": dm_q,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "frequencies": frequencies,
        "directions": {
            f"d{index}": {"direction_cart": direction.copy()}
            for index, direction in enumerate(GONZE_VELOCITY_DIRECTIONS_CART)
        },
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
python -m pytest kaldo/tests/test_gonze_lee_nac_velocity.py -k top_level_fields -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add kaldo/observables/harmonic_with_q.py kaldo/tests/test_gonze_lee_nac_velocity.py
git commit -m "test: add Gonze velocity parity harness"
```

## Task 3: Add Central-Q Velocity Parity

**Files:**
- Modify: `kaldo/tests/test_gonze_lee_nac_velocity.py`
- Modify: `kaldo/observables/harmonic_with_q.py`
- Test: `kaldo/tests/test_gonze_lee_nac_velocity.py`

- [ ] **Step 1: Write the failing central-q parity tests**

Add these imports, helpers, and tests to `kaldo/tests/test_gonze_lee_nac_velocity.py`:

```python
from kaldo.tests.gonze_debug_reference import (
    diagnostic_q_names_att3,
    format_tensor_diff,
    load_velocity_direction_tensor,
    load_velocity_json,
    load_velocity_q_tensor,
    require_nacl_att3_velocity_debug,
)


TOP_LEVEL_TENSOR_NAMES = [
    "q_red",
    "q_cart",
    "dm_q",
    "eigenvalues",
    "eigenvectors",
    "frequencies",
]


def _run_gonze_velocity_debug_for_q_name(nac_second_order, q_name, debug_root, out_root):
    q_point = load_velocity_q_tensor(debug_root, q_name, "q_red")
    q_index = int(q_name.split("-")[1])
    phonon = HarmonicWithQ(
        q_point=q_point,
        second=nac_second_order,
        storage="memory",
        is_unfolding=True,
        nac_method="gonze",
        nac_debug=True,
        nac_debug_folder=str(out_root),
        q_index=q_index,
    )
    _ = phonon._calculate_gonze_velocity_debug_data()
    return out_root / q_name


@pytest.mark.parametrize("q_name", diagnostic_q_names_att3())
@pytest.mark.parametrize("tensor_name", TOP_LEVEL_TENSOR_NAMES)
def test_gonze_velocity_top_level_tensors_match_phonopy_debug(
    nac_second_order, tmp_path, q_name, tensor_name
):
    debug_dir = require_nacl_att3_velocity_debug()
    actual_root = _run_gonze_velocity_debug_for_q_name(
        nac_second_order, q_name, debug_dir, tmp_path / "debug-velocity"
    )
    actual = np.load(actual_root / f"{tensor_name}.npy", allow_pickle=False)
    expected = load_velocity_q_tensor(debug_dir, q_name, tensor_name)
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.02,
        atol=1e-8,
        err_msg=format_tensor_diff(tensor_name, q_name, actual, expected),
    )


@pytest.mark.parametrize("q_name", diagnostic_q_names_att3())
def test_gonze_velocity_json_payloads_match_phonopy_debug(nac_second_order, tmp_path, q_name):
    debug_dir = require_nacl_att3_velocity_debug()
    actual_root = _run_gonze_velocity_debug_for_q_name(
        nac_second_order, q_name, debug_dir, tmp_path / "debug-velocity"
    )
    assert load_velocity_json(actual_root.parent, q_name, "degenerate_sets") == load_velocity_json(
        debug_dir, q_name, "degenerate_sets"
    )
    assert load_velocity_json(actual_root.parent, q_name, "nac_branch") == load_velocity_json(
        debug_dir, q_name, "nac_branch"
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest kaldo/tests/test_gonze_lee_nac_velocity.py -k top_level_tensors -q
```

Expected: FAIL because `q_red.npy`, `dm_q.npy`, `degenerate_sets.json`, or `nac_branch.json` are not yet written in the phonopy-like layout.

- [ ] **Step 3: Extend the private helper to dump phonopy-shaped top-level artifacts**

Update `kaldo/observables/harmonic_with_q.py` by replacing `_calculate_gonze_velocity_debug_data()` with:

```python
def _calculate_gonze_velocity_debug_data(self):
    static_data = self._build_gonze_static_data()
    q_red = np.array(self.q_point, dtype=float, copy=True)
    q_cart = static_data["reciprocal_lattice"] @ q_red
    dm_q = self._calculate_gonze_dynamical_matrix()
    eigenvalues, eigenvectors = np.linalg.eigh(dm_q)
    eigenvalues = eigenvalues.real
    frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)
    degenerate_sets = _gonze_degenerate_sets(frequencies)
    nac_branch = {
        "nac_applied": bool(np.linalg.norm(q_cart) >= 1e-5),
        "q_direction_red": None,
        "q_norm": float(np.linalg.norm(q_cart)),
        "q_red": q_red.tolist(),
        "tolerance": 1e-5,
    }
    data = {
        "q_red": q_red,
        "q_cart": q_cart,
        "dm_q": dm_q,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "frequencies": frequencies,
        "degenerate_sets": {"sets": degenerate_sets},
        "nac_branch": nac_branch,
        "directions": {
            f"d{index}": {"direction_cart": direction.copy()}
            for index, direction in enumerate(GONZE_VELOCITY_DIRECTIONS_CART)
        },
    }
    self._gonze_save_debug(
        self._gonze_debug_q_folder(),
        {
            "q_red": q_red,
            "q_cart": q_cart,
            "dm_q": dm_q,
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "frequencies": frequencies,
        },
    )
    self._gonze_save_debug_json(
        self._gonze_debug_q_folder(),
        {
            "degenerate_sets": {"sets": degenerate_sets},
            "nac_branch": nac_branch,
        },
    )
    return data
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python -m pytest kaldo/tests/test_gonze_lee_nac_velocity.py -k "top_level_tensors or json_payloads" -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add kaldo/observables/harmonic_with_q.py kaldo/tests/test_gonze_lee_nac_velocity.py
git commit -m "test: add central-q Gonze velocity parity"
```

## Task 4: Add Finite-Difference Direction Parity

**Files:**
- Modify: `kaldo/observables/harmonic_with_q.py`
- Modify: `kaldo/tests/test_gonze_lee_nac_velocity.py`
- Test: `kaldo/tests/test_gonze_lee_nac_velocity.py`

- [ ] **Step 1: Write the failing per-direction parity tests**

Add these constants and tests to `kaldo/tests/test_gonze_lee_nac_velocity.py`:

```python
DIRECTION_NAMES = ["d0", "d1", "d2", "d3"]
DIRECTION_TENSOR_NAMES = [
    "direction_cart",
    "dq_cart",
    "dq_red",
    "dm_minus",
    "dm_plus",
    "delta_dm",
    "ddm_fd",
]


@pytest.mark.parametrize("q_name", diagnostic_q_names_att3())
@pytest.mark.parametrize("direction_name", DIRECTION_NAMES)
@pytest.mark.parametrize("tensor_name", DIRECTION_TENSOR_NAMES)
def test_gonze_velocity_direction_tensors_match_phonopy_debug(
    nac_second_order, tmp_path, q_name, direction_name, tensor_name
):
    debug_dir = require_nacl_att3_velocity_debug()
    actual_root = _run_gonze_velocity_debug_for_q_name(
        nac_second_order, q_name, debug_dir, tmp_path / "debug-velocity"
    )
    actual = np.load(
        actual_root / direction_name / f"{tensor_name}.npy",
        allow_pickle=False,
    )
    expected = load_velocity_direction_tensor(debug_dir, q_name, direction_name, tensor_name)
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.02,
        atol=1e-8,
        err_msg=format_tensor_diff(f"{direction_name}.{tensor_name}", q_name, actual, expected),
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest kaldo/tests/test_gonze_lee_nac_velocity.py -k direction_tensors -q
```

Expected: FAIL because the per-direction `dq_*`, `dm_minus`, `dm_plus`, `delta_dm`, and `ddm_fd` artifacts are not yet produced.

- [ ] **Step 3: Implement the finite-difference helper path**

In `kaldo/observables/harmonic_with_q.py`, add these methods just above `_calculate_gonze_velocity_debug_data()` and update `_calculate_gonze_velocity_debug_data()` to call them:

```python
def _calculate_gonze_dynamical_matrix_for_q(self, q_red):
    original_q_point = np.array(self.q_point, dtype=float, copy=True)
    original_debug = self.nac_debug
    self.q_point = np.array(q_red, dtype=float, copy=True)
    self.nac_debug = False
    try:
        return self._calculate_gonze_dynamical_matrix()
    finally:
        self.q_point = original_q_point
        self.nac_debug = original_debug


def _calculate_gonze_velocity_direction_data(self, direction_index, static_data):
    if direction_index not in range(4):
        raise ValueError(f"direction_index must be in 0..3, got {direction_index}")
    direction_cart = np.array(
        GONZE_VELOCITY_DIRECTIONS_CART[direction_index], dtype=float, copy=True
    )
    dq_cart = direction_cart / np.linalg.norm(direction_cart) * GONZE_VELOCITY_Q_LENGTH
    dq_red = static_data["primitive_cell"] @ dq_cart
    q_red = np.array(self.q_point, dtype=float, copy=True)
    dm_minus = self._calculate_gonze_dynamical_matrix_for_q(q_red - dq_red)
    dm_plus = self._calculate_gonze_dynamical_matrix_for_q(q_red + dq_red)
    delta_dm = dm_plus - dm_minus
    ddm_fd = delta_dm / (2 * GONZE_VELOCITY_Q_LENGTH)
    return {
        "direction_cart": direction_cart,
        "dq_cart": dq_cart,
        "dq_red": dq_red,
        "dm_minus": dm_minus,
        "dm_plus": dm_plus,
        "delta_dm": delta_dm,
        "ddm_fd": ddm_fd,
    }
```

Replace the `directions` part of `_calculate_gonze_velocity_debug_data()` with:

```python
    directions = {}
    for index in range(4):
        direction_name = f"d{index}"
        direction_data = self._calculate_gonze_velocity_direction_data(index, static_data)
        directions[direction_name] = direction_data
        self._gonze_save_debug(self._gonze_debug_q_folder() / direction_name, direction_data)
```

and update the returned `data` dictionary to use `directions`.

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python -m pytest kaldo/tests/test_gonze_lee_nac_velocity.py -k direction_tensors -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add kaldo/observables/harmonic_with_q.py kaldo/tests/test_gonze_lee_nac_velocity.py
git commit -m "test: add Gonze velocity finite-difference parity"
```

## Task 5: Add Velocity Assembly Parity And Enable The Public API

**Files:**
- Modify: `kaldo/observables/harmonic_with_q.py`
- Modify: `kaldo/tests/test_gonze_lee_nac_velocity.py`
- Modify: `kaldo/tests/test_gonze_lee_nac_api.py`
- Test: `kaldo/tests/test_gonze_lee_nac_velocity.py`
- Test: `kaldo/tests/test_gonze_lee_nac_api.py`

- [ ] **Step 1: Write the failing velocity-assembly and public-API tests**

Add these tests to `kaldo/tests/test_gonze_lee_nac_velocity.py`:

```python
VELOCITY_TENSOR_NAMES = [
    "gv_raw",
    "gv_scaling_prefactor",
    "gv_cutoff_mask",
    "gv_scaled",
]


@pytest.mark.parametrize("q_name", diagnostic_q_names_att3())
@pytest.mark.parametrize("tensor_name", VELOCITY_TENSOR_NAMES)
def test_gonze_velocity_outputs_match_phonopy_debug(
    nac_second_order, tmp_path, q_name, tensor_name
):
    debug_dir = require_nacl_att3_velocity_debug()
    actual_root = _run_gonze_velocity_debug_for_q_name(
        nac_second_order, q_name, debug_dir, tmp_path / "debug-velocity"
    )
    actual = np.load(actual_root / f"{tensor_name}.npy", allow_pickle=False)
    expected = load_velocity_q_tensor(debug_dir, q_name, tensor_name)
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.02,
        atol=0.05,
        err_msg=format_tensor_diff(tensor_name, q_name, actual, expected),
    )
```

Add these imports to `kaldo/tests/test_gonze_lee_nac_api.py`:

```python
from kaldo.phonons import Phonons
from kaldo.tests.gonze_debug_reference import (
    format_tensor_diff,
    load_velocity_q_tensor,
    require_nacl_att3_velocity_debug,
)
```

Replace `test_gonze_velocity_raises_not_implemented` in `kaldo/tests/test_gonze_lee_nac_api.py` with:

```python
def test_gonze_velocity_matches_phonopy_debug(nac_second_order):
    debug_dir = require_nacl_att3_velocity_debug()
    q_name = "q-00013"
    phonon = HarmonicWithQ(
        q_point=load_velocity_q_tensor(debug_dir, q_name, "q_red"),
        second=nac_second_order,
        storage="memory",
        is_unfolding=True,
        nac_method="gonze",
    )
    actual = phonon.velocity[0]
    expected = load_velocity_q_tensor(debug_dir, q_name, "gv_scaled")
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.02,
        atol=0.05,
        err_msg=format_tensor_diff("gv_scaled", q_name, actual, expected),
    )
```

Add this smoke test just below it:

```python
def test_phonons_gonze_velocity_returns_finite_array():
    forceconstants = ForceConstants.from_folder(
        folder="examples/nacl_phonopy_v2",
        supercell=[8, 8, 8],
        only_second=True,
        is_acoustic_sum=True,
        format="shengbte-qe",
    )
    attach_reference_nac(forceconstants.second)
    phonons = Phonons(
        forceconstants=forceconstants,
        kpts=[1, 1, 1],
        storage="memory",
        nac_method="gonze",
    )
    velocity = phonons.velocity
    assert velocity.shape == (1, 6, 3)
    assert np.isfinite(velocity).all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest kaldo/tests/test_gonze_lee_nac_velocity.py -k velocity_outputs -q
python -m pytest kaldo/tests/test_gonze_lee_nac_api.py -k gonze_velocity -q
```

Expected: FAIL because `gv_raw`, `gv_scaled`, and the public `phonon.velocity` Gonze path are not implemented yet.

- [ ] **Step 3: Implement degeneracy projection, scaling, and public routing**

In `kaldo/observables/harmonic_with_q.py`, add these methods below `_calculate_gonze_velocity_direction_data()`:

```python
def _project_gonze_group_velocity_raw(self, ddms, eigenvectors, frequencies):
    gv_raw = np.zeros((len(frequencies), 3), dtype=float)
    degenerate_sets = _gonze_degenerate_sets(frequencies)
    for indices in degenerate_sets:
        subspace = eigenvectors[:, indices]
        perturbation = subspace.conj().T @ ddms[0] @ subspace
        _, rotation = np.linalg.eigh((perturbation + perturbation.conj().T) / 2)
        rotated = subspace @ rotation
        for axis, ddm in enumerate(ddms[1:]):
            projected = rotated.conj().T @ ddm @ rotated
            gv_raw[np.array(indices), axis] = np.real(np.diag(projected))
    return gv_raw


def _scale_gonze_group_velocity_raw(self, gv_raw, frequencies):
    scaling = np.zeros(len(frequencies), dtype=float)
    cutoff_mask = (frequencies > GONZE_VELOCITY_CUTOFF_FREQUENCY).astype(np.int64)
    active = cutoff_mask.astype(bool)
    scaling[active] = 1.0 / (8.0 * np.pi ** 2 * frequencies[active])
    gv_scaled = gv_raw * scaling[:, np.newaxis]
    gv_scaled[~active] = 0.0
    return gv_scaled, scaling, cutoff_mask
```

Update `_calculate_gonze_velocity_debug_data()` so that after the `directions` loop it computes and saves the final velocity tensors:

```python
    ddms = [directions[f"d{index}"]["ddm_fd"] for index in range(4)]
    gv_raw = self._project_gonze_group_velocity_raw(ddms, eigenvectors, frequencies)
    gv_scaled, gv_scaling_prefactor, gv_cutoff_mask = self._scale_gonze_group_velocity_raw(
        gv_raw, frequencies
    )
    self._gonze_save_debug(
        self._gonze_debug_q_folder(),
        {
            "q_red": q_red,
            "q_cart": q_cart,
            "dm_q": dm_q,
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "frequencies": frequencies,
            "gv_raw": gv_raw,
            "gv_scaling_prefactor": gv_scaling_prefactor,
            "gv_cutoff_mask": gv_cutoff_mask,
            "gv_scaled": gv_scaled,
        },
    )
    data["gv_raw"] = gv_raw
    data["gv_scaling_prefactor"] = gv_scaling_prefactor
    data["gv_cutoff_mask"] = gv_cutoff_mask
    data["gv_scaled"] = gv_scaled
```

Finally, replace the Gonze branch in `calculate_velocity()` with:

```python
def calculate_velocity(self):
    if self.nac_method == "gonze":
        return self._calculate_gonze_velocity_debug_data()["gv_scaled"][np.newaxis, ...]
    frequency = self.frequency[0]
    velocity = np.zeros((self.n_modes, 3))
    inverse_sqrt_freq = tf.cast(tf.convert_to_tensor(1 / np.sqrt(frequency)), tf.complex128)
    if self.is_amorphous:
        inverse_sqrt_freq = tf.cast(inverse_sqrt_freq, tf.float64)
    for alpha in range(3):
        if alpha == 0:
            sij = self._sij_x
        if alpha == 1:
            sij = self._sij_y
        if alpha == 2:
            sij = self._sij_z
        velocity_AF = 1 / (2 * np.pi) * contract(
            "mn,m,n->mn",
            sij,
            inverse_sqrt_freq,
            inverse_sqrt_freq,
            backend="tensorflow",
        ) / 2
        velocity_AF = tf.where(tf.math.is_nan(tf.math.real(velocity_AF)), 0.0, velocity_AF)
        velocity[..., alpha] = contract("mm->m", velocity_AF.numpy().imag)
    return velocity[np.newaxis, ...]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python -m pytest kaldo/tests/test_gonze_lee_nac_velocity.py -q
python -m pytest kaldo/tests/test_gonze_lee_nac_api.py -k gonze_velocity -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add kaldo/observables/harmonic_with_q.py kaldo/tests/test_gonze_lee_nac_velocity.py kaldo/tests/test_gonze_lee_nac_api.py
git commit -m "feat: enable Gonze NAC velocity parity path"
```

## Task 6: Run Focused Regression Coverage

**Files:**
- Test: `kaldo/tests/test_gonze_lee_nac_helpers.py`
- Test: `kaldo/tests/test_gonze_lee_nac_velocity.py`
- Test: `kaldo/tests/test_gonze_lee_nac_api.py`
- Test: `kaldo/tests/test_nac_qe.py`

- [ ] **Step 1: Add the final regression command list to your scratchpad**

Use this exact command set:

```bash
python -m pytest kaldo/tests/test_gonze_lee_nac_helpers.py -q
python -m pytest kaldo/tests/test_gonze_lee_nac_velocity.py -q
python -m pytest kaldo/tests/test_gonze_lee_nac_api.py -q
python -m pytest kaldo/tests/test_nac_qe.py -q
```

- [ ] **Step 2: Run the helper and parity suites**

Run:

```bash
python -m pytest kaldo/tests/test_gonze_lee_nac_helpers.py -q
python -m pytest kaldo/tests/test_gonze_lee_nac_velocity.py -q
python -m pytest kaldo/tests/test_gonze_lee_nac_api.py -q
```

Expected: PASS

- [ ] **Step 3: Run the existing NAC regression test**

Run:

```bash
python -m pytest kaldo/tests/test_nac_qe.py -q
```

Expected: PASS, including the legacy NAC velocity check.

- [ ] **Step 4: Inspect the diff before finishing**

Run:

```bash
git status --short
git diff -- kaldo/observables/harmonic_with_q.py kaldo/tests/gonze_debug_reference.py kaldo/tests/test_gonze_lee_nac_helpers.py kaldo/tests/test_gonze_lee_nac_velocity.py kaldo/tests/test_gonze_lee_nac_api.py
```

Expected: Only the planned files are modified, and the diff shows the staged Gonze velocity implementation plus parity coverage.

- [ ] **Step 5: Commit the regression-verified state**

```bash
git add kaldo/observables/harmonic_with_q.py kaldo/tests/gonze_debug_reference.py kaldo/tests/test_gonze_lee_nac_helpers.py kaldo/tests/test_gonze_lee_nac_velocity.py kaldo/tests/test_gonze_lee_nac_api.py
git commit -m "test: verify Gonze NAC velocity rollout"
```
