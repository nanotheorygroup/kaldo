"""End-to-end checks for the staged sparse dynamical-matrix command."""

import json

import ase.io
import numpy as np
import pytest
from scipy.sparse import load_npz

from kaldo.controllers.colored_fd.build_dynmat import main
from kaldo.observables.sparse_secondorder import SparseSecondOrder
from kaldo.tests.colored_fd_helpers import chain, rattled_simple_cubic


def _plan_args(root, atoms_path, *, description="toy-pair-v1"):
    return ["plan", "--workdir", str(root), "--structure", str(atoms_path),
            "--calculator", "kaldo.tests.colored_fd_helpers:ToyFiniteRange",
            "--calculator-description", description, "--delta-shift", "0.001",
            "--expected-atoms", "9", "--n-probe", "9", "--shards", "2"]


def test_staged_sparse_dynmat_and_resume(tmp_path):
    structure = tmp_path / "atoms.extxyz"
    atoms = chain(9)
    ase.io.write(structure, atoms)
    root = tmp_path / "colored"
    main(_plan_args(root, structure))
    config = json.loads((root / "run.json").read_text())
    assert config["color_radius_angstrom"] >= 2 * 1.3
    assert config["store_radius_angstrom"] <= config["color_radius_angstrom"]
    main(["run", "--workdir", str(root), "--index", "0"])
    with pytest.raises(ValueError, match="missing shard manifests"):
        main(["finalize", "--workdir", str(root)])
    chunks = list((root / "shard_0000_of_0002").glob("chunk_*.npz"))
    original_mtimes = [path.stat().st_mtime_ns for path in chunks]
    main(["run", "--workdir", str(root), "--index", "0"])
    assert [path.stat().st_mtime_ns for path in chunks] == original_mtimes
    main(["run", "--workdir", str(root), "--index", "1"])
    main(["finalize", "--workdir", str(root)])

    matrix_path = root / "dynmat_bsr.npz"
    matrix = load_npz(matrix_path)
    sparse_ifc = SparseSecondOrder.load(root / "merged")
    expected = sparse_ifc.mass_weighted_bsr(atoms.get_masses())
    assert matrix.format == "bsr"
    np.testing.assert_array_equal(matrix.indptr, expected.indptr)
    np.testing.assert_array_equal(matrix.indices, expected.indices)
    np.testing.assert_allclose(matrix.data, expected.data, rtol=0, atol=0)
    assert matrix.shape == (27, 27)
    assert not (root / "second.npy").exists()

    saved_mtime = matrix_path.stat().st_mtime_ns
    main(["finalize", "--workdir", str(root)])
    assert matrix_path.stat().st_mtime_ns == saved_mtime
    with pytest.raises(ValueError, match="configuration differs"):
        main(_plan_args(root, structure, description="changed-model"))


def test_plan_rejects_unexpected_atom_count(tmp_path):
    structure = tmp_path / "atoms.extxyz"
    ase.io.write(structure, chain(9))
    arguments = _plan_args(tmp_path / "colored", structure)
    arguments[arguments.index("--expected-atoms") + 1] = "10"
    with pytest.raises(ValueError, match="expected 10 atoms, found 9"):
        main(arguments)


def test_worker_rejects_changed_structure(tmp_path):
    structure = tmp_path / "atoms.extxyz"
    atoms = chain(9)
    ase.io.write(structure, atoms)
    root = tmp_path / "colored"
    main(_plan_args(root, structure))
    atoms.positions[0, 0] += 0.1
    ase.io.write(structure, atoms)
    with pytest.raises(ValueError, match="differ from the planned force input"):
        main(["run", "--workdir", str(root), "--index", "0"])


def test_plan_requires_cutoff_bound(tmp_path, monkeypatch):
    structure = tmp_path / "atoms.extxyz"
    ase.io.write(structure, chain(9))
    from kaldo.controllers.colored_fd import estimate as estimate_module

    monkeypatch.setattr(estimate_module, "resolve_cutoff_hint", lambda calculator, explicit: None)
    with pytest.raises(ValueError, match="finite calculator cutoff"):
        main(_plan_args(tmp_path / "colored", structure))


def test_periodic_many_body_gamma_matrix_has_translational_null_modes(tmp_path):
    structure = tmp_path / "periodic.extxyz"
    ase.io.write(structure, rattled_simple_cubic(6, seed=4))
    root = tmp_path / "periodic_run"
    main(["plan", "--workdir", str(root), "--structure", str(structure),
          "--calculator", "kaldo.tests.colored_fd_helpers:PeriodicToyEAM",
          "--delta-shift", "0.001", "--n-probe", "3", "--shards", "2"])
    for index in (0, 1):
        main(["run", "--workdir", str(root), "--index", str(index), "--flush-tasks", "32"])
    main(["finalize", "--workdir", str(root)])
    matrix = load_npz(root / "dynmat_bsr.npz")
    assert matrix.shape == (648, 648)
    np.testing.assert_allclose((matrix - matrix.T).data, 0, atol=1e-10)
    for axis in range(3):
        translation = np.zeros((216, 3))
        translation[:, axis] = 1
        np.testing.assert_allclose(matrix @ translation.ravel(), 0, atol=1e-10)
