"""A merged colored finite-difference IFC2 feeds the existing harmonic API and the sparse QHGK input consistently."""

import ase.io
import numpy as np
import pytest
from scipy.sparse import load_npz

from kaldo.forceconstants import ForceConstants
from kaldo.controllers.colored_fd.build_dynmat import main
from kaldo.observables.harmonic_with_q import HarmonicWithQ
from kaldo.observables.secondorder import SecondOrder
from kaldo.observables.sparse_harmonic import SparseHarmonic
from kaldo.phonons import Phonons
from kaldo.tests.colored_fd_helpers import rattled_simple_cubic


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    root = tmp_path_factory.mktemp("colored_bridge")
    structure = root / "periodic.extxyz"
    ase.io.write(structure, rattled_simple_cubic(6, seed=4))
    run = root / "run"
    main(["plan", "--workdir", str(run), "--structure", str(structure),
          "--calculator", "kaldo.tests.colored_fd_helpers:PeriodicToyEAM",
          "--delta-shift", "0.001", "--n-probe", "3", "--shards", "2"])
    for index in (0, 1):
        main(["run", "--workdir", str(run), "--index", str(index), "--flush-tasks", "32"])
    main(["finalize", "--workdir", str(run)])
    return run


def test_merged_ifc_loads_through_forceconstants(workdir):
    forceconstants = ForceConstants.from_folder(str(workdir / "merged"), format="colored", only_second=True)
    phonons = Phonons(forceconstants=forceconstants, storage="memory")
    eigenvalues = np.linalg.eigvalsh(load_npz(workdir / "dynmat_bsr.npz").toarray())
    expected = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)
    np.testing.assert_allclose(phonons.frequency.reshape(-1), expected, rtol=0,
                               atol=1e-8 * np.max(np.abs(expected)))


def test_sparse_harmonic_matches_existing_amorphous_derivatives(workdir):
    sparse = SparseHarmonic.load(workdir / "harmonic")
    second = SecondOrder.load(folder=str(workdir / "merged"), format="colored")
    harmonic = HarmonicWithQ(q_point=np.zeros(3), second=second, storage="memory", is_amorphous=True)
    panel = np.random.default_rng(7).normal(size=(3 * sparse.n_atoms, 5))
    for axis, operator in zip("xyz", sparse.operators):
        reference = np.asarray(getattr(harmonic, f"_dynmat_derivatives_{axis}"))
        assert np.max(np.abs(reference)) > 0
        np.testing.assert_allclose(operator @ panel, reference @ panel, rtol=0,
                                   atol=1e-10 * np.max(np.abs(reference @ panel)))


def test_finalize_reuses_the_sparse_harmonic(workdir):
    metadata = workdir / "harmonic" / "metadata.json"
    before = metadata.stat().st_mtime_ns
    main(["finalize", "--workdir", str(workdir)])
    assert metadata.stat().st_mtime_ns == before
