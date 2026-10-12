"""The large-cell command line exposes every stage and converts legacy harmonic input."""

import numpy as np
import pytest

from kaldo.large_cell_cli import main
from kaldo.observables.sparse_harmonic import SparseHarmonic


def test_help_lists_every_stage(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    usage = capsys.readouterr().out
    for command in ("ifc-plan", "ifc-run", "ifc-finalize", "convert", "shard", "reduce", "solve", "windows"):
        assert command in usage


def test_convert_writes_a_loadable_sparse_harmonic(tmp_path):
    rng = np.random.default_rng(3)
    n = 3
    positions = rng.normal(size=(n, 3))
    coupling = rng.normal(size=(3 * n, 3 * n)) * 0.01
    dense = coupling + coupling.T
    np.save(tmp_path / "d.npy", dense)
    np.save(tmp_path / "positions.npy", positions)
    main(["convert", str(tmp_path / "d.npy"), str(tmp_path / "positions.npy"), str(tmp_path / "sparse")])
    harmonic = SparseHarmonic.load(tmp_path / "sparse")
    assert harmonic.blocks.dtype == np.float64
    np.testing.assert_array_equal(harmonic.blocks.reshape(n, n, 3, 3).transpose(0, 2, 1, 3).reshape(3 * n, 3 * n),
                                  dense)
