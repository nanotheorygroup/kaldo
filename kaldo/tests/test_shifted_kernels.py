"""Factorized Wigner-Seitz IFC3 leg tables and shifted-grid projection kernels."""

import itertools

from ase import Atoms
import numpy as np
import pytest
import scipy.sparse

from kaldo.forceconstants import ForceConstants
from kaldo.grid import SupercellGrid, TranslationSupport
from kaldo.observables.thirdorder import ThirdOrder

_SHIFTED = np.array(list(itertools.product((0.25, -0.25), repeat=3)), dtype=np.float64)
_GAMMA_GRID = np.array(list(itertools.product((0.0, 0.5), repeat=3)), dtype=np.float64)


def _kernels():
    import kaldo.controllers._shifted_kernels as kernels
    return kernels


@pytest.fixture(scope="module")
def amorphous_third() -> ThirdOrder:
    """a-Si 216-atom IFC3 in eV/Angstrom^3 with one source translation."""
    forceconstants = ForceConstants.from_folder(folder="kaldo/tests/si-amorphous", format="eskm")
    third = forceconstants.third
    assert third.translation_support.size == 1
    assert third.translation_support.supercell.size == 1
    return third


def _tied_third() -> ThirdOrder:
    """Single-cell IFC3 whose atom pairs have 1, 2, 4 and 8 tied shortest images."""
    fractional = np.array([[0., 0., 0.], [0.5, 0., 0.], [0.5, 0.5, 0.], [0.5, 0.5, 0.5]])
    atoms = Atoms("H4", scaled_positions=fractional, cell=3.0 * np.eye(3), pbc=True)
    grid = SupercellGrid(np.eye(3, dtype=int))
    random = np.random.default_rng(611)
    value = random.normal(size=(4, 3, 1, 4, 3, 1, 4, 3))
    value[random.random(value.shape) < 0.3] = 0.
    return ThirdOrder(
        atoms=atoms, replicated_positions=atoms.positions[None], supercell=(1, 1, 1), folder="",
        value=value, supercell_grid=grid, translation_support=TranslationSupport.periodic(grid),
    )


def _layout_entries(layout) -> tuple[np.ndarray, ...]:
    """Return Cartesian ``(i,j,k)`` and eV/Angstrom^3 values of every CSR entry."""
    partner_j = np.repeat(np.arange(layout.n_modes, dtype=np.int64), np.diff(layout.j_ptr))
    rows_i, rows_j, columns, values = [], [], [], []
    for start, stop, matrix in layout.chunks:
        counts = np.diff(matrix.indptr)
        rows_i.append(np.repeat(layout.pair_i[start:stop].astype(np.int64), counts))
        rows_j.append(np.repeat(partner_j[start:stop], counts))
        columns.append(matrix.indices.astype(np.int64))
        values.append(matrix.data)
    return tuple(np.concatenate(parts) for parts in (rows_i, rows_j, columns, values))


@pytest.mark.parametrize("source", ["amorphous", "tied"])
def test_factorized_ifc3_matches_compiled_wigner_seitz(source, amorphous_third):
    third = amorphous_third if source == "amorphous" else _tied_third()
    layout = third._gamma_pair_csr()
    alpha, beta = _kernels().leg_tables(third, layout, _SHIFTED, _GAMMA_GRID, "wigner-seitz")
    assert alpha.dtype == np.complex128 and beta.dtype == np.float64
    random = np.random.default_rng(612)
    u, v, x = random.normal(size=(3, layout.n_modes))

    i, j, k, values = _layout_entries(layout)
    weighted = values * u[i] * v[j] * x[k]
    factorized = np.empty((len(_SHIFTED), len(_GAMMA_GRID)), dtype=np.complex128)
    for a, b in itertools.product(range(len(_SHIFTED)), range(len(_GAMMA_GRID))):
        factorized[a, b] = np.sum(weighted * alpha[a, i // 3, j // 3] * beta[b, i // 3, k // 3])

    compiled = third.get_interpolation("wigner-seitz")
    coords = compiled.value.coords.astype(np.int64)
    compiled_i, compiled_j, compiled_k = (3 * coords[0] + coords[1], 3 * coords[3] + coords[4],
                                          3 * coords[6] + coords[7])
    size = compiled.support.size
    translation_sums = np.bincount(
        coords[2] * size + coords[5], weights=compiled.value.data * u[compiled_i] * v[compiled_j] * x[compiled_k],
        minlength=size * size,
    ).reshape(size, size)
    reference = compiled.support.phases(_SHIFTED) @ translation_sums @ compiled.support.phases(_GAMMA_GRID).conj().T

    bound = np.sum(np.abs(weighted))
    np.testing.assert_allclose(factorized, reference, rtol=1e-12, atol=1e-14 * bound)
    a = int(np.flatnonzero(np.all(np.isclose(_SHIFTED, [0.25, 0.25, -0.25], rtol=0, atol=1e-15), axis=1))[0])
    b = int(np.flatnonzero(np.all(np.isclose(_GAMMA_GRID, [0.5, 0., 0.], rtol=0, atol=1e-15), axis=1))[0])
    assert abs(reference[a, b]) > 1e-6 * bound


def test_tied_images_give_fractional_leg_weights():
    third = _tied_third()
    alpha, beta = _kernels().leg_tables(third, third._gamma_pair_csr(), _SHIFTED, _GAMMA_GRID, "auto")
    # Atom 1 sits half a cell along x from atom 0: images R=0 and R=-x tie, so
    # beta_01(q'') = (1 + exp(2 pi i q''_x)) / 2 is 0 or 1 on the Gamma grid.
    np.testing.assert_allclose(beta[:, 0, 1], (1 + np.cos(2 * np.pi * _GAMMA_GRID[:, 0])) / 2, rtol=0, atol=1e-15)
    np.testing.assert_allclose(np.abs(alpha[:, 0, 3]), np.full(8, 2 ** -1.5), rtol=1e-14, atol=0)


def test_beta_is_real_on_gamma_grid(amorphous_third):
    layout = amorphous_third._gamma_pair_csr()
    alpha, beta = _kernels().leg_tables(amorphous_third, layout, _SHIFTED, _GAMMA_GRID, "wigner-seitz")
    assert alpha.shape == (8, 216, 216) and beta.shape == (8, 216, 216)
    assert beta.dtype == np.float64
    # At Gamma the tied weights of every present atom pair sum to one.
    i, j, k, _ = _layout_entries(layout)
    present = np.zeros((216, 216), dtype=bool)
    present[i // 3, j // 3] = present[i // 3, k // 3] = True
    np.testing.assert_allclose(beta[0][present], 1., rtol=0, atol=1e-14)
    assert np.all(np.abs(beta[:, present]) <= 1 + 1e-14)
    assert np.any(beta[1:, present] < 0)
    with pytest.raises(ValueError, match="real"):
        _kernels().leg_tables(amorphous_third, layout, _SHIFTED, _SHIFTED, "wigner-seitz")


@pytest.mark.parametrize("mode", ["periodic", "auto-file"])
def test_unfolded_modes_give_unit_legs(mode):
    third = _tied_third()
    if mode == "auto-file":
        third.translation_support = TranslationSupport(
            np.zeros((1, 3), dtype=np.int64), third.translation_support.supercell, provenance="file",
        )
    alpha, beta = _kernels().leg_tables(
        third, third._gamma_pair_csr(), _SHIFTED, _GAMMA_GRID, "periodic" if mode == "periodic" else "auto",
    )
    np.testing.assert_array_equal(alpha, np.ones((8, 4, 4), dtype=np.complex128))
    np.testing.assert_array_equal(beta, np.ones((8, 4, 4), dtype=np.float64))


def test_literal_nonzero_file_translation_is_rejected():
    third = _tied_third()
    third.translation_support = TranslationSupport(
        np.array([[1, 0, 0]], dtype=np.int64), third.translation_support.supercell, provenance="file",
    )
    with pytest.raises(ValueError, match="literal"):
        _kernels().leg_tables(third, third._gamma_pair_csr(), _SHIFTED, _GAMMA_GRID, "auto")


def test_stage1_matches_scipy():
    random = np.random.default_rng(613)
    n_atoms, n_pairs, width = 7, 40, 9
    n_modes = 3 * n_atoms
    matrix = scipy.sparse.random(n_pairs, n_modes, density=0.3, format="csr", random_state=614, dtype=np.float64)
    matrix.data[:] = random.normal(size=matrix.nnz)
    matrix.indptr[5] = matrix.indptr[4]  # an empty pair row
    matrix = scipy.sparse.csr_matrix((matrix.data, matrix.indices, matrix.indptr), shape=matrix.shape)
    pair_atom_i = random.integers(0, n_atoms, size=n_pairs).astype(np.int32)
    beta_b = random.normal(size=(n_atoms, n_atoms))
    columns = random.normal(size=(n_modes, width))
    out = np.full((n_pairs, width), np.nan)
    for start, stop in ((0, 13), (13, 14), (14, n_pairs)):
        chunk = matrix[start:stop]
        _kernels().stage1(chunk.indptr, chunk.indices, chunk.data, start, pair_atom_i, beta_b, columns, out)
    dense = matrix.toarray() * beta_b[pair_atom_i][:, np.arange(n_modes) // 3]
    np.testing.assert_allclose(out, dense @ columns, rtol=1e-13, atol=1e-14)


def test_stage1_rejects_mismatched_operands():
    columns = np.ones((6, 2))
    out = np.zeros((1, 2))
    arguments = (np.array([0, 1], dtype=np.int32), np.array([0], dtype=np.int32), np.ones(1), 0,
                 np.zeros(1, dtype=np.int32), np.ones((2, 2)))
    with pytest.raises((TypeError, ValueError)):
        _kernels().stage1(*arguments, columns.astype(np.float32), out)
    with pytest.raises((TypeError, ValueError)):
        _kernels().stage1(*arguments, columns, np.zeros((1, 3)))
    with pytest.raises((TypeError, ValueError)):
        _kernels().stage1(*arguments[:3], 1, *arguments[4:], columns, out)
    for k in (6, -1):  # one past the last Cartesian column, and negative
        indices = np.array([0, 1, 2, k], dtype=np.int32)
        with pytest.raises(ValueError, match="invalid"):
            _kernels().stage1(np.array([0, 4], dtype=np.int32), indices, np.ones(4), *arguments[3:], columns, out)


def _split_job(seed: int = 615) -> dict:
    """Random complex operands; frequencies in rad/ps sorted ascending per q."""
    random = np.random.default_rng(seed)
    n_modes, block, width, offset = 12, 3, 8, 3
    omega_initial = np.sort(random.uniform(0.5, 6., n_modes))
    omega_rows = np.sort(random.uniform(0.5, 6., n_modes))
    omega_cols = np.sort(random.uniform(0.5, 6., n_modes))
    physical_rows = np.ones(n_modes, dtype=bool)
    physical_cols = np.ones(n_modes, dtype=bool)
    physical_rows[[0, 5]] = False
    physical_cols[[1, 7]] = False
    modes = np.array([2, 6, 10], dtype=np.int64)
    threshold = 1.5
    tiles = _kernels().plan_tiles(modes, omega_initial, omega_rows, omega_cols, physical_rows, physical_cols,
                                  threshold)
    selected = (tiles[:, 5] < offset + width) & (tiles[:, 6] > offset)
    tiles = tiles[selected].copy()
    np.maximum(tiles[:, 5], offset, out=tiles[:, 5])
    np.minimum(tiles[:, 6], offset + width, out=tiles[:, 6])
    populations = random.uniform(0.1, 2., size=(2, 2, n_modes))
    return dict(
        tiles=tiles, rows_re=random.normal(size=(n_modes, n_modes)), rows_im=random.normal(size=(n_modes, n_modes)),
        inter_re=random.normal(size=(block, n_modes, width)), inter_im=random.normal(size=(block, n_modes, width)),
        column_offset=offset, omega_initial=omega_initial, omega_rows=omega_rows, omega_cols=omega_cols,
        physical_rows=physical_rows, physical_cols=physical_cols, pop_rows=populations[0], pop_cols=populations[1],
        hbar_factors=np.array([1., 0.7]), threshold=threshold, sigma=0.8,
        scales=random.uniform(0.5, 1.5, n_modes),
    )


def _split_reference(job: dict) -> tuple[np.ndarray, np.ndarray]:
    """Straightforward |A|^2 Gaussian reduction of each tile with complex NumPy."""
    rows = job["rows_re"] + 1j * job["rows_im"]
    inter = job["inter_re"] + 1j * job["inter_im"]
    n_populations, block = job["pop_rows"].shape[0], inter.shape[0]
    reduced = np.zeros((n_populations, block, 2))
    counts = np.zeros((block, 2), dtype=np.int64)
    sigma = job["sigma"]
    for slot, initial, channel, row_start, row_stop, column_start, column_stop in job["tiles"]:
        for row, column in itertools.product(range(row_start, row_stop), range(column_start, column_stop)):
            if not (job["physical_rows"][row] and job["physical_cols"][column]):
                continue
            sign = 1 if channel else -1
            difference = job["omega_initial"][initial] + sign * job["omega_rows"][row] - job["omega_cols"][column]
            if abs(difference) >= job["threshold"]:
                continue
            amplitude = rows[row] @ inter[slot, :, column - job["column_offset"]]
            phase = np.exp(-difference ** 2 / sigma ** 2) / np.sqrt(np.pi * sigma ** 2)
            squared = abs(amplitude) ** 2 / (job["omega_rows"][row] * job["omega_cols"][column])
            squared *= job["scales"][initial]
            for index in range(n_populations):
                row_population, column_population = job["pop_rows"][index, row], job["pop_cols"][index, column]
                if channel:
                    delta = row_population - column_population
                else:
                    delta = 0.5 * (1 + row_population + column_population)
                reduced[index, slot, 0] += phase * delta
                reduced[index, slot, 1] += squared * job["hbar_factors"][index] * phase * delta
            counts[slot, channel] += 1
    return reduced, counts


def test_reduce_tiles_split_matches_numpy():
    job = _split_job()
    assert job["tiles"].shape[0] > 0 and set(job["tiles"][:, 2]) == {0, 1}
    reduced, counts = _kernels().reduce_tiles_split(**job)
    expected, expected_counts = _split_reference(job)
    assert counts.dtype == np.int64 and reduced.shape == expected.shape
    np.testing.assert_array_equal(counts, expected_counts)
    assert np.all(expected_counts.sum(axis=0) > 0)
    np.testing.assert_allclose(reduced, expected, rtol=1e-12, atol=1e-15)


def test_reduce_tiles_split_rejects_bad_tiles():
    job = _split_job()
    job["tiles"] = job["tiles"].copy()
    job["tiles"][0, 6] = 13
    with pytest.raises(ValueError, match="tile"):
        _kernels().reduce_tiles_split(**job)
    job = _split_job()
    job["rows_im"] = np.asfortranarray(job["rows_im"])
    with pytest.raises((TypeError, ValueError)):
        _kernels().reduce_tiles_split(**job)


def test_plan_tiles_encloses_every_selected_pair():
    random = np.random.default_rng(616)
    n_modes = 300
    omega_initial = np.sort(random.uniform(0., 20., n_modes))
    omega_rows = np.sort(random.uniform(0., 20., n_modes))
    omega_cols = np.sort(random.uniform(0., 20., n_modes))
    omega_cols[100:104] = omega_cols[100]  # degenerate columns
    physical_rows = random.random(n_modes) > 0.1
    physical_cols = random.random(n_modes) > 0.1
    physical_cols[150:170] = False  # a hole in the middle of the spectrum
    modes = np.array([3, 77, 150, 299], dtype=np.int64)
    threshold = 0.9
    tiles = _kernels().plan_tiles(modes, omega_initial, omega_rows, omega_cols, physical_rows, physical_cols,
                                  threshold)
    assert tiles.dtype == np.int64 and tiles.shape[1] == 7
    cover = np.zeros((modes.size, 2, n_modes, n_modes), dtype=np.int64)
    for slot, initial, channel, row_start, row_stop, column_start, column_stop in tiles:
        assert initial == modes[slot]
        assert np.count_nonzero(physical_rows[row_start:row_stop]) <= 128
        cover[slot, channel, row_start:row_stop, column_start:column_stop] += 1
    for slot, initial in enumerate(modes):
        for channel in (0, 1):
            difference = (omega_initial[initial] + (2 * channel - 1) * omega_rows[:, None]) - omega_cols[None, :]
            selected = (np.abs(difference) < threshold) & physical_rows[:, None] & physical_cols[None, :]
            assert selected.any()
            np.testing.assert_array_equal(cover[slot, channel][selected], 1)
