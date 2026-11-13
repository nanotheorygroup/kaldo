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


def _stage2_layout(random):
    """Pair rows for 4 atoms: shared Cartesian rows, one partner with its own rows and one empty partner."""
    n_atoms = 4
    rows_of = []
    for atom in range(n_atoms):
        shared = np.sort(random.choice(3 * n_atoms, size=random.integers(3, 7), replace=False))
        rows_of += [shared, shared, shared]
    rows_of[4] = np.sort(random.choice(3 * n_atoms, size=5, replace=False))  # atom 1, y differs from x
    rows_of[8] = np.zeros(0, dtype=np.int64)  # atom 2, z has no pair rows
    j_ptr = np.concatenate(([0], np.cumsum([rows.size for rows in rows_of]))).astype(np.int64)
    return n_atoms, j_ptr, np.concatenate(rows_of).astype(np.int32)


@pytest.mark.parametrize("dtype, rtol", [(np.float64, 1e-12), (np.float32, 1e-5)], ids=["float64", "float32"])
def test_stage2_matches_dense_contraction(dtype, rtol):
    from concurrent.futures import ThreadPoolExecutor

    random = np.random.default_rng(616)
    n_atoms, j_ptr, pair_i = _stage2_layout(random)
    n_modes, block, width, trim = 3 * n_atoms, 5, 9, slice(2, 8)
    plan = _kernels().stage2_plan(j_ptr, pair_i, n_atoms)
    assert plan.reuse.tolist() == [False, True, True, False, False, False, False, True, False, False, True, True]
    alpha = random.normal(size=(n_atoms, n_atoms)) + 1j * random.normal(size=(n_atoms, n_atoms))
    initial = random.normal(size=(n_modes, block)) + 1j * random.normal(size=(n_modes, block))
    contracted = random.normal(size=(pair_i.size, width)).astype(dtype)[:, trim]  # row stride wider than the width
    planes = [np.full((n_modes, block, trim.stop - trim.start), np.nan, dtype=dtype) for _ in range(3)]
    with ThreadPoolExecutor(2) as pool:
        _kernels().stage2(plan, alpha, np.ascontiguousarray(initial.real), np.ascontiguousarray(initial.imag),
                          contracted, *planes, pool, 3)
    partner_j = np.repeat(np.arange(n_modes), np.diff(j_ptr))
    weights = initial[pair_i] * alpha[pair_i // 3, partner_j // 3][:, np.newaxis]
    expected = np.zeros((n_modes, block, contracted.shape[1]), dtype=np.complex128)
    np.add.at(expected, partner_j, np.einsum("pn,pw->pnw", weights, contracted))
    np.testing.assert_allclose(planes[0], expected.real, rtol=rtol, atol=rtol)
    np.testing.assert_allclose(planes[1], expected.imag, rtol=rtol, atol=rtol)
    np.testing.assert_array_equal(planes[2], planes[0] + planes[1])
    assert np.all(planes[0][8] == 0) and np.all(planes[2][8] == 0)


def test_stage2_plan_rejects_bad_layout():
    with pytest.raises(ValueError, match="j_ptr"):
        _kernels().stage2_plan(np.array([0, 2, 1, 3, 3, 3, 3]), np.zeros(3, dtype=np.int32), 2)
    with pytest.raises(ValueError, match="pair_i"):
        _kernels().stage2_plan(np.array([0, 1, 1, 1, 1, 1, 1]), np.array([6], dtype=np.int32), 2)


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


def _channel_target(channel, omega, omega_row):
    """The conserved ``omega''`` of a tile channel, in the kernels' floating-point order."""
    return omega_row - omega if channel == 2 else omega + (2 * channel - 1) * omega_row


def _split_job(seed: int = 615, swapped: bool = False) -> dict:
    """Random complex operands; frequencies in rad/ps sorted ascending per q. ``swapped`` adds the row side."""
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
                                  threshold, channels=(0, 1, 2) if swapped else (0, 1))
    selected = (tiles[:, 5] < offset + width) & (tiles[:, 6] > offset)
    tiles = tiles[selected].copy()
    np.maximum(tiles[:, 5], offset, out=tiles[:, 5])
    np.minimum(tiles[:, 6], offset + width, out=tiles[:, 6])
    populations = random.uniform(0.1, 2., size=(3, 2, n_modes))
    row_side = dict(pop_initial=populations[2], scales_rows=random.uniform(0.5, 1.5, n_modes)) if swapped else {}
    return dict(**row_side,
        tiles=tiles, rows_re=random.normal(size=(n_modes, n_modes)), rows_im=random.normal(size=(n_modes, n_modes)),
        inter_re=random.normal(size=(block, n_modes, width)), inter_im=random.normal(size=(block, n_modes, width)),
        column_offset=offset, omega_initial=omega_initial, omega_rows=omega_rows, omega_cols=omega_cols,
        physical_rows=physical_rows, physical_cols=physical_cols, pop_rows=populations[0], pop_cols=populations[1],
        hbar_factors=np.array([1., 0.7]), threshold=threshold, sigma=0.8,
        scales=random.uniform(0.5, 1.5, n_modes),
    )


def _split_reference(job: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Straightforward |A|^2 Gaussian reduction of each tile with complex NumPy, for both legs when swapped."""
    rows = job["rows_re"].astype(np.float64) + 1j * job["rows_im"]  # complex128 even for float32 operands
    inter = job["inter_re"].astype(np.float64) + 1j * job["inter_im"]
    n_populations, block, n_modes = job["pop_rows"].shape[0], inter.shape[0], rows.shape[0]
    reduced = np.zeros((n_populations, block, 2))
    reduced_rows = np.zeros((n_populations, n_modes, 2))
    counts = np.zeros((block, 3), dtype=np.int64)
    swapped = "pop_initial" in job
    sigma = job["sigma"]
    for slot, initial, channel, row_start, row_stop, column_start, column_stop in job["tiles"]:
        for row, column in itertools.product(range(row_start, row_stop), range(column_start, column_stop)):
            if not (job["physical_rows"][row] and job["physical_cols"][column]):
                continue
            difference = (_channel_target(channel, job["omega_initial"][initial], job["omega_rows"][row])
                          - job["omega_cols"][column])
            if abs(difference) >= job["threshold"]:
                continue
            amplitude = abs(rows[row] @ inter[slot, :, column - job["column_offset"]]) ** 2
            phase = np.exp(-difference ** 2 / sigma ** 2) / np.sqrt(np.pi * sigma ** 2)
            column_population = job["pop_cols"][:, column]
            if channel != 2:  # the initial mode's absorption (1) or decay (0)
                squared = amplitude / (job["omega_rows"][row] * job["omega_cols"][column]) * job["scales"][initial]
                row_population = job["pop_rows"][:, row]
                delta = row_population - column_population if channel else 0.5 * (1 + row_population
                                                                                    + column_population)
                reduced[:, slot, 0] += phase * delta
                reduced[:, slot, 1] += squared * job["hbar_factors"] * phase * delta
            if swapped and channel != 0:  # the row mode's absorption (1) or decay (2)
                squared = amplitude / (job["omega_initial"][initial] * job["omega_cols"][column])
                squared *= job["scales_rows"][row]
                initial_population = job["pop_initial"][:, initial]
                delta = (initial_population - column_population if channel == 1
                         else 0.5 * (1 + initial_population + column_population))
                reduced_rows[:, row, 0] += phase * delta
                reduced_rows[:, row, 1] += squared * job["hbar_factors"] * phase * delta
            counts[slot, channel] += 1
    return reduced, reduced_rows, counts


def _partner_major(job: dict) -> dict:
    """Return ``job`` with the production operand layout of ``project_shifted``.

    Stage 2 writes contiguous partner-major ``(n, block, w)`` planes; the reducer reads their ``(block, n, w)``
    transposed views (leading dimension ``block*w``) together with precomputed ``re + im`` sums.
    """
    job = dict(job)
    views = {}
    for name in ("inter_re", "inter_im"):
        plane = np.ascontiguousarray(job[name].transpose(1, 0, 2))
        views[name] = plane.transpose(1, 0, 2)
        assert not views[name].flags.c_contiguous and np.shares_memory(views[name], plane)
    job.update(views)
    job["inter_sum"] = np.ascontiguousarray((job["inter_re"] + job["inter_im"]).transpose(1, 0, 2)).transpose(1, 0, 2)
    job["rows_sum"] = job["rows_re"] + job["rows_im"]
    return job


@pytest.mark.parametrize("dtype, rtol", [(np.float64, 1e-12), (np.float32, 1e-5)], ids=["float64", "float32"])
@pytest.mark.parametrize("swapped", [False, True], ids=["initial-only", "swap-paired"])
@pytest.mark.parametrize("layout", ["block-major", "partner-major"])
def test_reduce_tiles_split_matches_numpy(layout, swapped, dtype, rtol):
    job = _split_job(swapped=swapped)
    for name in ("rows_re", "rows_im", "inter_re", "inter_im"):
        job[name] = job[name].astype(dtype)  # the reference reads the rounded values
    if layout == "partner-major":
        job = _partner_major(job)
    channels = {0, 1, 2} if swapped else {0, 1}
    assert job["tiles"].shape[0] > 0 and set(job["tiles"][:, 2]) == channels
    reduced, reduced_rows, counts = _kernels().reduce_tiles_split(**job)
    expected, expected_rows, expected_counts = _split_reference(job)
    assert counts.dtype == np.int64 and reduced.shape == expected.shape and reduced_rows.shape == expected_rows.shape
    np.testing.assert_array_equal(counts, expected_counts)
    assert np.all(expected_counts[:, sorted(channels)].sum(axis=0) > 0)
    np.testing.assert_allclose(reduced, expected, rtol=rtol, atol=1e-15)
    np.testing.assert_allclose(reduced_rows, expected_rows, rtol=rtol, atol=1e-15)
    assert np.any(expected_rows[..., 1] > 0) == swapped


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
    job = _split_job()
    job["inter_im"] = job["inter_im"].astype(np.float32)  # GEMM operands must share one dtype
    with pytest.raises(TypeError):
        _kernels().reduce_tiles_split(**job)
    job = _split_job(swapped=True)
    del job["pop_initial"], job["scales_rows"]
    with pytest.raises(ValueError, match="tile"):  # channel 2 without the row side
        _kernels().reduce_tiles_split(**job)


def _plan_tiles_definition(modes, omega_initial, omega_rows, omega_cols, physical_rows, physical_cols, threshold,
                           channels):
    """The vectorized ``searchsorted`` definition of ``plan_tiles`` (rows grouped in blocks of 128)."""
    plans = []
    padding = threshold * 1e-6 + 1e-12
    physical_before = np.concatenate(([0], np.cumsum(physical_cols, dtype=np.int64)))
    for slot, initial in enumerate(modes):
        for channel in channels:
            targets = _channel_target(channel, omega_initial[initial], omega_rows)
            lower = np.searchsorted(omega_cols, targets - threshold - padding, side="left")
            upper = np.searchsorted(omega_cols, targets + threshold + padding, side="right")
            rows = np.flatnonzero((physical_before[upper] > physical_before[lower]) & physical_rows)
            for offset in range(0, rows.size, 128):
                row_start, row_stop = rows[offset], rows[min(offset + 128, rows.size) - 1] + 1
                plans.append((slot, initial, channel, row_start, row_stop, min(lower[row_start], lower[row_stop - 1]),
                              max(upper[row_start], upper[row_stop - 1])))
    return np.asarray(plans, dtype=np.int64).reshape(-1, 7)


@pytest.mark.parametrize("channels", [(0, 1), (0, 1, 2)])
@pytest.mark.parametrize("threshold", [0.0, 0.05, 0.9, 30.])
def test_plan_tiles_equals_searchsorted_definition(threshold, channels):
    random = np.random.default_rng(617)
    n_modes = 700
    omega_initial = np.sort(random.uniform(0., 20., n_modes))
    omega_rows = np.sort(random.uniform(0., 20., n_modes))
    omega_cols = np.sort(random.uniform(0., 20., n_modes))
    omega_cols[100:110] = omega_cols[100]  # degenerate columns
    omega_rows[200:205] = omega_rows[200]  # degenerate rows
    omega_cols[300] = omega_initial[350] + omega_rows[20] + 0.9  # a column exactly on a window edge
    omega_cols.sort()
    physical_rows = random.random(n_modes) > 0.1
    physical_cols = random.random(n_modes) > 0.1
    physical_cols[400:450] = False
    modes = np.array([0, 3, 77, 350, 699], dtype=np.int64)
    arguments = (modes, omega_initial, omega_rows, omega_cols, physical_rows, physical_cols, threshold, channels)
    tiles = _kernels().plan_tiles(*arguments)
    expected = _plan_tiles_definition(*arguments)
    assert tiles.dtype == np.int64 and tiles.shape[1] == 7
    if threshold > 1:
        assert np.any(tiles[:, 4] - tiles[:, 3] >= 128), "row blocks of 128 must be exercised"
    np.testing.assert_array_equal(tiles, expected)


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
                                  threshold, channels=(0, 1, 2))
    assert tiles.dtype == np.int64 and tiles.shape[1] == 7
    cover = np.zeros((modes.size, 3, n_modes, n_modes), dtype=np.int64)
    for slot, initial, channel, row_start, row_stop, column_start, column_stop in tiles:
        assert initial == modes[slot]
        assert np.count_nonzero(physical_rows[row_start:row_stop]) <= 128
        cover[slot, channel, row_start:row_stop, column_start:column_stop] += 1
    for slot, initial in enumerate(modes):
        for channel in (0, 1, 2):
            difference = _channel_target(channel, omega_initial[initial], omega_rows[:, None]) - omega_cols[None, :]
            selected = (np.abs(difference) < threshold) & physical_rows[:, None] & physical_cols[None, :]
            assert selected.any()
            np.testing.assert_array_equal(cover[slot, channel][selected], 1)
