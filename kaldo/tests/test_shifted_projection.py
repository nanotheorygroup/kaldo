"""Shifted-grid three-phonon projection versus the dense literal reference on a-Si 216.

The route evaluates one absorption-form pass per ``(q, q')`` with ``q'' = q + q'`` on the Gamma-centred partner
grid and applies both channel selections to it; non-representative initial points are copied from their time
reversal partner. The reference (:mod:`kaldo.tests._three_phonon_reference`) instead evaluates kALDo's crystal
formulas literally, each channel with its own ``q''`` and conjugations, so agreement tests the channel pairing,
the factorized Wigner-Seitz legs, the conjugation conventions and the normalization together. The exact
configuration (``EXACT``: every pass evaluated, float64 GEMMs) is held to rounding; swap pairing reads the q'
linewidths off the transposed amplitude, which differs from the literal one by the IFC3's i <-> j asymmetry
(measured <= 4e-10 relative in the linewidths of a-Si 216), and the default float32 GEMMs moved them by <= 2e-7.
"""

import numpy as np
import pytest

from kaldo.controllers._bloch_eigensystem import population as bose_population
from kaldo.forceconstants import ForceConstants
from kaldo.observables.harmonic_with_q_temp import CLASSICAL_HBAR_SCALE
from kaldo.phonons import Phonons
from kaldo.tests._three_phonon_reference import reference_ps_and_gamma

THIRD_BANDWIDTH_THZ = 0.05
TEMPERATURE = 300
MODE = "auto"
INITIAL_Q = (0, 3)
INITIAL_MODES = np.array([5, 100, 333, 640], dtype=np.int64)
HOLE_MIN_FREQUENCY = 2.0
EXACT = dict(pair_swapped=False, precision="float64")
PAIRED = dict(pair_swapped=True, precision="float64")
SWAP_RTOL = 1e-8  # IFC3 i <-> j asymmetry bound, 25x the a-Si 216 maximum
FLOAT32_RTOL = 1e-5  # accepted linewidth error of float32 GEMMs (a-Si 216 maximum 2e-7)


class _Case:
    """Shifted harmonics, grid and both statistics' occupations of one a-Si 216 ``Phonons``."""

    def __init__(self, forceconstants, min_frequency=None):
        phonons = Phonons(
            forceconstants=forceconstants, kpts=(2, 2, 2), kpts_shift=(0.5, 0.5, 0.5), temperature=TEMPERATURE,
            is_classic=False, third_bandwidth=THIRD_BANDWIDTH_THZ, min_frequency=min_frequency, storage="memory",
        )
        self.third = forceconstants.third
        self.grid = phonons._reciprocal_grid
        self.harmonics = phonons._shifted_harmonics
        self.partner = self.harmonics.partner()
        self.populations = {}
        for is_classic in (False, True):
            self.populations[is_classic] = (
                bose_population(self.harmonics.frequency, self.harmonics.physical, TEMPERATURE, is_classic),
                bose_population(self.partner.frequency, self.partner.physical, TEMPERATURE, is_classic),
            )
        self._routes = {}

    def route(self, is_classic=False, n_workers=4, all_initial=False, **options):
        """Run (once) and return the ``(8*n, 2)`` route result; ``options`` go to ``project_shifted``."""
        key = (is_classic, n_workers, all_initial, tuple(sorted(options.items())))
        if key not in self._routes:
            from kaldo.controllers._shifted_projection import project_shifted

            population, partner_population = self.populations[is_classic]
            self._routes[key] = project_shifted(
                self.third, self.harmonics, self.grid, MODE, THIRD_BANDWIDTH_THZ, population, partner_population,
                _hbar_factor(is_classic), n_workers=n_workers, all_initial=all_initial, **options,
            )
        return self._routes[key]

    def reference(self, is_classic, index_k):
        """Literal reference ``(len(INITIAL_MODES), 2)`` at shifted point ``index_k``."""
        shifted_points = np.asarray(self.grid.fractional_points, dtype=np.float64)
        partner_points = np.asarray(self.grid.partner_grid().fractional_points, dtype=np.float64)
        masses = np.asarray(self.third.atoms.get_masses(), dtype=np.float64)
        inverse_sqrt_mass = np.repeat(1 / np.sqrt(masses), 3)[np.newaxis, :, np.newaxis]
        population, partner_population = self.populations[is_classic]
        tables = (
            (shifted_points, dict(
                vectors=np.asarray(self.harmonics.eigenvectors, dtype=np.complex128) * inverse_sqrt_mass,
                omega=2 * np.pi * self.harmonics.frequency, physical=self.harmonics.physical,
                population=population,
            )),
            (partner_points, dict(
                vectors=np.asarray(self.partner.eigenvectors, dtype=np.complex128) * inverse_sqrt_mass,
                omega=2 * np.pi * self.partner.frequency, physical=self.partner.physical,
                population=partner_population,
            )),
        )

        def lookup(name):
            def value(q):
                matches = []
                for points, data in tables:
                    distance = np.mod(points - np.asarray(q, dtype=np.float64) + 0.5, 1.0) - 0.5
                    matches += [data[name][index] for index in np.flatnonzero(np.all(np.abs(distance) < 1e-9, 1))]
                assert len(matches) == 1, f"q={q} is not a unique point of the shifted or partner grid"
                return matches[0]
            return value

        def partner_lookup(q, q_prime, is_plus):
            sign = 1.0 if is_plus else -1.0
            return np.mod(np.asarray(q, dtype=np.float64) + sign * np.asarray(q_prime, dtype=np.float64), 1.0)

        return reference_ps_and_gamma(
            self.third, MODE, shifted_points[index_k], INITIAL_MODES, shifted_points,
            vectors=lookup("vectors"), omega=lookup("omega"), physical=lookup("physical"),
            population=lookup("population"), partner_lookup=partner_lookup, sigma_thz=THIRD_BANDWIDTH_THZ,
            hbar_factor=_hbar_factor(is_classic), n_k=len(shifted_points),
        )


def _hbar_factor(is_classic):
    return CLASSICAL_HBAR_SCALE if is_classic else 1.0


@pytest.fixture(scope="module")
def forceconstants():
    return ForceConstants.from_folder(folder="kaldo/tests/si-amorphous", format="eskm")


@pytest.fixture(scope="module")
def case(forceconstants):
    return _Case(forceconstants)


@pytest.fixture(scope="module")
def hole_case(forceconstants):
    return _Case(forceconstants, min_frequency=HOLE_MIN_FREQUENCY)


def _rows(case, index_k):
    n_modes = case.harmonics.frequency.shape[1]
    return index_k * n_modes + INITIAL_MODES


@pytest.mark.parametrize("is_classic", [False, True], ids=["quantum", "classical"])
def test_route_matches_reference(case, is_classic):
    route = case.route(is_classic, **EXACT)
    n_modes = case.harmonics.frequency.shape[1]
    assert route.shape == (8 * n_modes, 2) and route.dtype == np.float64
    for index_k in INITIAL_Q:
        ref = case.reference(is_classic, index_k)
        assert np.all(ref > 0), "every selected initial mode must scatter, or the comparison is vacuous"
        np.testing.assert_allclose(route[_rows(case, index_k)], ref, rtol=1e-10, atol=1e-14)


def test_default_route_matches_reference(case):
    route = case.route()
    for index_k in INITIAL_Q:
        np.testing.assert_allclose(route[_rows(case, index_k)], case.reference(False, index_k), rtol=FLOAT32_RTOL,
                                   atol=1e-14)


def test_swap_pairing_matches_every_pass(case):
    exact, paired = case.route(**EXACT), case.route(**PAIRED)
    n_modes = case.harmonics.frequency.shape[1]
    assert np.count_nonzero(exact[:, 1]) > 0.9 * 8 * n_modes
    np.testing.assert_array_equal(paired == 0, exact == 0)
    np.testing.assert_allclose(paired[:, 0], exact[:, 0], rtol=1e-12, atol=0)  # selection is identical
    np.testing.assert_allclose(paired[:, 1], exact[:, 1], rtol=SWAP_RTOL, atol=0)


def test_route_matches_reference_with_mask_holes(hole_case):
    harmonics, partner = hole_case.harmonics, hole_case.partner
    assert not harmonics.physical[:, INITIAL_MODES[0]].any(), "a masked initial mode must be exercised"
    assert (~harmonics.physical).sum() > 8 and (~partner.physical).sum() > 8 * 3
    route = hole_case.route()
    for index_k in INITIAL_Q:
        ref = hole_case.reference(False, index_k)
        assert np.all(ref[1:] > 0)
        np.testing.assert_array_equal(ref[0], 0.)
        np.testing.assert_allclose(route[_rows(hole_case, index_k)], ref, rtol=FLOAT32_RTOL, atol=1e-14)
    n_modes = harmonics.frequency.shape[1]
    np.testing.assert_array_equal(route.reshape(8, n_modes, 2)[~harmonics.physical], 0.)


@pytest.mark.parametrize("all_initial, expected", [(False, (32, 20, 8)), (True, (64, 36, 8))],
                         ids=["representatives", "all-initial"])
def test_swap_pairing_covers_every_pass_once(case, all_initial, expected):
    from kaldo.controllers._shifted_projection import _passes

    initial_points = np.arange(8) if all_initial else np.asarray(case.harmonics.representatives)
    every = _passes(case.grid, initial_points, 8, pair_swapped=False)
    paired = _passes(case.grid, initial_points, 8, pair_swapped=True)
    assert (len(every), len(paired), sum(mirror is None for *_, mirror in paired)) == expected
    covered = [(q, b) for q, b, _, _ in paired] + [(mirror, b) for _, b, _, mirror in paired if mirror is not None]
    assert sorted(covered) == sorted((q, b) for q, b, _, _ in every)
    initial_of = {point: point if point in initial_points else case.grid.partner_id(point) for point in range(8)}
    assert all(mirror == initial_of[q_prime] for _, _, q_prime, mirror in paired if mirror is not None)


@pytest.mark.performance  # projects twice the passes, or on one worker (minutes)
@pytest.mark.parametrize("options, rtol", [(EXACT, 1e-12), (PAIRED, SWAP_RTOL)], ids=["exact", "paired"])
def test_time_reversal_replication_matches_full_projection(case, options, rtol):
    replicated = case.route(**options)
    full = case.route(all_initial=True, **options)
    n_modes = case.harmonics.frequency.shape[1]
    replicas = sorted(set(range(8)) - set(case.harmonics.representatives.tolist()))
    assert len(replicas) == 4
    assert np.all(full.reshape(8, n_modes, 2)[replicas][:, INITIAL_MODES] > 0)
    np.testing.assert_allclose(full, replicated, rtol=rtol, atol=0)


@pytest.mark.performance  # projects twice the passes, or on one worker (minutes)
def test_workers_do_not_change_results(case):
    np.testing.assert_allclose(case.route(n_workers=1), case.route(n_workers=4), rtol=1e-12, atol=0)


def test_population_stack_matches_single_runs(case):
    from kaldo.controllers._shifted_projection import project_shifted

    statistics = (False, True)
    population = np.stack([case.populations[is_classic][0] for is_classic in statistics])
    partner_population = np.stack([case.populations[is_classic][1] for is_classic in statistics])
    hbar_factors = np.array([_hbar_factor(is_classic) for is_classic in statistics], dtype=np.float64)
    stacked = project_shifted(
        case.third, case.harmonics, case.grid, MODE, THIRD_BANDWIDTH_THZ, population, partner_population,
        hbar_factors, n_workers=4,
    )
    n_modes = case.harmonics.frequency.shape[1]
    assert stacked.shape == (2, 8 * n_modes, 2)
    for index, is_classic in enumerate(statistics):
        np.testing.assert_allclose(stacked[index], case.route(is_classic), rtol=1e-12, atol=0)
