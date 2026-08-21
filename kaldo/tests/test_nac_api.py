import numpy as np
import pytest
from ase.build import bulk

from kaldo.forceconstants import ForceConstants
from kaldo.controllers.nac import bvk_supercell_matrix_key
from kaldo.observables.harmonic_with_q import HarmonicWithQ, _resolve_nac_activation


def nacl_phonopy_debug_supercell_matrix():
    return np.array([[-2, 2, 2], [2, -2, 2], [2, 2, -2]], dtype=int)


def nacl_phonopy_debug_supercell_matrix_att3():
    return np.diag([8, 8, 8]).astype(int)


@pytest.fixture
def nac_second_order(tmp_path_factory):
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/nacl_phonopy",
        supercell=[8, 8, 8],
        only_second=True,
        is_acoustic_sum=True,
        format="shengbte-qe",
    )
    forceconstants.second.folder = str(tmp_path_factory.mktemp("nac_runtime_cache"))
    # The machinery tests attach charges to the file constants and exercise the
    # non-analytic correction pipeline as if they were total force constants.
    forceconstants.second.atoms.info.pop("dipole_subtracted_fc", None)
    forceconstants.second._qe_q2r_header = None
    return forceconstants.second


def test_second_order_nac_short_range_force_constants_use_lazy_numpy_cache(
    nac_second_order, tmp_path, monkeypatch
):
    original_folder = nac_second_order.folder
    nac_second_order.folder = str(tmp_path)
    try:
        expected = np.arange(2 * 4 * 3 * 3, dtype=float).reshape(2, 4, 3, 3)
        calls = {"count": 0}

        def calculate_once():
            calls["count"] += 1
            return expected

        monkeypatch.setattr(
            nac_second_order,
            "calculate_nac_short_range_force_constants",
            calculate_once,
        )
        np.testing.assert_allclose(
            nac_second_order.nac_short_range_force_constants, expected
        )
        assert calls["count"] == 1
        assert (tmp_path / "nac_short_range_force_constants.npy").exists()

        def fail_if_recomputed():
            raise AssertionError(
                "cached NAC short-range force constants were not reused"
            )

        monkeypatch.setattr(
            nac_second_order,
            "calculate_nac_short_range_force_constants",
            fail_if_recomputed,
        )
        np.testing.assert_allclose(
            nac_second_order.nac_short_range_force_constants, expected
        )
    finally:
        nac_second_order.folder = original_folder


def test_second_order_nac_short_range_force_constants_use_matrix_specific_cache(
    nac_second_order, tmp_path, monkeypatch
):
    original_folder = nac_second_order.folder
    nac_second_order.folder = str(tmp_path)
    try:
        matrix = nacl_phonopy_debug_supercell_matrix()
        expected = np.arange(2 * 64 * 3 * 3, dtype=float).reshape(2, 64, 3, 3)
        calls = {"count": 0}

        def calculate_once(nac_bvk_supercell_matrix=None):
            calls["count"] += 1
            np.testing.assert_array_equal(nac_bvk_supercell_matrix, matrix)
            return expected

        monkeypatch.setattr(
            nac_second_order,
            "calculate_nac_short_range_force_constants",
            calculate_once,
        )
        actual = nac_second_order.get_nac_short_range_force_constants(matrix)
        np.testing.assert_allclose(actual, expected)
        assert calls["count"] == 1

        property_name = (
            "nac_short_range_force_constants_" + bvk_supercell_matrix_key(matrix)
        )
        assert (tmp_path / f"{property_name}.npy").exists()

        def fail_if_recomputed(nac_bvk_supercell_matrix=None):
            raise AssertionError("matrix-specific NAC cache was not reused")

        monkeypatch.setattr(
            nac_second_order,
            "calculate_nac_short_range_force_constants",
            fail_if_recomputed,
        )
        actual = nac_second_order.get_nac_short_range_force_constants(matrix)
        np.testing.assert_allclose(actual, expected)
    finally:
        nac_second_order.folder = original_folder


def test_second_order_nac_short_range_force_constants_reuse_in_memory_matrix_cache(
    nac_second_order, tmp_path, monkeypatch
):
    original_folder = nac_second_order.folder
    nac_second_order.folder = str(tmp_path)
    try:
        matrix = nacl_phonopy_debug_supercell_matrix()
        expected = np.arange(2 * 64 * 3 * 3, dtype=float).reshape(2, 64, 3, 3)

        def calculate_once(nac_bvk_supercell_matrix=None):
            np.testing.assert_array_equal(nac_bvk_supercell_matrix, matrix)
            return expected

        monkeypatch.setattr(
            nac_second_order,
            "calculate_nac_short_range_force_constants",
            calculate_once,
        )
        first = nac_second_order.get_nac_short_range_force_constants(matrix)
        np.testing.assert_allclose(first, expected)

        def fail_if_loaded(*args, **kwargs):
            raise AssertionError("matrix-specific NAC array was not reused from memory")

        monkeypatch.setattr(nac_second_order, "_load_property", fail_if_loaded)
        monkeypatch.setattr(
            nac_second_order,
            "calculate_nac_short_range_force_constants",
            fail_if_loaded,
        )
        second = nac_second_order.get_nac_short_range_force_constants(matrix)
        np.testing.assert_allclose(second, expected)
    finally:
        nac_second_order.folder = original_folder


def test_nac_velocity_shape_and_finite(nac_second_order):
    phonon = HarmonicWithQ(
        q_point=np.array([0.1, 0.0, 0.0]),
        second=nac_second_order,
        storage="memory",
        nac_bvk_supercell_matrix=nacl_phonopy_debug_supercell_matrix_att3(),
    )
    velocity = phonon.velocity
    assert velocity.shape == (1, 6, 3)
    assert np.isfinite(velocity).all()


def test_dielectric_without_charges_raises():
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/nacl_phonopy",
        supercell=[8, 8, 8],
        only_second=True,
        is_acoustic_sum=True,
        format="shengbte-qe",
    )
    second = forceconstants.second
    # These fixtures attach reference charges by hand and exercise the
    # machinery on the file constants as if they were totals.
    second.atoms.info.pop("dipole_subtracted_fc", None)
    if "charges" in second.atoms.arrays:
        del second.atoms.arrays["charges"]
    assert "dielectric" in second.atoms.info
    with pytest.raises(ValueError, match="charges"):
        HarmonicWithQ(q_point=np.zeros(3), second=second, storage="memory")
    # Explicit NAC-off is a diagnostic model and therefore does not require a
    # complete polar block that it has been instructed not to use.
    harmonic = HarmonicWithQ(
        q_point=np.zeros(3), second=second, storage="memory", is_nac=False
    )
    assert harmonic.is_nac is False


def test_required_nac_without_polar_metadata_has_actionable_error(nac_second_order):
    """An explicit NAC request should identify missing data and alternatives."""
    second = nac_second_order
    second.atoms.info.pop("dielectric", None)
    if "charges" in second.atoms.arrays:
        del second.atoms.arrays["charges"]

    with pytest.raises(ValueError) as error:
        HarmonicWithQ(
            q_point=np.zeros(3), second=second, storage="memory", is_nac=True
        )

    message = str(error.value)
    assert "atoms.info['dielectric'] is missing" in message
    assert "atoms.arrays['charges'] is missing" in message
    assert "is_nac=None for automatic detection" in message
    assert "is_nac=False to disable NAC" in message


# ---- QE q2r short-range convention ----
# q2r.x writes dipole-subtracted IFCs when the file embeds dielectric and Born
# data. The loader preserves that provenance and the harmonic NAC path adds the
# matching QE rigid-ion correction without applying Gonze subtraction again.
@pytest.fixture(scope="module")
def mgo_second(tmp_path_factory):
    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/mgo",
        supercell=[5, 5, 5],
        only_second=True,
        format="qe-d3q",
    )
    second = forceconstants.second
    second.folder = str(tmp_path_factory.mktemp("mgo_nac_test"))
    return second


def test_embedded_charges_mark_the_convention(mgo_second):
    assert mgo_second.atoms.info.get("dipole_subtracted_fc") is True
    assert "dielectric" in mgo_second.atoms.info
    assert "charges" in mgo_second.atoms.arrays


def test_explicit_nac_control_supports_auto_required_and_off_modes(mgo_second):
    """NAC defaults to auto but can be required or disabled explicitly."""
    q_point = np.array([0.3, 0.0, 0.3], dtype=np.float64)
    automatic = HarmonicWithQ(
        q_point=q_point,
        second=mgo_second,
        storage="memory",
        ifc_interpolation="auto",
    )
    required = HarmonicWithQ(
        q_point=q_point,
        second=mgo_second,
        storage="memory",
        ifc_interpolation="auto",
        is_nac=True,
    )
    disabled = HarmonicWithQ(
        q_point=q_point,
        second=mgo_second,
        storage="memory",
        ifc_interpolation="auto",
        is_nac=False,
    )

    assert automatic.is_nac is True
    assert required.is_nac is True
    assert disabled.is_nac is False
    assert "/nac_off/" in disabled.get_folder_from_label("")
    assert disabled.get_folder_from_label("") != automatic.get_folder_from_label("")
    np.testing.assert_allclose(
        required.frequency,
        automatic.frequency,
        rtol=0.0,
        atol=0.0,
    )
    assert np.max(np.abs(disabled.frequency - automatic.frequency)) > 0.1


@pytest.mark.parametrize(
    "q_point, expected_frequency_cm, degenerate_pairs",
    [
        pytest.param(
            [0.3, 0.0, 0.3],
            [239.7640, 239.7640, 367.6916, 422.9322, 422.9322, 582.6500],
            ((0, 1), (3, 4)),
            id="q-0.3-0-0.3",
        ),
        pytest.param(
            [0.1, 0.0, 0.0],
            [77.0501, 77.0501, 135.2487, 389.4064, 389.4064, 691.1982],
            ((0, 1), (3, 4)),
            id="q-0.1-0-0",
        ),
        pytest.param(
            [0.15, 0.15, 0.15],
            [113.7407, 113.7407, 200.8237, 387.8639, 387.8639, 684.7630],
            ((0, 1), (3, 4)),
            id="q-0.15-0.15-0.15",
        ),
        pytest.param(
            [0.3, 0.1, 0.0],
            [203.9083, 210.4709, 344.9263, 386.7525, 394.9880, 643.5431],
            (),
            id="q-0.3-0.1-0",
        ),
    ],
)
def test_qe_q2r_frequencies_match_matdyn_off_gamma(
    mgo_second, q_point, expected_frequency_cm, degenerate_pairs
):
    """Match all four QE matdyn.x controls retained by the legacy-bug branch."""
    phonon = HarmonicWithQ(
        q_point=np.asarray(q_point, dtype=np.float64),
        second=mgo_second,
        storage="memory",
        ifc_interpolation="auto",
    )
    frequency_cm = phonon.frequency[0] * 33.3564095198152
    assert frequency_cm.shape == (6,)
    assert np.isfinite(frequency_cm).all()
    # QE 7.6 matdyn.x with asr='simple'. These replace branch-generated values
    # from the plain replica transform and cover several directions away from
    # the defining q2r mesh without relaxing the former single-point tolerance.
    np.testing.assert_allclose(
        frequency_cm,
        expected_frequency_cm,
        rtol=0.0,
        atol=1.0e-2,
    )
    for left, right in degenerate_pairs:
        np.testing.assert_allclose(
            frequency_cm[left],
            frequency_cm[right],
            rtol=0.0,
            atol=5.0e-6,
        )


def test_qe_nac_velocity_uses_the_corrected_dispersion_gradient(mgo_second):
    """Keep the QE velocity correction without relaxing the legacy tolerance."""
    phonon = HarmonicWithQ(
        q_point=np.array([0.3, 0.0, 0.3], dtype=np.float64),
        second=mgo_second,
        storage="memory",
        ifc_interpolation="auto",
    )
    velocity_norm = np.linalg.norm(np.asarray(phonon.velocity), axis=-1).reshape(-1)
    # The removed reference [33.906, 33.906, 32.844, 8.682, 8.682, 21.596]
    # came from a velocity operator that was not the gradient of the dynamical
    # matrix. Central finite differences give the corrected values below.
    np.testing.assert_allclose(
        velocity_norm,
        [29.512, 29.512, 46.846, 9.993, 9.993, 32.476],
        rtol=0.0,
        atol=5.0e-3,
    )


def test_bvk_matrix_must_match_the_force_constant_grid(nac_second_order):
    """The short-range pipeline pairs the interleaved force constants (fixed
    enumeration on the FC supercell) with the mapping's translation ordering,
    so any other BvK lattice must be refused loudly instead of failing as an
    opaque broadcast error, or worse, silently mispairing blocks."""
    non_diagonal = np.array([[-2, 2, 2], [2, -2, 2], [2, 2, -2]], dtype=int)
    with pytest.raises(NotImplementedError, match="diag"):
        nac_second_order.calculate_nac_short_range_force_constants(non_diagonal)
    mismatched_diagonal = np.diag([4, 4, 4]).astype(int)
    with pytest.raises(NotImplementedError, match="diag"):
        nac_second_order.calculate_nac_short_range_force_constants(mismatched_diagonal)


def test_scalar_charge_column_is_not_polar_metadata():
    """A per-atom charge column (LAMMPS/extxyz) must not look like Born data."""
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.set_array("charges", np.full(len(atoms), 0.3))
    assert _resolve_nac_activation(atoms, None) is False
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        _resolve_nac_activation(atoms, True)


def test_born_charges_without_dielectric_point_to_is_nac_false():
    """Nonzero Born charges without a dielectric tensor fail actionably."""
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.set_array("charges", np.tile(np.eye(3) * 2.0, (len(atoms), 1, 1)))
    with pytest.raises(ValueError, match="is_nac=False"):
        _resolve_nac_activation(atoms, None)
