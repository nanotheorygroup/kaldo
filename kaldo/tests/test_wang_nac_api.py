import numpy as np
import pytest

from kaldo.forceconstants import ForceConstants
from kaldo.interfaces import shengbte_io
from kaldo.observables.harmonic_with_q import HarmonicWithQ
from kaldo.phonons import Phonons
from kaldo.tests.wang_debug_reference import (
    diagnostic_q_names_wang_att3,
    load_q_tensor,
    require_wang_att3_debug,
)


def attach_reference_nac(second_order, nac_file="examples/nacl_phonopy/espresso.ifc2"):
    _, _, charges = shengbte_io.read_second_order_qe_matrix(nac_file)
    if charges is None:
        raise ValueError(f"No NAC data found in {nac_file}")
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
    forceconstants.second.folder = str(tmp_path_factory.mktemp("wang_runtime_cache"))
    return attach_reference_nac(forceconstants.second)


def test_harmonic_with_q_accepts_wang_nac_options(nac_second_order):
    phonon = HarmonicWithQ(
        q_point=np.array([0.1, 0.0, 0.1]),
        second=nac_second_order,
        storage="memory",
        nac_method="wang",
        nac_debug=True,
        nac_debug_folder="debug",
        q_index=7,
    )
    assert phonon.nac_method == "wang"
    assert phonon.nac_debug is True
    assert phonon.nac_debug_folder == "debug"
    assert phonon.q_index == 7


def test_harmonic_with_q_rejects_wang_without_nac_metadata():
    forceconstants = ForceConstants.from_folder(
        folder="examples/nacl_phonopy_v2",
        supercell=[8, 8, 8],
        only_second=True,
        is_acoustic_sum=True,
        format="shengbte-qe",
    )
    with pytest.raises(ValueError, match=r"nac_method='wang' requires"):
        HarmonicWithQ(
            q_point=np.array([0.1, 0.0, 0.1]),
            second=forceconstants.second,
            storage="memory",
            nac_method="wang",
        )


def test_phonons_accepts_wang_nac_options():
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
        nac_method="wang",
        nac_debug=True,
        nac_debug_folder="debug",
    )
    assert phonons.nac_method == "wang"
    assert phonons.nac_debug is True
    assert phonons.nac_debug_folder == "debug"
    frequencies = phonons.frequency
    assert frequencies.shape == (1, phonons.n_modes)


def test_phonons_rejects_wang_without_nac_metadata():
    forceconstants = ForceConstants.from_folder(
        folder="examples/nacl_phonopy_v2",
        supercell=[8, 8, 8],
        only_second=True,
        is_acoustic_sum=True,
        format="shengbte-qe",
    )
    with pytest.raises(ValueError, match=r"nac_method='wang' requires"):
        phonons = Phonons(
            forceconstants=forceconstants,
            kpts=[1, 1, 1],
            storage="memory",
            nac_method="wang",
        )
        _ = phonons.frequency


def test_wang_frequencies_match_phonopy_debug(nac_second_order):
    root = require_wang_att3_debug()
    q_names = diagnostic_q_names_wang_att3()
    factor_q_name = "q-00010"
    gv_eigvals = np.asarray(load_q_tensor(root, factor_q_name, "py_group_velocity_eigvals"), dtype=float)
    gv_freqs = np.asarray(load_q_tensor(root, factor_q_name, "py_group_velocity_freqs"), dtype=float)
    factor_mask = gv_eigvals > 1e-12
    frequency_factor = float(np.median(gv_freqs[factor_mask] / np.sqrt(gv_eigvals[factor_mask])))
    n_modes = nac_second_order.atoms.positions.shape[0] * 3

    for q_name in q_names:
        q_point = np.asarray(load_q_tensor(root, q_name, "py_qpoints"), dtype=float)
        q_direction = np.asarray(load_q_tensor(root, q_name, "py_q_direction"), dtype=float)
        traced_dynmat = np.fromfile(
            root / q_name / "wang_dynmat_after_hermitian.bin",
            dtype=np.complex128,
        ).reshape((n_modes, n_modes))
        traced_eigvals = np.linalg.eigvalsh(traced_dynmat).real
        expected = np.sign(traced_eigvals) * np.sqrt(np.abs(traced_eigvals)) * frequency_factor

        phonon = HarmonicWithQ(
            q_point=q_point,
            second=nac_second_order,
            storage="memory",
            nac_method="wang",
            nac_q_direction=q_direction,
            q_index=int(q_name.split("-")[1]),
        )

        actual = np.asarray(phonon.frequency, dtype=float).reshape(-1)
        np.testing.assert_allclose(actual, expected, atol=0.06, rtol=0.03)


def test_wang_full_dynamical_matrix_is_hermitian(nac_second_order):
    root = require_wang_att3_debug()
    factor_q_name = "q-00010"
    gv_eigvals = np.asarray(load_q_tensor(root, factor_q_name, "py_group_velocity_eigvals"), dtype=float)
    gv_freqs = np.asarray(load_q_tensor(root, factor_q_name, "py_group_velocity_freqs"), dtype=float)
    factor_mask = gv_eigvals > 1e-12
    frequency_factor = float(np.median(gv_freqs[factor_mask] / np.sqrt(gv_eigvals[factor_mask])))
    matrix_scale = (2 * np.pi * frequency_factor) ** 2
    n_modes = nac_second_order.atoms.positions.shape[0] * 3

    for q_name in diagnostic_q_names_wang_att3():
        q_point = np.asarray(load_q_tensor(root, q_name, "py_qpoints"), dtype=float)
        q_direction = np.asarray(load_q_tensor(root, q_name, "py_q_direction"), dtype=float)
        phonon = HarmonicWithQ(
            q_point=q_point,
            second=nac_second_order,
            storage="memory",
            nac_method="wang",
            nac_q_direction=q_direction,
            q_index=int(q_name.split("-")[1]),
        )

        dynamical_matrix = phonon._calculate_wang_dynamical_matrix()
        traced_dynmat = np.fromfile(
            root / q_name / "wang_dynmat_after_hermitian.bin",
            dtype=np.complex128,
        ).reshape((n_modes, n_modes)) * matrix_scale
        np.testing.assert_allclose(
            dynamical_matrix,
            dynamical_matrix.conj().T,
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            dynamical_matrix,
            traced_dynmat,
            atol=6.0,
            rtol=0.01,
        )
