"""
Unit test for verifying cached values work correctly when switching
between quantum and classical simulations in the same folder.
"""

# Import package, test suite, and other packages as needed
from kaldo.forceconstants import ForceConstants
import numpy as np
from kaldo.phonons import Phonons
import pytest
import tempfile
import shutil
import os


def test_quantum_then_classic_cache():
    """
    Test that running a quantum simulation followed by a classical simulation
    in the same folder handles cached values correctly.
    """
    # Create a temporary directory for this test
    temp_dir = tempfile.mkdtemp(prefix="kaldo_test_")

    try:
        # Create force constants object (shared by both simulations)
        forceconstants = ForceConstants.from_folder(
            folder="kaldo/tests/si-crystal",
            supercell=[3, 3, 3],
            format="eskm"
        )

        # First: Run quantum simulation
        phonons_quantum = Phonons(
            forceconstants=forceconstants,
            kpts=[5, 5, 5],
            is_classic=False,
            temperature=300,
            storage="formatted",
            folder=temp_dir,
        )

        # Access properties to trigger calculation and caching
        quantum_phase_space = phonons_quantum.phase_space
        quantum_frequency = phonons_quantum.frequency

        # Verify quantum results
        np.testing.assert_approx_equal(quantum_phase_space.sum(), 113, significant=3)

        # Second: Run classical simulation in the SAME folder
        # This should handle cached quantum values correctly
        phonons_classic = Phonons(
            forceconstants=forceconstants,
            kpts=[5, 5, 5],
            is_classic=True,
            temperature=300,
            storage="formatted",
            folder=temp_dir,
        )

        # Access properties to trigger calculation
        classic_phase_space = phonons_classic.phase_space
        classic_frequency = phonons_classic.frequency

        # Verify classical results are different from quantum
        # Phase space should be different due to different statistics
        assert not np.allclose(quantum_phase_space, classic_phase_space, rtol=0.01)

        # Frequencies should be the same (harmonic property, independent of statistics)
        np.testing.assert_allclose(quantum_frequency, classic_frequency, rtol=1e-10)

        # Verify both phonon objects coexist without interference
        # Re-access quantum properties to ensure they're still correct
        np.testing.assert_approx_equal(
            phonons_quantum.phase_space.sum(),
            113,
            significant=3
        )

    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_free_energy_statistics(tmp_path):
    """Classical free energy must differ from quantum and cache separately.

    Regression: free_energy ignored is_classic (always the quantum formula)
    and its cache label lacked <statistics>, so quantum and classical runs
    shared a storage path. The classical formula is F = k_B T ln(x) with
    x = hbar*omega/(k_B T), the x -> 0 limit of the quantum
    F = k_B T ln(1 - exp(-x)) + hbar*omega/2 (the zero-point term cancels
    against the expansion of the logarithm); it uses the physical hbar and
    contains no zero-point energy.
    """
    import ase.units as units

    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/si-crystal", supercell=[3, 3, 3], format="eskm"
    )
    kwargs = dict(forceconstants=forceconstants, kpts=[3, 3, 3],
                  storage="formatted", folder=str(tmp_path))

    # Quantum first, same folder: without <statistics> in the label the
    # classical run below would silently read this cache.
    ph_q = Phonons(is_classic=False, temperature=300, **kwargs)
    f_q = ph_q.free_energy

    ph_c = Phonons(is_classic=True, temperature=300, **kwargs)
    f_c = ph_c.free_energy

    frequency = ph_c.frequency
    physical = ph_c.physical_mode.reshape(frequency.shape) & (frequency > 0)
    x = units._hbar * frequency * 2.0 * np.pi * 1.0e12 / (units._k * 300.0)
    kT_eV = units._k * 300.0 / units._e

    expected = np.zeros_like(x)
    expected[physical] = kT_eV * np.log(x[physical])
    expected /= ph_c.n_k_points
    np.testing.assert_allclose(f_c, expected, rtol=1e-10, atol=1e-16)

    # And the quantum cache must have stayed quantum.
    assert not np.allclose(f_q, f_c, rtol=1e-3)
    np.testing.assert_allclose(ph_q.free_energy, f_q, rtol=0, atol=0)


def test_free_energy_quantum_minus_classical_is_wigner_correction(tmp_path):
    """Independent physics cross-check of both statistics branches.

    In the high-temperature regime the quantum and classical harmonic free
    energies differ by the leading Wigner correction,
    F_q - F_cl = (hbar*omega)^2 / (24 k_B T) per mode, with relative
    corrections of O(x^2). At 2000 K silicon's stiffest mode has x ~ 0.4,
    so the identity holds to a few parts in 1e3. A wrong hbar convention,
    a surviving zero-point term, or a sign slip in either branch breaks
    this by orders of magnitude.
    """
    import ase.units as units

    forceconstants = ForceConstants.from_folder(
        folder="kaldo/tests/si-crystal", supercell=[3, 3, 3], format="eskm"
    )
    temperature = 2000.0
    kwargs = dict(forceconstants=forceconstants, kpts=[3, 3, 3],
                  temperature=temperature, storage="memory")

    f_q = Phonons(is_classic=False, **kwargs).free_energy
    ph_c = Phonons(is_classic=True, **kwargs)
    f_c = ph_c.free_energy

    frequency = ph_c.frequency
    physical = ph_c.physical_mode.reshape(frequency.shape) & (frequency > 0)
    x = units._hbar * frequency * 2.0 * np.pi * 1.0e12 / (units._k * temperature)
    kT_eV = units._k * temperature / units._e

    wigner = (x[physical] ** 2 / 24.0) * kT_eV / ph_c.n_k_points
    np.testing.assert_allclose((f_q - f_c)[physical], wigner, rtol=5e-3)
