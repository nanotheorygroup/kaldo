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
