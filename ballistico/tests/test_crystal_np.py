"""
Unit and regression test for the ballistico package.
"""

# Import package, test suite, and other packages as needed
from finitedifference.finitedifference import FiniteDifference
import numpy as np
from ballistico.phonons import Phonons
import ballistico.conductivity as bac
import shutil
import pytest

TMP_FOLDER = 'ballistico/tests/tmp-folder'

@pytest.yield_fixture(scope="session")
def phonons():
    print ("Preparing phonons object.")
    finite_difference = FiniteDifference.import_from_dlpoly_folder(folder='ballistico/tests/si-crystal',
                                                                   supercell=[3, 3, 3])

    # Create a phonon object
    si_bulk = Phonons(finite_difference=finite_difference,
                      kpts=[5, 5, 5],
                      is_classic=False,
                      temperature=300,
                      folder=TMP_FOLDER,
                      is_tf_backend=False)
    yield si_bulk
    print ("Cleaning up.")
    shutil.rmtree(TMP_FOLDER, ignore_errors=True)


def test_sc_conductivity(phonons):
    cond = np.abs(np.mean(bac.conductivity(phonons, method='sc', max_n_iterations=71)[0].sum(axis=0).diagonal()))
    np.testing.assert_approx_equal(cond, 255, significant=3)


def test_qhgk_conductivity(phonons):
    cond = bac.conductivity(phonons, method='qhgk').sum(axis=0)
    cond = np.abs(np.mean(cond.diagonal()))
    np.testing.assert_approx_equal(cond, 230, significant=3)


def test_rta_conductivity(phonons):
    cond = np.abs(np.mean(bac.conductivity(phonons, method='rta').sum(axis=0).diagonal()))
    np.testing.assert_approx_equal(cond, 226, significant=3)


def test_inverse_conductivity(phonons):
    cond = np.abs(np.mean(bac.conductivity(phonons, method='inverse').sum(axis=0).diagonal()))
    np.testing.assert_approx_equal(cond, 256, significant=3)