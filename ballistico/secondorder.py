from ballistico.ifc import Ifc

import numpy as np
import ase.units as units
from ballistico.helpers.logger import get_logger
from opt_einsum import contract

logging = get_logger()
EVTOTENJOVERMOL = units.mol / (10 * units.J)


def acoustic_sum_rule(dynmat):
    n_unit = dynmat[0].shape[0]
    sumrulecorr = 0.
    for i in range(n_unit):
        off_diag_sum = np.sum(dynmat[0, i, :, :, :, :], axis=(-2, -3))
        dynmat[0, i, :, 0, i, :] -= off_diag_sum
        sumrulecorr += np.sum(off_diag_sum)
    logging.info('error sum rule: ' + str(sumrulecorr))
    return dynmat


class SecondOrder(Ifc):
    def __init__(self, atoms, replicated_positions, supercell=None, force_constant=None, is_acoustic_sum=False):
        if is_acoustic_sum:
            force_constant = acoustic_sum_rule(force_constant)
        Ifc.__init__(self, atoms, replicated_positions, supercell, force_constant)
        self._list_of_replicas = None


    @classmethod
    def from_supercell(cls, atoms, grid_type, supercell=None, force_constant=None, is_acoustic_sum=False):
        if force_constant is not None and is_acoustic_sum is not None:
            force_constant = acoustic_sum_rule(force_constant)
        ifc = super(SecondOrder, cls).from_supercell(atoms, supercell, grid_type, force_constant)
        return ifc


    def dynmat(self, mass):
        """Obtain the second order force constant matrix either by loading in
           or performing calculation using the provided calculator.
        """
        # Once confirming that the second order does not exist, try load in
        dynmat = self.force_constant
        dynmat = contract('mialjb,i,j->mialjb', dynmat, 1 / np.sqrt(mass), 1 / np.sqrt(mass))
        return dynmat * EVTOTENJOVERMOL


