from opt_einsum import contract
from scipy.linalg.lapack import dsyev
import numpy as np
from ballistico.helpers.tools import apply_boundary_with_cell



class HarmonicSingleQ:
    def __init__(self, **kwargs):
        self.qvec = kwargs.pop('qvec', (0, 0, 0))
        self.qvec = apply_boundary_with_cell(self.qvec)
        self.frequency_threshold = kwargs.pop('frequency_threshold')
        finite_difference = kwargs.pop('finite_difference')
        self.dynmat = finite_difference.dynmat
        self.positions = finite_difference.atoms.positions
        self.replicated_cell = finite_difference.replicated_atoms.cell
        self.replicated_cell_inv = finite_difference.replicated_cell_inv
        self.cell_inv = finite_difference.cell_inv
        self.list_of_replicas = finite_difference.list_of_replicas
        self.n_atoms = finite_difference.n_atoms
        self.n_modes = self.n_atoms * 3

        self._is_at_gamma = (self.qvec == (0, 0, 0)).all()

        if self._is_at_gamma:
            self.is_nanowire = kwargs.pop('is_nw', False)
            if self.is_nanowire:
                self._first_physical_index = 4
            else:
                self._first_physical_index = 3


    def calculate_eigensystem(self, only_eigenvals=False):
        dynmat = self.dynmat
        if self._is_at_gamma:
            dyn_s = contract('ialjb->iajb', dynmat)
        else:
            dyn_s = contract('ialjb,l->iajb', dynmat, self.chi())
        dyn_s = dyn_s.reshape((self.n_modes, self.n_modes), order='C')
        if only_eigenvals:
            evals = np.linalg.eigvalsh(dyn_s)
            return evals
        else:
            if self._is_at_gamma:
                evals, evects = dsyev(dyn_s)[:2]
            else:
                evals, evects = np.linalg.eigh(dyn_s)
            return np.vstack((evals, evects))


    def calculate_dynmat_derivatives(self):
        dynmat = self.dynmat
        positions = self.positions
        if self._is_at_gamma:
            dxij = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dxij = apply_boundary_with_cell(dxij, self.replicated_cell, self.replicated_cell_inv)
            dynmat_derivatives = contract('ija,ibjc->ibjca', dxij, dynmat[:, :, 0, :, :])
        else:
            list_of_replicas = self.list_of_replicas
            dxij = positions[:, np.newaxis, np.newaxis, :] - (
                        positions[np.newaxis, np.newaxis, :, :] + list_of_replicas[np.newaxis, :, np.newaxis, :])
            dynmat_derivatives = contract('ilja,ibljc,l->ibjca', dxij, dynmat, self.chi())
        dynmat_derivatives = dynmat_derivatives.reshape((self.n_modes, self.n_modes, 3), order='C')
        return dynmat_derivatives


    def calculate_frequencies(self):
        eigenvals = self.calculate_eigensystem(only_eigenvals=True)
        frequencies = np.abs(eigenvals) ** .5 * np.sign(eigenvals) / (np.pi * 2.)
        return frequencies


    def calculate_velocities_AF(self):
        dynmat_derivatives = self.calculate_dynmat_derivatives()
        frequencies = self.calculate_frequencies()
        eigenvects = self.calculate_eigensystem()[1:, :]
        physical_modes = frequencies > self.frequency_threshold
        if self._is_at_gamma:
            physical_modes[:self._first_physical_index] = False

        velocities_AF = contract('im,ija,jn->mna', eigenvects[:, :].conj(), dynmat_derivatives, eigenvects[:, :])
        velocities_AF = contract('mna,mn->mna', velocities_AF,
                                 1 / (2 * np.pi * np.sqrt(frequencies[:, np.newaxis]) * np.sqrt(
                                     frequencies[np.newaxis, :])))
        velocities_AF[np.invert(physical_modes), :, :] = 0
        velocities_AF[:, np.invert(physical_modes), :] = 0
        velocities_AF = velocities_AF / 2
        return velocities_AF


    def calculate_velocities(self):
        velocities_AF = self.calculate_velocities_AF()
        velocities = 1j * np.diagonal(velocities_AF).T
        return velocities


    def chi(self):
        qvec = self.qvec
        dxij = self.list_of_replicas
        cell_inv = self.cell_inv
        chi_k = np.exp(1j * 2 * np.pi * dxij.dot(cell_inv.dot(qvec)))
        return chi_k