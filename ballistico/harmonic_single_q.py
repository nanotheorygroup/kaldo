from opt_einsum import contract
from scipy.linalg.lapack import dsyev
import numpy as np
from ballistico.helpers.tools import apply_boundary_with_cell



class HarmonicSingleQ:
    def __init__(self, **kwargs):
        self.qvec = kwargs.pop('qvec', (0, 0, 0))
        self.qvec = apply_boundary_with_cell(self.qvec)
        self.min_frequency = kwargs.pop('min_frequency')
        self.max_frequency = kwargs.pop('max_frequency')
        self.is_amorphous = kwargs.pop('is_amorphous')
        finite_difference = kwargs.pop('finite_difference')
        self.dynmat = finite_difference.dynmat
        self.positions = finite_difference.atoms.positions
        self.replicated_atoms = finite_difference.replicated_atoms
        self.replicated_cell = finite_difference.replicated_atoms.cell
        self.replicated_cell_inv = finite_difference.replicated_cell_inv
        self.cell_inv = finite_difference.cell_inv
        self.list_of_replicas = finite_difference.list_of_replicas
        self.n_atoms = finite_difference.n_atoms
        self.n_modes = self.n_atoms * 3
        self.n_replicas = finite_difference.n_replicas

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
        if self.is_amorphous:
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


    def calculate_sij(self, is_antisymmetrizing=False):
        # TODO: THis should be lazy loaded instead
        dynmat_derivatives = self.calculate_dynmat_derivatives()
        frequencies = self.calculate_frequencies()
        eigenvects = self.calculate_eigensystem()[1:, :]
        physical_modes = np.ones_like(frequencies, dtype=bool)
        if self.min_frequency is not None:
            physical_modes = physical_modes & (frequencies > self.min_frequency)
        if self.max_frequency is not None:
            physical_modes = physical_modes & (frequencies < self.max_frequency)
        if self._is_at_gamma:
            physical_modes[:self._first_physical_index] = False

        if is_antisymmetrizing:
            error = np.linalg.norm(dynmat_derivatives + dynmat_derivatives.swapaxes(0, 1)) / 2
            dynmat_derivatives = (dynmat_derivatives - dynmat_derivatives.swapaxes(0, 1)) / 2
            print('Symmetrization errror: ' + str(error))
        if self.is_amorphous:
            sij = contract('im,ija,jn->mna', eigenvects[:, :], dynmat_derivatives, eigenvects[:, :])
        else:
            sij = contract('im,ija,jn->mna', eigenvects[:, :].conj(), dynmat_derivatives, eigenvects[:, :])
        return sij


    def calculate_velocities_af(self, is_antisymmetrizing=False):
        frequencies = self.calculate_frequencies()
        # TODO: Here we should reuse the phonons._sij
        sij = self.calculate_sij(is_antisymmetrizing)
        velocities_AF = contract('mna,mn->mna', sij,
                                 1 / (2 * np.pi * np.sqrt(frequencies[:, np.newaxis]) * np.sqrt(
                                     frequencies[np.newaxis, :]))) / 2
        return velocities_AF


    def calculate_velocities(self, is_antisymmetrizing=False):
        velocities_AF = self.calculate_velocities_af(is_antisymmetrizing=is_antisymmetrizing)
        velocities = 1j * np.diagonal(velocities_AF).T
        return velocities


    def chi(self):
        qvec = self.qvec
        dxij = self.list_of_replicas
        cell_inv = self.cell_inv
        chi_k = np.exp(1j * 2 * np.pi * dxij.dot(cell_inv.dot(qvec)))
        return chi_k