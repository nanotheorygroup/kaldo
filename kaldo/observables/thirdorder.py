from kaldo.observables.forceconstant import ForceConstant
import os
import ase.io
import numpy as np
from scipy.sparse import save_npz
import ase.units as units
from kaldo.interfaces import ForceConstantData, load_forceconstants, resolve_structure
from kaldo.controllers.displacement import calculate_third
from kaldo.helpers.logger import get_logger
from kaldo.grid import Grid

logging = get_logger()

class ThirdOrder(ForceConstant):

    @classmethod
    def from_data(cls, data: ForceConstantData, *, folder: str | None = None) -> "ThirdOrder":
        replicas = data.replicated_atoms.positions if data.replicated_atoms is not None else None
        obj = cls(
            atoms=data.unit_atoms,
            replicated_positions=replicas,
            supercell=data.supercell,
            value=data.value,
            folder=folder,
        )
        obj._grid_order = data.grid_order
        return obj

    @classmethod
    def load(cls,
             folder: str,
             supercell: tuple[int, int, int] = (1, 1, 1),
             format: str = 'sparse',
             third_energy_threshold: float = 0.,
             chunk_size: int = 100000):
        """Load third-order force constants for the requested format."""
        resolved = resolve_structure(folder, format_hint=format, supercell_hint=supercell)
        data = load_forceconstants(3, format, resolved, folder=folder,
                                   third_energy_threshold=third_energy_threshold,
                                   chunk_size=chunk_size)
        return cls.from_data(data, folder=folder)

    def save(self, filename='THIRD', format='sparse', min_force=1e-6):
        folder = self.folder
        filename = folder + '/' + filename
        n_atoms = self.atoms.positions.shape[0]
        match format:
            case 'eskm':
                logging.info('Exporting third in eskm format')
                n_replicas = self.n_replicas
                n_replicated_atoms = n_atoms * n_replicas
                tenjovermoltoev = 10 * units.J / units.mol
                third = self.value.reshape((n_atoms, 3, n_replicated_atoms, 3, n_replicated_atoms, 3)) / tenjovermoltoev
                with open(filename, 'w') as out_file:
                    for i in range(n_atoms):
                        for alpha in range(3):
                            for j in range(n_replicated_atoms):
                                for beta in range(3):
                                    value = third[i, alpha, j, beta].todense()
                                    mask = np.argwhere(np.linalg.norm(value, axis=1) > min_force)
                                    if mask.any():
                                        for k in mask:
                                            k = k[0]
                                            out_file.write("{:5d} ".format(i + 1))
                                            out_file.write("{:5d} ".format(alpha + 1))
                                            out_file.write("{:5d} ".format(j + 1))
                                            out_file.write("{:5d} ".format(beta + 1))
                                            out_file.write("{:5d} ".format(k + 1))
                                            for gamma in range(3):
                                                out_file.write(' {:16.6f}'.format(third[i, alpha, j, beta, k, gamma]))
                                            out_file.write('\n')
                logging.info('Done exporting third.')
            case 'sparse' | 'numpy':
                config_file = folder + REPLICATED_ATOMS_THIRD_FILE
                ase.io.write(config_file, self.replicated_atoms, format='extxyz')

                save_npz(folder + '/' + THIRD_ORDER_FILE_SPARSE, self.value.reshape((n_atoms * 3 * self.n_replicas *
                                                                            n_atoms * 3, self.n_replicas *
                                                                            n_atoms * 3)).to_scipy_sparse())
            case _:
                super(ThirdOrder, self).save(filename, format)



    def calculate(self, calculator, delta_shift=1e-4, distance_threshold=None, is_storing=True, is_verbose=False):
        atoms = self.atoms
        replicated_atoms = self.replicated_atoms
        replicated_atoms.calc = calculator
        if is_storing:
            try:
                self.value = ThirdOrder.load(folder=self.folder, supercell=self.supercell).value

            except FileNotFoundError:
                logging.info('Third order not found. Calculating.')
                self.value = calculate_third(atoms,
                                             replicated_atoms,
                                             delta_shift,
                                             distance_threshold=distance_threshold,
                                             is_verbose=is_verbose)
                self.save('third')
                ase.io.write(self.folder + '/' + REPLICATED_ATOMS_THIRD_FILE, self.replicated_atoms, 'extxyz')
            else:
                logging.info('Reading stored third')
        else:
            self.value = calculate_third(atoms,
                                         replicated_atoms,
                                         delta_shift,
                                         distance_threshold=distance_threshold,
                                         is_verbose=is_verbose)
            if is_storing:
                self.save('third')
                ase.io.write(self.folder + '/' + REPLICATED_ATOMS_THIRD_FILE, self.replicated_atoms, 'extxyz')




    def __str__(self):
        return 'third'
