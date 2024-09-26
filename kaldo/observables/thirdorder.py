from kaldo.observables.forceconstant import ForceConstant
from ase import Atoms
import os
import ase.io
import numpy as np
from scipy.sparse import load_npz, save_npz
from sparse import COO
from kaldo.interfaces.eskm_io import import_from_files
import kaldo.interfaces.shengbte_io as shengbte_io
import ase.units as units
from kaldo.controllers.displacement import calculate_third
from kaldo.helpers.logger import get_logger
from kaldo.grid import Grid

logging = get_logger()

REPLICATED_ATOMS_THIRD_FILE = 'replicated_atoms_third.xyz'
REPLICATED_ATOMS_FILE = 'replicated_atoms.xyz'
THIRD_ORDER_FILE_SPARSE = 'third.npz'
THIRD_ORDER_FILE = 'third.npy'

class ThirdOrder(ForceConstant):

    @classmethod
    def load(cls, folder, supercell=(1, 1, 1), format='sparse', third_energy_threshold=0.):
        """
        Create a finite difference object from a folder
        :param folder:
        :param supercell:
        :param format:
        :param third_energy_threshold:
        :param is_acoustic_sum:
        :return:
        """

        if format == 'sparse':

            if folder[-1] != '/':
                folder = folder + '/'
            try:
                config_file = folder + REPLICATED_ATOMS_THIRD_FILE
                replicated_atoms = ase.io.read(config_file, format='extxyz')
            except FileNotFoundError:
                config_file = folder + REPLICATED_ATOMS_FILE
                replicated_atoms = ase.io.read(config_file, format='extxyz')

            n_replicas = np.prod(supercell)
            n_total_atoms = replicated_atoms.positions.shape[0]
            n_unit_atoms = int(n_total_atoms / n_replicas)
            unit_symbols = []
            unit_positions = []
            for i in range(n_unit_atoms):
                unit_symbols.append(replicated_atoms.get_chemical_symbols()[i])
                unit_positions.append(replicated_atoms.positions[i])
            unit_cell = replicated_atoms.cell / supercell

            atoms = Atoms(unit_symbols,
                          positions=unit_positions,
                          cell=unit_cell,
                          pbc=[1, 1, 1])

            _third_order = COO.from_scipy_sparse(load_npz(folder + THIRD_ORDER_FILE_SPARSE)) \
                .reshape((n_unit_atoms * 3, n_replicas * n_unit_atoms * 3, n_replicas * n_unit_atoms * 3))
            third_order = ThirdOrder(atoms=atoms,
                                     replicated_positions=replicated_atoms.positions,
                                     supercell=supercell,
                                     value=_third_order,
                                     folder=folder)

        elif format == 'eskm' or format == 'lammps':
            if format == 'eskm':
                config_file = str(folder) + "/CONFIG"
                replicated_atoms = ase.io.read(config_file, format='dlp4')
            elif format == 'lammps':
                config_file = str(folder) + "/replicated_atoms.xyz"
                replicated_atoms = ase.io.read(config_file, format='extxyz')

            third_file = str(folder) + "/THIRD"
            n_replicas = np.prod(supercell)
            n_total_atoms = replicated_atoms.positions.shape[0]
            n_unit_atoms = int(n_total_atoms / n_replicas)
            unit_symbols = []
            unit_positions = []
            for i in range(n_unit_atoms):
                unit_symbols.append(replicated_atoms.get_chemical_symbols()[i])
                unit_positions.append(replicated_atoms.positions[i])
            unit_cell = replicated_atoms.cell / supercell

            atoms = Atoms(unit_symbols,
                          positions=unit_positions,
                          cell=unit_cell,
                          pbc=[1, 1, 1])


            out = import_from_files(replicated_atoms=replicated_atoms,
                                                third_file=third_file,
                                                supercell=supercell,
                                                third_energy_threshold=third_energy_threshold)
            third_order = ThirdOrder(atoms=atoms,
                                     replicated_positions=replicated_atoms.positions,
                                     supercell=supercell,
                                     value=out[1],
                                     folder=folder)


        elif format == 'shengbte' or format == 'shengbte-qe' or format=='shengbte-d3q':
            grid_type='F'
            config_file = folder + '/' + 'CONTROL'
            try:
                atoms, _supercell, charges = shengbte_io.import_control_file(config_file)
            except FileNotFoundError as err:
                config_file = folder + '/' + 'POSCAR'
                logging.info('\nTrying to open POSCAR')
                atoms = ase.io.read(config_file)

            third_file = folder + '/' + 'FORCE_CONSTANTS_3RD'
            if format == 'shengbte-d3q':
                third_order = shengbte_io.read_third_d3q(third_file, atoms, supercell, order='C')
            else:
                third_order = shengbte_io.read_third_order_matrix(third_file, atoms, supercell, order='C')
            third_order = ThirdOrder.from_supercell(atoms=atoms,
                                                    grid_type=grid_type,
                                                    supercell=supercell,
                                                    value=third_order,
                                                    folder=folder)

        elif format == 'hiphive':
            filename = 'atom_prim.xyz'
            # TODO: add replicated filename in example
            replicated_filename = 'replicated_atoms.xyz'
            try:
                import kaldo.interfaces.hiphive_io as hiphive_io
            except ImportError:
                logging.error('In order to use hiphive along with kaldo, hiphive is required. \
                      Please consider installing hihphive. More info can be found at: \
                      https://hiphive.materialsmodeling.org/')

            atom_prime_file = str(folder) + '/' + filename
            replicated_atom_prime_file = str(folder) + '/' + replicated_filename
            # TODO: Make this independent of replicated file
            atoms = ase.io.read(atom_prime_file)
            try:
                replicated_atoms = ase.io.read(replicated_atom_prime_file)
            except FileNotFoundError:
                logging.warning('Replicated atoms file not found. Please check if the file exists. Using the unit cell atoms instead.')
                replicated_atoms = atoms * (supercell[0], 1, 1) * (1, supercell[1], 1) * (1, 1, supercell[2])

            if 'model3.fcs' in os.listdir(str(folder)):
                # Derive constants used for third-order reshape
                supercell = np.array(supercell)
                n_prim = atoms.copy().get_masses().shape[0]
                n_sc = np.prod(supercell)
                pbc_conditions = replicated_atoms.get_pbc()
                dim = len(pbc_conditions[pbc_conditions == True])
                _third_order = hiphive_io.import_third_from_hiphive(atoms, supercell, folder)
                _third_order = _third_order[0].reshape(n_prim * dim, n_sc * n_prim * dim,
                                                                       n_sc * n_prim * dim)
                third_order = cls(atoms=atoms,
                                  replicated_positions=replicated_atoms.positions,
                                  supercell=supercell,
                                  value=_third_order,
                                  folder=folder)
        elif format == 'sscha':
            filename = 'atom_prim.xyz'
            replicated_filename = 'replicated_atoms.xyz'
            try:
                from hiphive import ForceConstants as hFC
            except ImportError:
                logging.error('In order to use hiphive along with kaldo, hiphive is required. \
                                  Please consider installing hihphive. More info can be found at: \
                                  https://hiphive.materialsmodeling.org/')
            atom_prime_file = str(folder) + '/' + filename
            replicated_atom_prime_file = str(folder) + '/' + replicated_filename
            atoms = ase.io.read(atom_prime_file)
            replicated_atoms = ase.io.read(replicated_atom_prime_file)
            if 'THIRD' in os.listdir(str(folder)):
                supercell = np.array(supercell)
                n_prim = atoms.copy().get_masses().shape[0]
                n_sc = np.prod(supercell)
                pbc_conditions = replicated_atoms.get_pbc()
                dim = len(pbc_conditions[pbc_conditions == True])
                third_hiphive_file = str(folder) + '/THIRD'
                supercell = np.array(supercell)
                replicated_atoms = read(str(folder) + '/replicated_atoms.xyz')
                # Derive constants used for third-order reshape
                fcs3 = hFC.read_shengBTE(supercell=supercell, fname=third_hiphive_file, prim=prim)
                _third_order = fcs3.get_fc_array(3).transpose(0, 3, 1, 4, 2, 5).reshape(n_sc, n_prim, dim,
                                                                                        n_sc, n_prim, dim, n_sc, n_prim,
                                                                                        dim)
                _third_order = _third_order[0].reshape(n_prim * dim, n_sc * n_prim * dim,
                                                       n_sc * n_prim * dim)
                third_order = cls(atoms=atoms,
                                  replicated_positions=replicated_atoms.positions,
                                  supercell=supercell,
                                  value=_third_order,
                                  folder=folder)

        # Newly added by me!!!!
        elif format == 'sscha':
            filename = 'atom_prim.xyz'
            replicated_filename = 'replicated_atoms.xyz'
            try:
                from hiphive import ForceConstants as hFC
            except ImportError:
                logging.error('In order to use hiphive along with kaldo, hiphive is required. \
                                  Please consider installing hihphive. More info can be found at: \
                                  https://hiphive.materialsmodeling.org/')
            atom_prime_file = str(folder) + '/' + filename
            replicated_atom_prime_file = str(folder) + '/' + replicated_filename
            atoms = ase.io.read(atom_prime_file)
            replicated_atoms = ase.io.read(replicated_atom_prime_file)
            with open('FC3', 'r') as f:
                lines = f.readlines()
                f.close()
            c = 13.605693012183622 * (1.889725989 ** (3))
            with open(str(folder + '/THIRD.sheng'), 'w') as f:
                i = 1
                for line in lines:
                    val = line.split()  # New
                    if i % 31 == 2:
                        f.write('\n')
                        f.write(line)
                    elif (i - 1) % 31 > 4 and (i - 1) % 31 < 30:  # New
                        f.write(
                            ' {0}  {1}  {2}'.format(val[0], val[1], val[2]) + ' ' * 5 + str(float(val[3]) * c) + '\n')
                    elif (i - 1) % 31 == 0 and i != 1:  # New
                        f.write(
                            ' {0}  {1}  {2}'.format(val[0], val[1], val[2]) + ' ' * 5 + str(float(val[3]) * c) + '\n')
                    else:
                        f.write(line)
                    i = i + 1
                f.close()
            if 'THIRD.sheng' in os.listdir(str(folder)):
                n_prim = atoms.copy().get_masses().shape[0]
                n_sc = np.prod(supercell)
                pbc_conditions = replicated_atoms.get_pbc()
                dim = len(pbc_conditions[pbc_conditions == True])
                third_hiphive_file = str(folder) + '/THIRD'
                supercell = np.array(supercell)
                # Derive constants used for third-order reshape
                fcs3 = hFC.read_shengBTE(supercell=replicated_atoms,fname=third_hiphive_file,prim=atoms)
                _third_order = fcs3.get_fc_array(3).transpose(0, 3, 1, 4, 2, 5).reshape(n_sc, n_prim, dim,
                                                                                       n_sc, n_prim, dim, n_sc, n_prim,
                                                                                       dim)
                _third_order = _third_order[0].reshape(n_prim * dim, n_sc * n_prim * dim,
                                                       n_sc * n_prim * dim)
                third_order = cls(atoms=atoms,
                                  replicated_positions=replicated_atoms.positions,
                                  supercell=supercell,
                                  value=_third_order,
                                  folder=folder)

        elif format == 'tdep':
            uc = ase.io.read(folder+'/infile.ucposcar', format='vasp')
            sc = ase.io.read(folder+'/infile.ssposcar', format='vasp')
            fc_filename = folder+'/infile.forceconstant_thirdorder'
            n_unit_atoms = uc.positions.shape[0]
            n_replicas = np.prod(supercell)
            order = 'C'

            second_cell_list = []
            third_cell_list = []

            current_grid = Grid(supercell, order=order).grid(is_wrapping=True)
            list_of_index = current_grid
            list_of_replicas = list_of_index.dot(uc.cell)

            with open(fc_filename, 'r') as file:
                line = file.readline()
                num1 = int(line.split()[0])
                line = file.readline()
                lines = file.readlines()
                file.close()

            num_triplets = []
            new_ind = 0
            count = 0
            if count == 0:
                n_t = int(lines[0].split()[0])
                num_triplets.append(n_t)
                new_ind += int(n_t*15+1)
                count += 1
            while count != 0 and new_ind < len(lines):
                n_t = int(lines[new_ind].split()[0])
                num_triplets.append(n_t)
                new_ind += int(n_t * 15 + 1)


            coords = []
            frcs = np.zeros((n_unit_atoms, 3, n_replicas, n_unit_atoms, 3, n_replicas, n_unit_atoms, 3))

            for count1 in range(num1):
                for j in range(len(num_triplets)):
                    n_trip = num_triplets[j]
                    lower = 0
                    for i in range(j):
                        lower += int(num_triplets[i] * 15 + 1)
                    upper = lower + int(n_trip*15+1)
                    subset = lines[lower:upper]
                    subset = subset[1:]
                    num2 = int(len(subset) / 15)
                    for count2 in range(num2):
                        lower2 = int(count2 * 15)
                        upper2 = int((count2 + 1) * 15)
                        ssubset = subset[lower2:upper2]
                        atom_i = int(ssubset[0].split()[0]) - 1
                        atom_j = int(ssubset[1].split()[0]) - 1
                        atom_k = int(ssubset[2].split()[0]) - 1
                        R1 = np.array(ssubset[3].split(), dtype=float)
                        R2 = np.array(ssubset[4].split(), dtype=float)
                        R3 = np.array(ssubset[5].split(), dtype=float)
                        phi1 = ssubset[6].split()
                        phi2 = ssubset[7].split()
                        phi3 = ssubset[8].split()
                        phi4 = ssubset[9].split()
                        phi5 = ssubset[10].split()
                        phi6 = ssubset[11].split()
                        phi7 = ssubset[12].split()
                        phi8 = ssubset[13].split()
                        phi9 = ssubset[14].split()
                        phi = np.array(
                            [[[phi1[0], phi1[1], phi1[2]], [phi2[0], phi2[1], phi2[2]], [phi3[0], phi3[1], phi3[2]]],
                            [[phi4[0], phi4[1], phi4[2]], [phi5[0], phi5[1], phi5[2]], [phi6[0], phi6[1], phi6[2]]],
                            [[phi7[0], phi7[1], phi7[2]], [phi8[0], phi8[1], phi8[2]], [phi9[0], phi9[1], phi9[2]]]],
                            dtype=float)
                        second_cell_list.append(R2)
                        second_cell_id = (list_of_index[:] == R2).prod(axis=1)
                        second_cell_id = np.argwhere(second_cell_id).flatten()
                        third_cell_list.append(R3)
                        third_cell_id = (list_of_index[:] == R3).prod(axis=1)
                        third_cell_id = np.argwhere(third_cell_id).flatten()
                        for alpha in range(3):
                            for beta in range(3):
                                for gamma in range(3):
                                    #coords.append((atom_i, alpha, second_cell_id[0], atom_j, beta, third_cell_id[0], atom_k, gamma))
                                    #frcs.append(phi[alpha, beta, gamma])
                                    frcs[atom_i, alpha, second_cell_id[0], atom_j, beta, third_cell_id[0], atom_k, gamma] = phi[alpha, beta, gamma]

            sparse_frcs = []
            for n1 in range(n_unit_atoms):
                for a in range(3):
                    for nr1 in range(n_replicas):
                        for n2 in range(n_unit_atoms):
                            for b in range(3):
                                for nr2 in range(n_replicas):
                                    for n3 in range(n_unit_atoms):
                                        for c in range(3):
                                            coords.append((n1, a, nr1, n2, b, nr2, n3, c))
                                            sparse_frcs.append(frcs[n1, a, nr1, n2, b, nr2, n3, c])


            third_ifcs = COO(np.array(coords).T, np.array(sparse_frcs), shape=(n_unit_atoms, 3, n_replicas, n_unit_atoms, 3, n_replicas, n_unit_atoms, 3))

            third_ifcs.reshape((n_unit_atoms * 3, n_replicas * n_unit_atoms * 3, n_replicas * n_unit_atoms * 3))
            third_order = cls(atoms=uc,
                              replicated_positions=sc.positions,
                              supercell=supercell,
                              value=third_ifcs,
                              folder=folder)

        else:
            logging.error('Third order format not recognized: ' + str(format))
            raise ValueError
        return third_order


    def save(self, filename='THIRD', format='sparse', min_force=1e-6):
        folder = self.folder
        filename = folder + '/' + filename
        n_atoms = self.atoms.positions.shape[0]
        if format == 'eskm':
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
        elif format=='sparse':
            config_file = folder + REPLICATED_ATOMS_THIRD_FILE
            ase.io.write(config_file, self.replicated_atoms, format='extxyz')

            save_npz(folder + '/' + THIRD_ORDER_FILE_SPARSE, self.value.reshape((n_atoms * 3 * self.n_replicas *
                                                                           n_atoms * 3, self.n_replicas *
                                                                           n_atoms * 3)).to_scipy_sparse())
        else:
            super(ThirdOrder, self).save(filename, format)



    def calculate(self, calculator, delta_shift=1e-4, distance_threshold=None, is_storing=True, is_verbose=False):
        atoms = self.atoms
        replicated_atoms = self.replicated_atoms
        atoms.set_calculator(calculator)
        replicated_atoms.set_calculator(calculator)
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