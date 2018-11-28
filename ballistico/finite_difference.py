import numpy as np
from ballistico.logger import Logger
from ballistico.constants import *
from scipy.optimize import minimize
import ase.io as io
import os
import ballistico.atoms_helper as atoms_helper
import numpy as np
import ase.io
import os
import ballistico.constants as constants
from ase import Atoms
from ballistico.logger import Logger
import thirdorder_core
from thirdorder_common import *
from thirdorder_espresso import gen_supercell, calc_frange, calc_dists
from ase.calculators.espresso import Espresso


import ase.io as io

SECOND_ORDER_FILE = 'second.npy'
THIRD_ORDER_FILE = 'third.npy'

REPLICATED_ATOMS_FILE = 'replicated_atoms.xyz'
LIST_OF_INDEX_FILE = 'list_of_index.npy'


class FiniteDifference (object):
    def __init__(self, atoms, supercell=(1,1,1), second_order=None, third_order=None, calculator=None, calculator_inputs=None, pseudopotentials=None, is_persistency_enabled=True, delta_shift=1e-5, folder='displacement', optimization_method=None, tolerance=None):
        self.atoms = atoms
        self.supercell = supercell

        self._replicated_atoms = None
        #TODO: move list of index in phonon class
        self._list_of_index = None
        self.calculator = calculator
        self.calculator_inputs = calculator_inputs
        self.pseudopotentials = pseudopotentials
        self._second_order = None
        self._third_order = None
        self.is_persistency_enabled = is_persistency_enabled
        self.folder_name = folder
        if self.is_persistency_enabled:
            if not os.path.exists (folder):
                os.makedirs (folder)
        if second_order is not None:
            self.second_order = second_order
        if third_order is not None:
            self.third_order = third_order
        self.second_order_delta = delta_shift
        self.third_order_delta = delta_shift
        
        if optimization_method is not None:
            self.optimize(optimization_method, tolerance)
            

    @property
    def poscar(self):
        return self._poscar

    @poscar.getter
    def poscar(self):
        self._poscar = atoms_helper.convert_to_poscar(self.atoms)
        return self._poscar

    @poscar.setter
    def poscar(self, new_poscar):
        atoms, _ = atoms_helper.convert_to_atoms_and_super_cell(new_poscar)
        self._poscar = new_poscar
        self.atoms = atoms

    @property
    def replicated_poscar(self):
        return self._replicated_poscar

    @replicated_poscar.getter
    def replicated_poscar(self):
        na, nb, nc = self.supercell
        # self._replicated_poscar = self.gen_supercell ()
        self._replicated_poscar = gen_supercell (self.poscar, na, nb, nc)

        return self._replicated_poscar


    @property
    def second_order(self):
        return self._second_order

    @second_order.getter
    def second_order(self):
        if self._second_order is None:
            if self.is_persistency_enabled:
                try:
                    folder = self.folder_name
                    folder += '/'
                    self._second_order = np.load (folder + SECOND_ORDER_FILE)
                except FileNotFoundError as e:
                    Logger().info(e)
            if self._second_order is None:
                self.second_order = self.calculate_second ()
        return self._second_order

    @second_order.setter
    def second_order(self, new_second_order):
        if self.is_persistency_enabled:
            folder = self.folder_name
            folder += '/'
            np.save (folder + SECOND_ORDER_FILE, new_second_order)
        self._second_order = new_second_order

    @property
    def third_order(self):
        return self._third_order

    @third_order.getter
    def third_order(self):
        if self._third_order is None:
            if self.is_persistency_enabled:
                try:
                    folder = self.folder_name
                    folder += '/'
                    self._third_order = np.load (folder + THIRD_ORDER_FILE)
                except FileNotFoundError as e:
                    Logger().info(e)
            if self._third_order is None:
                self.third_order = self.calculate_third ()
        return self._third_order

    @third_order.setter
    def third_order(self, new_third_order):
        if self.is_persistency_enabled:
            folder = self.folder_name
            folder += '/'
            np.save (folder + THIRD_ORDER_FILE, new_third_order)
        self._third_order = new_third_order

    @property
    def replicated_atoms(self):
        return self._replicated_atoms

    @replicated_atoms.getter
    def replicated_atoms(self):
        if self._replicated_atoms is None:
            if self.is_persistency_enabled:
                try:
                    folder = self.folder_name
                    folder += '/'
                    self._replicated_atoms = ase.io.read (folder + REPLICATED_ATOMS_FILE, format='extxyz')
                except FileNotFoundError as e:
                    Logger().info(e)
            if self._replicated_atoms is None:
                # atoms = self.atoms
                # supercell = self.supercell
                # # TODO: refactor removing atoms object
                # n_replicas = supercell[0] * supercell[1] * supercell[2]
                # replica_id = 0
                # list_of_index = np.zeros ((n_replicas, 3))
                #
                # range_0 = np.arange (int (supercell[0]))
                # range_0[range_0 > supercell[0] / 2] = range_0[range_0 > supercell[0] / 2] - supercell[0]
                # range_1 = np.arange (int (supercell[1]))
                # range_1[range_1 > supercell[1] / 2] = range_1[range_1 > supercell[1] / 2] - supercell[1]
                # range_2 = np.arange (int (supercell[2]))
                # range_2[range_2 > supercell[2] / 2] = range_2[range_2 > supercell[2] / 2] - supercell[2]
                #
                # for lx in range_0:
                #     for ly in range_1:
                #         for lz in range_2:
                #             index = np.array ([lx, ly, lz])
                #             list_of_index[replica_id] = index
                #             replica_id += 1
                #
                # list_of_index = list_of_index.dot (self.atoms.cell)
                #
                # replicated_symbols = []
                # n_replicas = list_of_index.shape[0]
                # n_unit_atoms = len (atoms.numbers)
                # replicated_geometry = np.zeros ((n_replicas, n_unit_atoms, 3))
                # for i in range (n_replicas):
                #     vector = list_of_index[i]
                #     replicated_symbols.extend (atoms.get_chemical_symbols ())
                #     replicated_geometry[i, :, :] = atoms.positions + vector
                # replicated_geometry = replicated_geometry.reshape ((n_replicas * n_unit_atoms, 3))
                # replicated_cell = atoms.cell * supercell
                # replicated_atoms = Atoms (positions=replicated_geometry,
                #                           symbols=replicated_symbols, cell=replicated_cell, pbc=[1, 1, 1])
                # self.replicated_atoms = replicated_atoms
                replicated_poscar = self.replicated_poscar
                self.replicated_atoms, _ = atoms_helper.convert_to_atoms_and_super_cell (replicated_poscar)

        return self._replicated_atoms

    @replicated_atoms.setter
    def replicated_atoms(self, new_replicated_atoms):
        if self.is_persistency_enabled:
            folder = self.folder_name
            folder += '/'
            ase.io.write (folder + REPLICATED_ATOMS_FILE, new_replicated_atoms, format='extxyz')
        self._replicated_atoms = new_replicated_atoms

    @property
    def list_of_index(self):
        return self._list_of_index

    @list_of_index.getter
    def list_of_index(self):
        if self._list_of_index is None:
            if self.is_persistency_enabled:
                try:
                    folder = self.folder_name
                    folder += '/'
                    self._list_of_index = np.load (folder + LIST_OF_INDEX_FILE)
                except FileNotFoundError as e:
                    Logger().info(e)
            if self._list_of_index is None:
                n_replicas = np.prod(self.supercell)
                atoms = self.atoms
                n_unit_atoms = self.atoms.positions.shape[0]
                list_of_replicas = (self.replicated_atoms.positions.reshape ((n_replicas, n_unit_atoms, 3)) - atoms.positions[np.newaxis,:, :])
                self.list_of_index =  list_of_replicas[:,0,:]
        return self._list_of_index

    @list_of_index.setter
    def list_of_index(self, new_list_of_index):
        if self.is_persistency_enabled:
            folder = self.folder_name
            folder += '/'
            np.save (folder + LIST_OF_INDEX_FILE, new_list_of_index)
        self._list_of_index = new_list_of_index

    def gen_supercell(self):
        atoms = self.atoms
        supercell = self.supercell
        n_replicas = supercell[0] * supercell[1] * supercell[2]
    
        range_0 = np.arange (int (supercell[0]))
        range_0[range_0 > supercell[0] / 2] = range_0[range_0 > supercell[0] / 2] - supercell[0]
        range_1 = np.arange (int (supercell[1]))
        range_1[range_1 > supercell[1] / 2] = range_1[range_1 > supercell[1] / 2] - supercell[1]
        range_2 = np.arange (int (supercell[2]))
        range_2[range_2 > supercell[2] / 2] = range_2[range_2 > supercell[2] / 2] - supercell[2]

        replica_id = 0
        replica_positions = np.zeros ((n_replicas, 3))
        for ly in range_1:
            for lx in range_0:
                for lz in range_2:
                    index = np.array ([lx, ly, lz])
                    replica_positions[replica_id] = index
                    replica_id += 1

        replica_positions = replica_positions.dot (self.atoms.cell)
        replicated_symbols = []
        n_replicas = replica_positions.shape[0]
        n_unit_atoms = len (atoms.numbers)
    
        for i in range (n_replicas):
            replicated_symbols.extend (atoms.get_chemical_symbols ())
        replicated_positions = atoms.positions[np.newaxis, :, :] + replica_positions[:, np.newaxis, np.newaxis]
        replicated_positions = replicated_positions.reshape ((n_replicas * n_unit_atoms, 3))
        replicated_cell = atoms.cell * supercell
        replicated_atoms = Atoms (positions=replicated_positions,
                                  symbols=replicated_symbols, cell=replicated_cell, pbc=[1, 1, 1])
        
        return atoms_helper.convert_to_poscar(replicated_atoms, supercell)

    def optimize(self, method, tol=None):
        # Mimimization of the sructure
        if not ((method == 'none') or (method == False)):
            Logger ().info ('Initial max force: ' + str(self.max_force (self.atoms.positions, self.atoms)))
            Logger().info('Optimization method ' + method)
            result = minimize (self.max_force, self.atoms.positions, args=self.atoms, jac=self.gradient, method=method, tol=tol)
            Logger().info(result.message)
            self.atoms.positions = result.x.reshape((int(result.x.size / 3), 3))
            io.write ('minimized_' + str(self) + '.xyz', self.atoms, format='extxyz')
            Logger ().info ('Final max force: ' + str(self.max_force (self.atoms.positions, self.atoms)))
            return self.max_force(self.atoms.positions, self.atoms)

    def max_force(self, x, atoms):
        grad = self.gradient (x, atoms)
        return np.linalg.norm (grad, 2)

    def gradient(self, x, input_atoms):
        atoms = input_atoms.copy ()
        atoms.positions = np.reshape (x, (int (x.size / 3.), 3))
        if self.calculator == ase.calculators.lammpslib.LAMMPSlib:
            calc = self.calculator(lmpcmds=self.calculator_inputs, log_file='log_lammps.out')
        else:
            calc = Espresso (pseudopotentials=self.pseudopotentials,
                             tstress=True, tprnfor=True,  # kwargs added to parameters
                             input_data=self.calculator_inputs)
        atoms.set_calculator (calc)
        gr = -1. * atoms.get_forces ()
        grad = np.reshape (gr, gr.size)
        return grad
    
        
    def calculate_second(self):
        '''
        Calculate the second order derivative of the force using finite differences
        :return:
        tensor with second derivative in eV/A^2
        '''
        supercell = self.supercell
        atoms = self.atoms
        Logger().info ('Calculating second order potential derivatives')
        n_in_unit_cell = len (atoms.numbers)
        replicated_atoms = self.replicated_atoms

        n_atoms = len (replicated_atoms.numbers)
        
        dx = self.second_order_delta
        second = np.zeros ((n_atoms * 3, n_atoms * 3))
        for alpha in range (3):
            for i in range (n_atoms):
                Logger().info('atom ' + str(i) + ', direction ' + str(alpha))

                for move in (-1, 1):
                    shift = np.zeros ((n_atoms, 3))
                    shift[i, alpha] += move * dx
                    second[i * 3 + alpha, :] += move * self.gradient (replicated_atoms.positions + shift, replicated_atoms)
                    
        n_supercell = int(replicated_atoms.positions.shape[0] / n_in_unit_cell)
        second = second.reshape ((n_supercell, n_in_unit_cell, 3, n_supercell, n_in_unit_cell, 3))
        second = second / (2. * dx)
        second *= evoverdlpoly
        return second
    
    def calculate_third(self):
        atoms = self.atoms
        replicated_atoms = self.replicated_atoms
        
        # TODO: Here we should create it sparse
        n_in_unit_cell = len (atoms.numbers)
        replicated_atoms = replicated_atoms
        n_replicated_atoms = len (replicated_atoms.numbers)
        n_supercell = int(replicated_atoms.positions.shape[0] / n_in_unit_cell)
        dx = self.third_order_delta
        third = np.zeros ((n_in_unit_cell, 3, n_supercell * n_in_unit_cell, 3, n_supercell * n_in_unit_cell * 3))
        # cell_inv = np.linalg.inv (self.replicated_atoms.cell)#*1000


        poscar = self.poscar
        SYMPREC = 1e-5  # Tolerance for symmetry search
        NNEIGH = 4
        FRANGE = None

        natoms = len (poscar["types"])
        Logger().info("Analyzing symmetries")
        symops = thirdorder_core.SymmetryOperations (
            poscar["lattvec"], poscar["types"], poscar["positions"].T, SYMPREC)
        Logger().info("- Symmetry group {0} detected".format (symops.symbol))
        Logger().info("- {0} symmetry operations".format (symops.translations.shape[0]))
        Logger().info("Creating the supercell")
        na, nb, nc = self.supercell
        sposcar = self.replicated_poscar
        ntot = natoms * na * nb * nc
        Logger().info("Computing all distances in the supercell")
        dmin, nequi, shifts = calc_dists (sposcar)
        if NNEIGH != None:
            frange = calc_frange (poscar, sposcar, NNEIGH, dmin)
            Logger().info("- Automatic cutoff: {0} nm".format (frange))
        else:
            frange = FRANGE
            Logger().info("- User-defined cutoff: {0} nm".format (frange))
        Logger().info("Looking for an irreducible set of third-order IFCs")
        wedge = thirdorder_core.Wedge (poscar, sposcar, symops, dmin, nequi, shifts,
                                       frange)
        self.wedge = wedge
        Logger().info("- {0} triplet equivalence classes found".format (wedge.nlist))
        self.list4 = wedge.build_list4 ()
        Logger().info('object created')
        Logger().info ('Calculating third order potential derivatives')
        nirred = len(self.list4)
        na, nb, nc = self.supercell
        phipart = np.zeros((3, nirred, n_replicated_atoms))
        for i, e in enumerate(self.list4):
            jat, iat, jcoord, icoord = e
            Logger().info('atoms ' + str(iat) + ',' + str(jat) + ', direction ' + str(icoord) + ',' + str(jcoord))

        # for iat in range(n_in_unit_cell):
        #     for icoord in range(3):
        #         for jat in range(n_supercell * n_in_unit_cell):
        #             for jcoord in range(3):
            for n in range(4):
                shift = np.zeros ((n_replicated_atoms, 3))
                isign = (-1)**(n // 2)
                jsign = -(-1)**(n % 2)
                delta = np.zeros(3)
                delta[icoord] = isign * dx
                shift[iat, :] += delta
                delta = np.zeros(3)
                delta[jcoord] = jsign * dx
                shift[jat, :] += delta
                phipart[:, i, :] += isign * jsign * (-1 * self.gradient (replicated_atoms.positions + shift, replicated_atoms)
                                                     .reshape(replicated_atoms.positions.shape).T)

                # third[iat, icoord, jat, jcoord, :] += isign * jsign * (
                #     -1. * self.gradient (replicated_atoms.positions + shift, replicated_atoms))
        phifull = np.array(thirdorder_core.reconstruct_ifcs (phipart, self.wedge, self.list4,
                                          self.poscar, self.replicated_poscar))
        phifull = phifull.swapaxes(3, 2).swapaxes(1, 2).swapaxes(0, 1)
        phifull = phifull.swapaxes(4, 3).swapaxes(3, 2)
        phifull = phifull.swapaxes(5, 4)

        phifull = phifull.reshape ((n_in_unit_cell, 3, n_supercell, n_in_unit_cell, 3, n_supercell, n_in_unit_cell, 3))
        phifull = phifull[np.newaxis, :, :, :, :, :]/(4 * dx * dx)
        return phifull
        return third
        # third = np.zeros ((n_in_unit_cell, 3, n_supercell * n_in_unit_cell, 3, n_supercell * n_in_unit_cell * 3))
        # list4 = self.list4
        # nirred = len(self.list4)
        # na, nb, nc = self.supercell
        # ntot = n_atoms * na * nb * nc
        #
        # phipart = np.zeros((3, nirred, ntot))
        # cell_inv = np.linalg.inv(self.replicated_atoms.cell)
        # phifull = np.zeros((n_in_unit_cell, 3, n_supercell * n_in_unit_cell, 3, n_supercell * n_in_unit_cell, 3))
        #
        # for i, e in enumerate(list4):
        #     shift = np.zeros ((ntot, 3))
        #     jat, iat, jcoord, icoord = e
        # for iat in range(n_in_unit_cell):
        #     for icoord in range(3):
        #         for jat in range(n_supercell * n_in_unit_cell):
        #             for jcoord in range(3):
        #                 for n in range(4):
        #                     shift = np.zeros((ntot, 3))
        #                     isign = (-1)**(n // 2)
        #                     jsign = -(-1)**(n % 2)
        #                     icoord_shift = np.zeros(3)
        #                     icoord_shift[icoord] += dx * isign
        #                     jcoord_shift = np.zeros(3)
        #                     jcoord_shift[jcoord] += dx * jsign
        #
        #                     shift[iat, :] += cell_inv.dot(icoord_shift)
        #                     shift[iat, :] += (icoord_shift)
        #                     shift[jat, :] += cell_inv.dot(jcoord_shift)
                            # shift[jat, :] += (jcoord_shift)
                            # phifull[iat, icoord, jat, jcoord] += isign * jsign * (
                            #             -1. * self.gradient (replicated_atoms.positions + shift, replicated_atoms).reshape(replicated_atoms.positions.shape))
                            #
                            # phipart[:, i, :] += isign * jsign * (-1 * self.gradient (replicated_atoms.positions + shift, replicated_atoms))
        # phifull = np.array(thirdorder_core.reconstruct_ifcs (phipart, self.wedge, list4,
        #                                   self.poscar, self.replicated_poscar))
        # phifull = phifull.swapaxes(3, 2).swapaxes(1, 2).swapaxes(0, 1)
        # phifull = phifull.swapaxes(4, 3).swapaxes(3, 2)
        # phifull = phifull.swapaxes(5, 4)
        #
        # phifull = phifull.reshape ((n_in_unit_cell, 3, n_supercell, n_in_unit_cell, 3, n_supercell, n_in_unit_cell, 3))
        # phifull = phifull[np.newaxis, :, :, :, :, :]/(4 * dx * dx)
        # return phifull

    