import numpy as np
from ballistico.phonons import Phonons
import pandas as pd
from ballistico.helpers.storage import LAZY_PREFIX
from scipy.linalg.lapack import zheev
from ase.units import Rydberg, Bohr
from ase import units
from ballistico.grid import Grid


def myexp(input):
    # input = -1j * input
    return np.cos(input.real) - 1j * np.sin(input.real)

def q_points_to_read(folder):
    q_points = pd.read_csv(folder + '/BTE.qpoints_full', header=None, delim_whitespace=True)
    return q_points.values[:,2:5]


def wrap_coords_shen(mmm, scell):
    m1, m2, m3 = mmm
    t1 = np.mod(m1, scell[0])
    if (t1 < 0):
        t1 = t1 + scell[0]
    t2 = np.mod(m2, scell[1])
    if (t2 < 0):
        t2 = t2 + scell[1]
    t3 = np.mod(m3, scell[2])
    if (t3 < 0):
        t3 = t3 + scell[3]
    return np.array([t1, t2, t3])



class ShengBTEPhonons(Phonons):

    @Phonons.frequency.setter
    def frequency(self, loaded_attr):
        attr = LAZY_PREFIX + 'frequency'
        setattr(self, attr, loaded_attr)

    @Phonons._eigensystem.setter
    def _eigensystem(self, loaded_attr):
        attr = LAZY_PREFIX + '_eigensystem'
        setattr(self, attr, loaded_attr)

    @Phonons.velocity.setter
    def velocity(self, loaded_attr):
        attr = LAZY_PREFIX + 'velocity'
        setattr(self, attr, loaded_attr)


    @Phonons.bandwidth.setter
    def bandwidth(self, loaded_attr):
        attr = LAZY_PREFIX + 'bandwidth'
        setattr(self, attr, loaded_attr)

    @Phonons.phase_space.setter
    def phase_space(self, loaded_attr):
        attr = LAZY_PREFIX + 'phase_space'
        setattr(self, attr, loaded_attr)



    @classmethod
    def from_shengbte(cls, finite_difference, kpts, temperature, folder):
        is_classic=False
        phonons = cls(finite_difference=finite_difference,
                          kpts=kpts,
                          is_classic=is_classic,
                          temperature=temperature,
                          folder=folder,
                          storage='memory')

        new_shape = [phonons.n_k_points, phonons.n_modes]
        phonons.frequency = phonons.read_energy_data().reshape(new_shape) / (2 * np.pi)
        phonons.velocity = (phonons.read_velocity_data() * 10).reshape((phonons.n_k_points, phonons.n_modes, 3))
        phonons.bandwidth = phonons.read_decay_rate_data().reshape(new_shape)
        phonons.phase_space = phonons.read_ps_data().reshape(new_shape)
        phonons.is_able_to_calculate = False
        return phonons


    def qpoints_mapper(self):
        q_points = pd.read_csv (self.folder + '/BTE.qpoints_full', header=None, delim_whitespace=True)
        return q_points.values


    def irreducible_indices(self):
        return np.unique(self.qpoints_mapper()[:,1])


    def q_points(self):
        return self.qpoints_mapper()[:,2:5]


    def read_energy_data(self):
        # We read in rad/ps
        omega = pd.read_csv (self.folder + '/BTE.omega', header=None, delim_whitespace=True)
        n_qpoints = self.qpoints_mapper().shape[0]
        n_branches = omega.shape[1]
        energy_data = np.zeros ((n_qpoints, n_branches))
        for index, reduced_index, q_point_x, q_point_y, q_point_z in self.qpoints_mapper():
            energy_data[int (index - 1)] = omega.loc[[int (reduced_index - 1)]].values
        return energy_data


    def read_ps_data(self, type=None):
        if type == 'plus':
            file = 'BTE.WP3_plus'
        elif type == 'minus':
            file = 'BTE.WP3_minus'
        else:
            file = 'BTE.WP3'
        temperature = str (int (self.temperature))
        decay = pd.read_csv (self.folder + '/T' + temperature + 'K/' + file, header=None,
                             delim_whitespace=True)
        # decay = pd.read_csv (self.folder + 'T' + temperature +
        # 'K/BTE.w_anharmonic', header=None, delim_whitespace=True)
        n_branches = int (decay.shape[0] / self.irreducible_indices().max ())
        n_qpoints_reduced = int (decay.shape[0] / n_branches)
        n_qpoints = self.qpoints_mapper().shape[0]
        decay = np.delete (decay.values, 0, 1)
        decay = decay.reshape ((n_branches, n_qpoints_reduced))
        decay_data = np.zeros ((n_qpoints, n_branches))
        for index, reduced_index, q_point_x, q_point_y, q_point_z in self.qpoints_mapper():
            decay_data[int (index - 1)] = decay[:, int (reduced_index - 1)]
        return decay_data


    def read_decay_rate_data(self, type=None):
        if type == 'plus':
            file = 'BTE.w_anharmonic_plus'
        elif type == 'minus':
            file = 'BTE.w_anharmonic_minus'
        else:
            file = 'BTE.w'
        temperature = str(int(self.temperature))
        decay = pd.read_csv (self.folder + '/T' + temperature + 'K/' + file, header=None,
                             delim_whitespace=True)
        # decay = pd.read_csv (self.folder + 'T' + temperature +
        # 'K/BTE.w_anharmonic', header=None, delim_whitespace=True)
        n_branches = int (decay.shape[0] /self. irreducible_indices().max ())
        n_qpoints_reduced = int (decay.shape[0] / n_branches)
        n_qpoints = self.qpoints_mapper().shape[0]
        decay = np.delete(decay.values,0,1)
        decay = decay.reshape((n_branches, n_qpoints_reduced))
        decay_data = np.zeros ((n_qpoints, n_branches))
        for index, reduced_index, q_point_x, q_point_y, q_point_z in self.qpoints_mapper():
            decay_data[int (index - 1)] = decay[:, int(reduced_index-1)]
        return decay_data


    def read_velocity_data(self):
        shenbte_folder = self.folder
        velocity = pd.read_csv (shenbte_folder + '/BTE.v_full', header=None, delim_whitespace=True)
        n_velocity = velocity.shape[0]
        n_kpts = self.n_k_points
        n_modes = int(n_velocity / n_kpts)

        velocity_array = velocity.values.reshape (n_modes, n_kpts, 3)

        velocity = np.zeros((self.kpts[0], self.kpts[1], self.kpts[2], n_modes, 3))

        z = 0
        for k in range (self.kpts[2]):
            for j in range(self.kpts[1]):
                for i in range (self.kpts[0]):
                    velocity[i, j, k, :, :] = velocity_array[:, z, :]
                    z += 1
        return velocity


    def read_conductivity(self, converged=False):
        folder = self.folder
        if converged:
            conduct_file = '/BTE.KappaTensorVsT_CONV'
        else:
            conduct_file = '/BTE.KappaTensorVsT_RTA'

        conductivity_array = np.loadtxt (folder + conduct_file)
        conductivity_array = np.delete (conductivity_array, 0)
        n_steps = 0
        if converged:
            n_steps = int (conductivity_array[-1])
            conductivity_array = np.delete (conductivity_array, -1)

        conductivity = conductivity_array.reshape (3, 3)
        if converged:
            return conductivity, n_steps
        else:
            return conductivity

    def import_scattering_matrix(self):
        temperature = str(int(self.temperature))
        filename_gamma = self.folder + '/T' + temperature + 'K/GGG.Gamma_Tensor'
        filename_tau_zero = self.folder + '/T' + temperature + 'K/GGG.tau_zero'
        self.tau_zero = np.zeros((self.n_modes, self.n_k_points))
        with open(filename_tau_zero, "r+") as f:
            for line in f:
                items = line.split()
                self.tau_zero[int(items[0]) - 1, int(items[1]) - 1] = float(items[2])

        n0 = []
        n1 = []
        k0 = []
        k1 = []
        gamma_value = []

        with open(filename_gamma, "r+") as f:
            for line in f:
                items = line.split()

                n0.append(int(items[0]) - 1)
                k0.append(int(items[1]) - 1)
                n1.append(int(items[2]) - 1)
                k1.append(int(items[3]) - 1)

                gamma_value.append(float(items[4]))
        gamma_tensor = np.zeros((self.n_k_points, self.n_modes, self.n_k_points,self.n_modes))
        gamma_tensor[k0, n0, k1, n1] = gamma_value
        return gamma_tensor


    @classmethod
    def from_espresso_dyn(cls, finite_difference, supercell, kpts, temperature, folder):
        atoms = finite_difference.atoms
        n_unit_cell = atoms.positions.shape[0]
        mass = atoms.get_masses()
        masses_2d = np.zeros((n_unit_cell, n_unit_cell))
        distance = np.zeros((n_unit_cell, n_unit_cell, 3))
        positions = atoms.positions
        replicated_cell = atoms.cell * supercell

        # TODO: introduce new ase unit, second/THZ
        # use finite difference here and add acoustic sum rule

        ev_s = (units._hplanck) * units.J
        toTHz = 2 * np.pi * units.Rydberg / ev_s * 1e-12 # 20670.687
        # toTHz = 20670.687 # 2 * np.pi * ase.units.Rydberg * 241.8
        # massfactor = 1.8218779 * 6.022e-4
        massfactor = 2 * units._me * units._Nav * 1000

        fc_s = finite_difference.second_order.force_constant / (Rydberg / (Bohr ** 2))
        fc_s = fc_s.reshape((n_unit_cell, 3, supercell[0], supercell[1], supercell[2], n_unit_cell, 3))
        # fc_s = fc_s.transpose(1, 6, 0, 5, 2, 3, 4)

        for i in np.arange(n_unit_cell):
            masses_2d[i, i] = mass[i]
            distance[i, i, :] = 0
            for j in np.arange(i, n_unit_cell):
                masses_2d[i, j] = np.sqrt(mass[i] * mass[j])
                distance[i, j, :3] = positions[i, :3] - positions[j, :3]
                masses_2d[j, i] = masses_2d[i, j]
                distance[j, i, :3] = -distance[i, j, :3]
        masses_2d = masses_2d / massfactor
        j = 0
        list_of_replicated_cells = np.zeros((124, 3))
        norm_of_replicated_cells = np.zeros((124))
        for id_0 in np.arange(-2, 3):
            for id_1 in np.arange(-2, 3):
                for id_2 in np.arange(-2, 3):
                    if (np.all([id_0, id_1, id_2] == [0, 0, 0])):
                        continue
                    for i in np.arange(3):
                        list_of_replicated_cells[j, i] = replicated_cell[0, i] * id_0 + replicated_cell[1, i] * id_1 + replicated_cell[2, i] * id_2
                    norm_of_replicated_cells[j] = 0.5 * np.dot(list_of_replicated_cells[j, :3], list_of_replicated_cells[j, :3])
                    j = j + 1
        nk = np.prod(np.array(kpts))
        veckspace = Grid(kpts, order='C').unitary_grid()
        # kspace = np.zeros((nk, 3))
        # veckspace = np.zeros((nk, 3))
        # for ii in range(kpts[0]):
        #     for jj in range(kpts[1]):
        #         for kk in range(kpts[2]):
        #             indexK=((kk)*kpts[2]+(jj))*kpts[1]+ii
        #             vec = np.array([ii/kpts[0], jj/kpts[1], kk/kpts[2]])
        #             veckspace[indexK,:]=vec

        n_kpoints = kpts[0] * kpts[1] * kpts[2]
        dyn_s = np.zeros((n_kpoints, n_unit_cell * 3, n_unit_cell * 3), dtype=np.complex)
        eigenvects = np.zeros((n_kpoints, n_unit_cell * 3, n_unit_cell * 3), dtype=np.complex)
        ddyn_s = np.zeros((n_kpoints, n_unit_cell * 3, n_unit_cell * 3, 3), dtype=np.complex)


        for iat in np.arange(n_unit_cell):
            for jat in np.arange(n_unit_cell):
                total_weight = 0.0
                for replica_0 in np.arange(-2 * supercell[0], 2 * supercell[0] + 1):
                    for replica_1 in np.arange(-2 * supercell[1], 2 * supercell[1] + 1):
                        for replica_2 in np.arange(-2 * supercell[2], 2 * supercell[2] + 1):

                            # print(replica_0, m2, replica_2)
                            replica_id = np.array([replica_0, replica_1, replica_2])
                            replica_position = np.tensordot(replica_id, atoms.cell, (-1, 0))
                            atom_absolute_position = replica_position + distance[iat, jat]
                            weight = 0.
                            nreq = 1
                            j = 0
                            for ir in np.arange(124):
                                ck = np.dot(atom_absolute_position, list_of_replicated_cells[ir, :3])\
                                     - norm_of_replicated_cells[ir]
                                if (ck > 1e-6):
                                    j = 1
                                    continue
                                if (abs(ck) <= 1e-6):
                                    nreq = nreq + 1
                            if (j == 0):
                                weight = 1.0 / (nreq)

                            ttt = np.zeros_like(replica_id)
                            if (weight > 0.0):
                                t1, t2, t3 = wrap_coords_shen(replica_id, supercell)
                                # t1, t2, t3 = wrap_coordinates(replica_id, np.diag(supercell)).astype(np.int)

                                # print('id_cell_studied replica_id: ', replica_id)
                                # print('exponent, R_t: ',t[0], t[1], t[2])
                                # print('dynmat(t1, t2, t3): ',t1, t2, t3)
                                # print('-')

                                for ik in np.arange(nk):
                                    # t = np.dot(replica_id, atoms.cell)
                                    # kt = 2. * np.pi * np.dot(np.dot(cell_inv, veckspace[ik, :3]), t[:3])

                                    kt = 2. * np.pi * np.dot(veckspace[ik, :], replica_id[:])

                                    for ipol in np.arange(3):
                                        idim = (iat) * 3 + ipol
                                        for jpol in np.arange(3):
                                            jdim = (jat) * 3 + jpol

                                            dyn_s[ik, idim, jdim] = dyn_s[ik, idim, jdim] + fc_s[
                                                iat, ipol, t1, t2, t3, jat, jpol] * myexp(kt) * weight
                                            ddyn_s[ik, idim, jdim, :3] = ddyn_s[ik, idim, jdim, :3] - 1j * replica_position[:3] * fc_s[
                                                iat, ipol, t1, t2, t3, jat, jpol] * myexp(kt) * weight

                            total_weight = total_weight + weight
        frequency = np.zeros((nk, n_unit_cell * 3))
        velocities = np.zeros((nk, n_unit_cell * 3, 3))
        nbands = n_unit_cell * 3
        for ik in np.arange(nk):
            dyn = dyn_s[ik, :, :]
            ddyn = ddyn_s[ik, :, :, :]
            for ipol in np.arange(3):
                for jpol in np.arange(3):
                    for iat in np.arange(n_unit_cell):
                        for jat in np.arange(n_unit_cell):
                            idim = (iat) * 3 + ipol
                            jdim = (jat) * 3 + jpol
                            dyn[idim, jdim] = dyn[idim, jdim] / masses_2d[iat, jat]
                            ddyn[idim, jdim, :3] = ddyn[idim, jdim, :3] / masses_2d[iat, jat]

            omega2,eigenvect,info = zheev(dyn)
            eigenvects[ik] = eigenvect
            frequency[ik, :] = np.sign(omega2) * np.sqrt(np.abs(omega2))

            for i in np.arange(nbands):
                for j in np.arange(3):
                    velocities[ik, i, j] = np.real(np.tensordot(eigenvect[:, i].conj(), np.tensordot(ddyn[:, :, j], eigenvect[:, i], (-1, 0)), (-1, 0)))
                velocities[ik, i, :] = velocities[ik, i, :] / (2. * frequency[ik, i])

        frequency = frequency * toTHz / np.pi / 2
        velocities = velocities * toTHz

        is_classic=False
        phonons = cls(finite_difference=finite_difference,
                          kpts=kpts,
                          is_classic=is_classic,
                          temperature=temperature,
                          folder=folder,
                          storage='memory',
                      is_tf_backend=False,
                      grid_type='C')

        phonons._eigensystem = np.zeros((phonons.n_k_points, phonons.n_modes + 1, phonons.n_modes), dtype=np.complex)
        phonons._eigensystem[:, 1:, :] = eigenvects#.swapaxes(1, 2)
        phonons._eigensystem[:, 0, :] = (2 * np.pi * frequency) ** 2#.swapaxes(1, 2)

        new_shape = [phonons.kpts[0] * phonons.kpts[1] * phonons.kpts[2], phonons.n_modes]
        phonons.frequency = frequency.reshape(new_shape)
        phonons.velocity = velocities
        phonons.is_able_to_calculate = False
        # phonons.bandwidth = phonons.read_decay_rate_data()

        return phonons