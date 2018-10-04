
import ballistico.geometry_helper as ghl
from ballistico.constants import *
from ballistico.tools import *
import matplotlib.pyplot as plt
import ballistico.atoms_helper as ath

# np.set_printoptions(suppress=True)
from scipy.interpolate import RegularGridInterpolator
#from ballistico.interpolation_controller import resample_fourier

# LENGTH_THREESHOLD = 1e31
# THREESHOLD = 1e-31
BUFFER_PLOT = .2

class ShengbteHelper (object):
    def __init__(self, system, k_size, parameters={}):
        self.system = system
        self.k_size = k_size
        
        self._qpoints_mapper = None
        self._energies = None
        self._decay_rate_data = None
        self._decay_rates_2 = None
        self._velocity_data = None
        self._frequencies = None
        
        
        # TODO: move these initializations into a getter
        if 'convergence' in parameters:
            self.convergence = parameters['convergence']
        else:
            self.convergence = False
            
        if 'only_gamma' in parameters:
            self.only_gamma = parameters['only_gamma']
        else:
            self.only_gamma = False
            

        if 'post_processing' in parameters:
            self.post_processing = parameters['post_processing']
        else:
            self.post_processing = None
    
        self.length = np.zeros(3)
        if 'l_x' in parameters:
            self.length[0] = parameters['l_x']
        # else:
            # self.length[0] = LENGTH_THREESHOLD
            
            
        if 'l_y' in parameters:
            self.length[1] = parameters['l_y']
        # else:
            # self.length[1] = LENGTH_THREESHOLD
        #
        if 'l_z' in parameters:
            self.length[2] = parameters['l_z']
        # else:
            # self.length[2] = LENGTH_THREESHOLD
            
        if 'classical' in parameters:
            self.is_classical = parameters['classical']
        # else:
            # self.is_classical = LENGTH_THREESHOLD
        
        self.folder = get_current_folder () + '/' + str(self) + '/'
        is_fold_present = is_folder_present(self.folder)
        
        # if not is_fold_present:
        # TODO: n_processors shouldnt be hardcoded

        self.create_control_file ()
        



    
    @property
    def qpoints_mapper(self):
        return self._qpoints_mapper
    
    @qpoints_mapper.getter
    def qpoints_mapper(self):
        if self._qpoints_mapper is None:
            self.read_qpoints_mapper ()
        return self._qpoints_mapper
    
    @qpoints_mapper.setter
    def qpoints_mapper(self, new_qpoints_mapper):
        self._qpoints_mapper = new_qpoints_mapper
    
    @property
    def energies(self):
        return self._energies
    
    @energies.getter
    def energies(self):
        if self._energies is None:
            self.read_energy_data ()
        return self._energies
    
    @energies.setter
    def energies(self, new_energy_data):
        self._energies = new_energy_data
    
    @property
    def frequencies(self):
        return self._frequencies
    
    @frequencies.getter
    def frequencies(self):
        if self._frequencies is None:
            # self._frequencies = self.energies.reshape ((self.k_mesh[0], self.k_mesh[1], self.k_mesh[2], self.energies.shape[1])).swapaxes (0, 2) / 2. / np.pi
            self._frequencies = self.energies.reshape ((self.k_size[0], self.k_size[1], self.k_size[2], self.energies.shape[1])) / 2. / np.pi
        return self._frequencies
    
    @frequencies.setter
    def frequencies(self, new_frequencies_data):
        self._frequencies = new_frequencies_data
    
    @property
    def decay_rates(self):
        return self._decay_rate_data
    
    @decay_rates.getter
    def decay_rates(self):
        if self._decay_rate_data is None:
            self._decay_rate_data = self.read_decay_rate_data ()
            # self._decay_rate_data[np.where (np.isnan (self._decay_rate_data))] = THREESHOLD
        return self._decay_rate_data
    
    @decay_rates.setter
    def decay_rates(self, new_decay_rate_data):
        self._decay_rate_data = new_decay_rate_data
    
    @property
    def decay_rates_2(self):
        return self._decay_rates_2
    
    @decay_rates_2.getter
    def decay_rates_2(self):
        if self._decay_rates_2 is None:
            self.import_scattering_matrix ()
        return self._decay_rates_2 #+ THREESHOLD
    
    @decay_rates_2.setter
    def decay_rates_2(self, new_decay_rate_data):
        self._decay_rates_2 = new_decay_rate_data
    
    @property
    def velocities(self):
        return self._velocity_data
    
    @velocities.getter
    def velocities(self):
        if self._velocity_data is None:
            self.read_velocity_data ()
        return self._velocity_data #+ THREESHOLD
    
    @velocities.setter
    def velocities(self, new_velocity_data):
        self._velocity_data = new_velocity_data


    # def read_file(self, filename):
        
        # file = pd.read_csv (self.folder + filename, header=None, delim_whitespace=True)
        # read_array = file.values
        # read_array = read_array.reshape (read_array.size)
        # # check for nan
        # # read_array = read_array[~np.isnan (read_array)]
        # if read_array.size == 1:
        #     read_array = int (read_array)
        # return read_array

    def read_file(self):
        filename = 'fort.11101'
        file = pd.read_csv (self.folder + filename, header=None, delim_whitespace=True)
        n_particles = int(self.n_modes()/3)
        dynmat = np.zeros( (n_particles* 3, n_particles*  3)).astype(np.complex)
        for i in range(self.n_modes() ** 2):
            record = file.iloc[i]
            dynmat[int(record[0])-1, int(record[1])-1] = record[2] + 1j * record[3]

        filename = 'fort.11102'

        file = pd.read_csv (self.folder + filename, header=None, delim_whitespace=True)
        n_particles = int (self.n_modes () / 3)
        ddynmat = np.zeros ((3, n_particles, 3, n_particles, 3)).astype (np.complex)
        for i in range (self.n_modes () ** 2):
            record = file.iloc[i]
            ddynmat[int (record[0]) - 1, int (record[1]) - 1, int (record[2]) - 1, int (record[3]) - 1, int(record[4]) - 1] = record[
                                                                                                             5] + 1j * \
                                                                                                         record[6]

        return dynmat, ddynmat



    def n_phonons(self):
        return self.n_modes() * self.n_k_points()


    def n_modes(self):
        return self.energies.shape[1]
    
    def n_k_points(self):
        return self.energies.shape[0]

    
    
    def save_second_order_qe_matrix(self):
        
        shenbte_folder = self.folder
        harmonic_system = self.system
        filename = 'espresso.ifc2'
        n_particles = harmonic_system.second_order.shape[1]
        
        filename = shenbte_folder + filename
        file = open ('%s' % filename, 'a+')
        
        # harmonic_system.second_order.shape
        file.write (self.header ())

        for alpha in range (3):
            for beta in range (3):
                
                for i in range (n_particles):
                    for j in range (n_particles):
                        file.write ('\t' + str (alpha + 1) + '\t' + str (beta + 1) + '\t' + str (i + 1) + '\t' + str (j + 1) + '\n')
                        for id_replica in range(harmonic_system.list_of_indices.shape[0]):
                            index = harmonic_system.list_of_indices[id_replica]
                            l_1 = int (index[0]) % self.system.replicas[0] + 1
                            l_2 = int (index[1]) % self.system.replicas[1] + 1
                            l_3 = int (index[2]) % self.system.replicas[2] + 1
                            
                            
                            file.write ('\t' + str (l_1) + '\t' + str (l_2) + '\t' + str (l_3))
                            
                            # TODO: Swap indices, index_first_cell should also be removed
                            matrix_element = harmonic_system.second_order[self.system.index_first_cell, j, beta, id_replica, i, alpha]
                            # matrix_element = harmonic_system.second_order[id_replica, j, beta, index_first_cell, i, alpha]
                            
                            matrix_element = matrix_element / evoverdlpoly / rydbergoverev * (bohroverangstrom ** 2)
                            file.write ('\t %.11E' % matrix_element)
                            file.write ('\n')
        file.close ()
        print('second order saved')
    
    
    def save_third_order_matrix_new(self):
        system = self.system
        filename = 'FORCE_CONSTANTS_3RD'
        filename = self.folder + filename
        file = open ('%s' % filename, 'w')
        n_in_unit_cell = len (system.configuration.numbers)
        n_replicas = np.prod (system.replicas)
        block_counter = 0
        # third_order = system.third_order.reshape((n_replicas, n_in_unit_cell, 3, n_replicas, n_in_unit_cell, 3, n_replicas, n_in_unit_cell, 3))
        for i_0 in range (n_in_unit_cell):
            for n_1 in range (n_replicas):
                for i_1 in range (n_in_unit_cell):
                    for n_2 in range (n_replicas):
                        for i_2 in range (n_in_unit_cell):
                            
                            three_particles_interaction = system.third_order[0, i_0, :, n_1, i_1, :, n_2, i_2, :]
                            # three_particles_interaction = system.third_order[system.index_first_cell, i_0, :, n_1, i_1, :, n_2, i_2, :]
                            # three_particles_interaction = system.third_order[i_0, :, n_1, i_1, :, n_2, i_2, :]

                            if (np.abs (three_particles_interaction) > 1e-9).any ():
                                block_counter += 1
                                replica = system.list_of_replicas#
                                file.write ('\n  ' + str (block_counter))
                                rep_position = ath.apply_boundary (system.replicated_configuration,replica[n_1])
                                file.write ('\n  ' + str (rep_position[0]) + ' ' + str (rep_position[1]) + ' ' + str (
                                    rep_position[2]))
                                rep_position = ath.apply_boundary (system.replicated_configuration,replica[n_2])
                                file.write ('\n  ' + str (rep_position[0]) + ' ' + str (rep_position[1]) + ' ' + str (
                                    rep_position[2]))
                                file.write ('\n  ' + str (i_0 + 1) + ' ' + str (i_1 + 1) + ' ' + str (i_2 + 1))
                                
                                for alpha_0 in range (3):
                                    for alpha_1 in range (3):
                                        for alpha_2 in range (3):
                                            file.write (
                                                '\n  ' + str (alpha_0 + 1) + ' ' + str (alpha_1 + 1) + ' ' + str (
                                                    alpha_2 + 1) + "  %.11E" % three_particles_interaction[
                                                    alpha_0, alpha_1, alpha_2])
                                file.write ('\n')
        file.close ()
        with open (filename, 'r') as original:
            data = original.read ()
        with open (filename, 'w') as modified:
            modified.write ('  ' + str (block_counter) + '\n' + data)
        print('third order saved')
    
    def save_third_order_matrix(self, filename='FORCE_CONSTANTS_3RD'):
        shenbte_folder = self.folder
        anharmonic_system = self.system
        filename = shenbte_folder + filename
        file = open ('%s' % filename, 'a+')
        
        block_counter = 0
        for j_0, j_1, j_2 in (self.system._third_order_indices[:]):
            i_0, n_0 = anharmonic_system.atom_and_replica_index(j_0)
            i_1, n_1 = anharmonic_system.atom_and_replica_index(j_1)
            i_2, n_2 = anharmonic_system.atom_and_replica_index(j_2)
            if n_0 == 0:
                three_particles_interaction = anharmonic_system.third_order_with_indices (j_0, j_1, j_2)
                block_counter += 1
    
                file.write ('\n  ' + str (block_counter))
                rep_position = ath.apply_boundary(anharmonic_system.replicated_configuration, anharmonic_system.list_of_replicas[n_1])
                file.write (
                    '\n  ' + str (rep_position[0]) + ' ' + str (rep_position[1]) + ' ' + str (
                        rep_position[2]))
                rep_position = ath.apply_boundary(anharmonic_system.replicated_configuration, anharmonic_system.list_of_replicas[n_2])
                file.write (
                    '\n  ' + str (rep_position[0]) + ' ' + str (rep_position[1]) + ' ' + str (
                        rep_position[2]))
                file.write ('\n  ' + str (i_0 + 1) + ' ' + str (i_1 + 1) + ' ' + str (i_2 + 1))
                
                for alpha_0 in range (3):
                    for alpha_1 in range (3):
                        for alpha_2 in range (3):
                            file.write (
                                '\n  ' + str (alpha_0 + 1) + ' ' + str (alpha_1 + 1) + ' ' + str (
                                    alpha_2 + 1) + "  %.11E" %
                                three_particles_interaction[alpha_0, alpha_1, alpha_2])
                file.write ('\n')
        file.close ()
        with open (filename, 'r') as original:
            data = original.read ()
        with open (filename, 'w') as modified:
            modified.write ('  ' + str (block_counter) + '\n' + data)
    
    def run_sheng_bte(self, n_processors=1):
        if n_processors == 1:
            cmd = 'ShengBTE'
        else:
            cmd = 'mpirun -np ' + str(n_processors) + ' ShengBTE'
        return run_script (cmd, self.folder)
    
    
    def create_control_file_string(self):
        k_points = self.k_size
        system = self.system
        elements = self.system.configuration.get_chemical_symbols ()
        unique_elements = np.unique (self.system.configuration.get_chemical_symbols ())
        string = ''
        string += '&allocations\n'
        string += '\tnelements=' + str(len(unique_elements)) + '\n'
        string += '\tnatoms=' + str(len(elements)) + '\n'
        string += '\tngrid(:)=' + str (k_points[0]) + ' ' + str (k_points[1]) + ' ' + str (k_points[2]) + '\n'
        string += '&end\n'
        string += '&crystal\n'
        string += '\tlfactor=0.1,\n'
        for i in range (system.configuration.cell.shape[0]):
            vector = system.configuration.cell[i]
            string += '\tlattvec(:,' + str (i + 1) + ')= ' + str (vector[0]) + ' ' + str (vector[1]) + ' ' + str (
                vector[2]) + '\n'
        string += '\telements= '
        for element in np.unique(system.configuration.get_chemical_symbols()):
            string += '\"' + element + '\",'
        string +='\n'
        string += '\ttypes='
        for element in system.configuration.get_chemical_symbols():
            string += str(ath.type_element_id(system.configuration, element) + 1) + ' '
        string += ',\n'
        for i in range (system.configuration.positions.shape[0]):
            # TODO: double check this for more complicated geometries
            cellinv = np.linalg.inv (system.configuration.cell)
            vector = cellinv.dot(system.configuration.positions[i])
            string += '\tpositions(:,' + str (i + 1) + ')= ' + str (vector[0]) + ' ' + str (vector[1]) + ' ' + str (
                vector[2]) + '\n'
        string += '\tscell(:)=' + str (system.replicas[0]) + ' ' + str (system.replicas[1]) + ' ' + str (
            system.replicas[2]) + '\n'
        # if (self.length).any():
        # 	string += '\tlength(:)=' + str(self.length[0]) + ' ' + str(self.length[1]) + ' ' + str(self.length[2]) + '\n'
        string += '&end\n'
        string += '&parameters\n'
        string += '\tT=' + str (system.temperature) + '\n'
        string += '\tscalebroad=1.0\n'
        string += '&end\n'
        string += '&flags\n'
        string += '\tespresso=.true.\n'
        
        if self.only_gamma:
            string += '\tonly_gamma=.true.\n'
            
        if self.is_classical:
            string += '\tclassical=.true.\n'
        
        if self.convergence:
            string += '\tconvergence=.true.\n'
        else:
            string += '\tconvergence=.false.\n'
        
            
        string += '\tnonanalytic=.false.\n'
        string += '\tisotopes=.false.\n'
        string += '&end\n'
        return string
    
    def create_control_file(self):
        folder = self.folder
        filename = folder + '/CONTROL'
        string = self.create_control_file_string ()
        
        with open (filename, 'w') as file:
            file.write (string)
    
        
    def read_qpoints_mapper(self):
        q_points = pd.read_csv (self.folder + 'BTE.qpoints_full', header=None, delim_whitespace=True)
        self.qpoints_mapper = q_points.values
    
    def irreducible_indices(self):
        return np.unique(self.qpoints_mapper[:,1])
    
    def q_points(self):
        return self.qpoints_mapper[:,2:5]

    def q_points_per_irreducible_index(self, reduced_index):
        indices_per_q = np.where(self.qpoints_mapper[:,1]==reduced_index)
        equivalent_q_points = self.qpoints_mapper[indices_per_q]
        return np.delete(equivalent_q_points, 1, 1)
    
    def read_energy_data(self):
        # We read in rad/ps
        omega = pd.read_csv (self.folder + 'BTE.omega', header=None, delim_whitespace=True)
        n_qpoints = self.qpoints_mapper.shape[0]
        n_branches = omega.shape[1]
        energy_data = np.zeros ((n_qpoints, n_branches))
        for index, reduced_index, q_point_x, q_point_y, q_point_z in self.qpoints_mapper:
            energy_data[int (index - 1)] = omega.loc[[int (reduced_index - 1)]].values
        self.energies = energy_data

    def read_ps_data(self, type=None):
        if type == 'plus':
            file = 'BTE.WP3_plus'
        elif type == 'minus':
            file = 'BTE.WP3_minus'
        else:
            file = 'BTE.WP3'
        temperature = str (int (self.system.temperature))
        decay = pd.read_csv (self.folder + 'T' + temperature + 'K/' + file, header=None, delim_whitespace=True)
        # decay = pd.read_csv (self.folder + 'T' + temperature + 'K/BTE.w_anharmonic', header=None, delim_whitespace=True)
        n_branches = int (decay.shape[0] / self.irreducible_indices ().max ())
        n_qpoints_reduced = int (decay.shape[0] / n_branches)
        n_qpoints = self.qpoints_mapper.shape[0]
        decay = np.delete (decay.values, 0, 1)
        decay = decay.reshape ((n_branches, n_qpoints_reduced))
        decay_data = np.zeros ((n_qpoints, n_branches))
        for index, reduced_index, q_point_x, q_point_y, q_point_z in self.qpoints_mapper:
            decay_data[int (index - 1)] = decay[:, int (reduced_index - 1)]
        return decay_data
    
    def read_decay_rate_data(self, type=None):
        if type == 'plus':
            file = 'BTE.w_anharmonic_plus'
        elif type == 'minus':
            file = 'BTE.w_anharmonic_minus'
        else:
            file = 'BTE.w_anharmonic'
        temperature = str(int(self.system.temperature))
        decay = pd.read_csv (self.folder + 'T' + temperature + 'K/' + file, header=None, delim_whitespace=True)
        # decay = pd.read_csv (self.folder + 'T' + temperature + 'K/BTE.w_anharmonic', header=None, delim_whitespace=True)
        n_branches = int (decay.shape[0] / self.irreducible_indices ().max ())
        n_qpoints_reduced = int (decay.shape[0] / n_branches)
        n_qpoints = self.qpoints_mapper.shape[0]
        decay = np.delete(decay.values,0,1)
        decay = decay.reshape((n_branches, n_qpoints_reduced))
        decay_data = np.zeros ((n_qpoints, n_branches))
        for index, reduced_index, q_point_x, q_point_y, q_point_z in self.qpoints_mapper:
            decay_data[int (index - 1)] = decay[:,int(reduced_index-1)]
        return decay_data
    
    def read_velocity_data(self):
        shenbte_folder = self.folder
        velocities = pd.read_csv (shenbte_folder + 'BTE.v_full', header=None, delim_whitespace=True)
        n_velocities = velocities.shape[0]
        n_qpoints = self.qpoints_mapper.shape[0]
        n_modes = int(n_velocities / n_qpoints)
        
        velocity_array = velocities.values.reshape (n_modes, n_qpoints, 3)

        self.velocities =  np.zeros((self.k_size[0], self.k_size[1], self.k_size[2], n_modes, 3))

        z = 0
        for k in range (self.k_size[2]):
            for j in range(self.k_size[1]):
                for i in range (self.k_size[0]):
                    self.velocities[i,j,k,:,:] = velocity_array[:, z, :]
                    z += 1
        
   
    def calculate_broadening(self, velocity, n_grid):
        # TODO: I don't know why there's a 10 here, copied by sheng bte
        cellinv = np.linalg.inv (self.system.cell)
        rlattvec = cellinv * 2 * np.pi * 10.
        base_sigma = 0.
        for nu in range(3):
            base_sigma += (np.dot(rlattvec[nu, :], velocity)/n_grid[nu]) ** 2
        base_sigma = np.sqrt(base_sigma / 6.)
        # sigma = scalebroad *
        return base_sigma
    
    def header(self):
        system = self.system
        
        # this convert masses to qm masses
        mass_factor = 1.8218779 * 6.022e-4
    
        nat = len(self.system.configuration.get_chemical_symbols ())
        
        # TODO: The dielectric calculation is not implemented yet
        dielectric_constant = 1.
        born_eff_charge = 0.000000
        
        ntype = len(np.unique(self.system.configuration.get_chemical_symbols ()))

        
        # in quantum espresso ibrav = 0, do not use symmetry and use cartesian vectors to specify symmetries
        ibrav = 0
        header_str = ''
        header_str += str (ntype) + ' '
        header_str += str (nat) + ' '
        header_str += str (ibrav) + ' '
        
        # TODO: I'd like to have ibrav = 1 and put the actual positions here
        header_str += '0.0000000 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000 \n'
        header_str += matrix_to_string (system.configuration.cell)

        for i in range (ntype):
            mass = np.unique(self.system.replicated_configuration.get_masses())[i] / mass_factor
            label = np.unique(self.system.replicated_configuration.get_chemical_symbols())[i]
            header_str += str (i + 1) + ' \'' + label + '\' ' + str (mass) + '\n'
        
        # TODO: this needs to be changed, it works only if all the atoms in the unit cell are different species
        for i in range (nat):
            header_str += str (i + 1) + '  ' + str (i + 1) + '  ' + matrix_to_string (system.configuration.positions[i])
        header_str += 'T \n'
        header_str += matrix_to_string (np.diag (np.ones (3)) * dielectric_constant)
        for i in range (nat):
            header_str += str (i + 1) + '\n'
            header_str += matrix_to_string (np.diag (np.ones (3)) * born_eff_charge * (-1) ** i)
        header_str += str (system.replicas[0]) + ' '
        header_str += str (system.replicas[1]) + ' '
        header_str += str (system.replicas[2]) + '\n'
        return header_str
    
    def save_data(self):
        omega = self.energies
        lifetime = 1. / self.decay_rates
        n_modes = omega.shape[1]
        if self.is_classical:
            filename = "data_classic"
        else:
            filename = "data_quantum"
        filename = filename + '_' + str(self.system.temperature)
        filename = filename + ".csv"

        filename = self.folder + filename
        print('saving ' + filename)
        with open (filename, "w") as csv:
            str_to_write = 'k_x,k_y,k_z,'
            for i in range (n_modes):
                str_to_write += 'omega_' + str (i) + ' (rad/ps),'
            for i in range (n_modes):
                str_to_write += 'tau_' + str (i) + ' (ps),'
            for alpha in range (3):
                coord = 'x'
                if alpha == 1:
                    coord = 'y'
                if alpha == 2:
                    coord = 'z'
                    
                for i in range (n_modes):
                    str_to_write += 'v^' + coord + '_' + str (i) + ' (km/s),'
            str_to_write += '\n'
            csv.write (str_to_write)
            for k in range (self.q_points ().shape[0]):
                str_to_write = str (self.q_points ()[k, 0]) + ',' + str (self.q_points ()[k, 1]) + ',' + str (
                    self.q_points ()[k, 2]) + ','
                for i in range (n_modes):
                    str_to_write += str (self.energies[k, i]) + ','
                for i in range (n_modes):
                    str_to_write += str (lifetime[k, i]) + ','
                
                for alpha in range (3):
                    for i in range (n_modes):
                        str_to_write += str (self.velocities[k, i, alpha]) + ','
                str_to_write += '\n'
                csv.write (str_to_write)
        
    def decay_plot(self):
        omega = self.energies
        lifetime = 1. / self.decay_rates

        fig = plt.figure ()
        ax = plt.gca ()
        # full_data = np.array((omega.flatten(), lifetime.flatten()))
        # np.savetxt ('energy_vs_lifetime.csv', full_data.T, delimiter=',')
        
        
        
        ax.scatter (omega, lifetime)
        plt.grid ()
        plt.xlabel ("frequency (Thz)")
        plt.ylabel ("lifetime (ps)")
        # ax.set_xlim(10-2,omega.max()*1.05)
        # ax.set_ylim(10-2,(lifetime[1:,:]).max()*1.05)
        
        ax.set_xscale ('log')
        ax.set_yscale ('log')
        
        plt.show ()
    
    def read_conductivity(self, converged=True, is_classical=False):
        folder = self.folder
        if converged:
            conduct_file = 'BTE.KappaTensorVsT_CONV'
        else:
            conduct_file = 'BTE.KappaTensorVsT_RTA'
        
        conductivity_array = np.loadtxt (folder + conduct_file)
        conductivity_array = np.delete (conductivity_array, 0)
        n_steps = 0
        if converged:
            n_steps = int (conductivity_array[-1])
            conductivity_array = np.delete (conductivity_array, -1)
            
        filename = folder + 'conductivity_'
        if is_classical:
            filename = filename + 'classical_'
        else:
            filename = filename + 'quantum_'
        filename = filename + str (self.system.temperature)
        
        filename = filename + '.csv'
        conductivity = conductivity_array.reshape (3, 3)
        np.savetxt(filename , conductivity, delimiter=',')
        print('saving ' + filename)
        return conductivity
    
    def energy_easy_plot(self, nanowire=False):
        if not nanowire:
            q_points_filtered, modes_filtered = ghl.filter_modes_and_k_points (self.q_points(), self.energies)
        else:
            q_points_filtered = self.q_points()[:,2]
            modes_filtered = self.energies
        plt.ylabel ("frequency (rad/ps)")
        plt.xlabel ("wavevector")
        plt.ylim (0, self.energies.max () * 1.05)

        plt.grid ('on')
        plt.xlim(0,.5)
        # modes so far are in rad/ps, here we change them to rad/ps
        plt.plot (q_points_filtered, modes_filtered, "-", color='black')
    
    
    def __str__(self):
        # string = 'S_'
        # string += str(self.ph_system)
        # string += '_k' + str (self.k_mesh).replace (" ", "").replace ('[', "").replace (']', "")
        # if (self.length.any() != 0):
        # 	string += '_l'
        # 	for length in self.length:
        # 		if length < THREESHOLD:
        # 			string += str(length)
        string = str(self.system)
        # if self.is_classical:
        #     string += '_c'
        # else:
        #     string += '_q'
        # return string
        return string
    
