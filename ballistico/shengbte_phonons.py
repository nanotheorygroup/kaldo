from ballistico.logger import Logger
from ballistico.phonons import Phonons
import ballistico.geometry_helper as ghl
from ballistico.constants import *
from ballistico.tools import *
import matplotlib.pyplot as plt
import ballistico.atoms_helper as ath

BUFFER_PLOT = .2


class Shengbte_phonons (Phonons):
    def __init__(self,  atoms, supercell=(1, 1, 1), kpts=(1, 1, 1), is_classic=False, temperature=300, second_order=None, third_order=None, is_persistency_enabled=True, parameters={}):
        super(self.__class__, self).__init__(atoms=atoms, folder_name=type(self).__name__, supercell=supercell, kpts=kpts, is_classic=is_classic, temperature=temperature, is_persistency_enabled=is_persistency_enabled)

        self.second_order = second_order
        self.third_order = third_order
        self._qpoints_mapper = None
        self._energies = None

        

        
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
            
        self.run()

        
    @property
    def qpoints_mapper(self):
        return self._qpoints_mapper
    
    @qpoints_mapper.getter
    def qpoints_mapper(self):
        if self._qpoints_mapper is None:
            self.read_qpoints_mapper ()
        return self._qpoints_mapper
    
    @property
    def energies(self):
        return self._energies
    
    @energies.getter
    def energies(self):
        if self._energies is None:
            self._energies = self.read_energy_data ()
        return self._energies

    @property
    def frequencies(self):
        return super ().frequencies

    @frequencies.getter
    def frequencies(self):
        if super (self.__class__, self).frequencies is not None:
            return super (self.__class__, self).frequencies
        frequencies = self.read_energy_data () / (2 * np.pi)
        self.frequencies = frequencies
        return frequencies

    @frequencies.setter
    def frequencies(self, new_frequencies):
        Phonons.frequencies.fset (self, new_frequencies)

    @property
    def velocities(self):
        return super ().velocities

    @velocities.getter
    def velocities(self):
        if super (self.__class__, self).velocities is not None:
            return super (self.__class__, self).velocities
        velocities = self.read_velocity_data ()
        self.velocities = velocities
        return velocities

    @velocities.setter
    def velocities(self, new_velocities):
        Phonons.velocities.fset (self, new_velocities)

    @property
    def gamma(self):
        return super ().gamma

    @gamma.getter
    def gamma(self):
        if super (self.__class__, self).gamma is not None:
            return super (self.__class__, self).gamma
        gamma = self.read_decay_rate_data ()
        self.gamma = gamma
        return gamma

    @gamma.setter
    def gamma(self, new_gamma):
        Phonons.gamma.fset (self, new_gamma)
        
        
    def save_second_order_matrix(self):
        second_order = self.second_order
        shenbte_folder = self.folder_name + '/'
        filename = 'espresso.ifc2'
        n_particles = second_order.shape[1]
        filename = shenbte_folder + filename
        file = open ('%s' % filename, 'a+')
        cell_inv = np.linalg.inv(self.atoms.cell)
        list_of_indices = np.zeros_like(self.list_of_replicas, dtype=np.int)
        for replica_id in range(self.list_of_replicas.shape[0]):
            list_of_indices[replica_id] = cell_inv.dot(self.list_of_replicas[replica_id])
        file.write (self.header ())
        for alpha in range (3):
            for beta in range (3):
                for i in range (n_particles):
                    for j in range (n_particles):
                        file.write ('\t' + str (alpha + 1) + '\t' + str (beta + 1) + '\t' + str (i + 1) + '\t' + str (j + 1) + '\n')
                        for id_replica in range(list_of_indices.shape[0]):
                            index = list_of_indices[id_replica]
                            l_vec = np.array(index % self.supercell + 1).astype(np.int)
                            file.write ('\t' + str (l_vec[0]) + '\t' + str (l_vec[1]) + '\t' + str (l_vec[2]))
                            
                            # TODO: WHy are they flipped?
                            matrix_element = self.second_order[0, j, beta, id_replica, i, alpha]
                            
                            matrix_element = matrix_element / evoverdlpoly / rydbergoverev * (bohroverangstrom ** 2)
                            file.write ('\t %.11E' % matrix_element)
                            file.write ('\n')
        file.close ()
        Logger().info ('second order saved')
    
    
    def save_third_order_matrix(self):
        filename = 'FORCE_CONSTANTS_3RD'
        filename = self.folder_name + '/' + filename
        file = open ('%s' % filename, 'w')
        n_in_unit_cell = len (self.atoms.numbers)
        n_replicas = np.prod (self.supercell)
        block_counter = 0
        for i_0 in range (n_in_unit_cell):
            for n_1 in range (n_replicas):
                for i_1 in range (n_in_unit_cell):
                    for n_2 in range (n_replicas):
                        for i_2 in range (n_in_unit_cell):
                            
                            three_particles_interaction = self.third_order[0, i_0, :, n_1, i_1, :, n_2, i_2, :]
                            
                            if (np.abs (three_particles_interaction) > 1e-9).any ():
                                block_counter += 1
                                replica = self.list_of_replicas#
                                file.write ('\n  ' + str (block_counter))
                                rep_position = ath.apply_boundary (self.replicated_atoms,replica[n_1])
                                file.write ('\n  ' + str (rep_position[0]) + ' ' + str (rep_position[1]) + ' ' + str (
                                    rep_position[2]))
                                rep_position = ath.apply_boundary (self.replicated_atoms,replica[n_2])
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
        Logger().info ('third order saved')
    
    def run(self, n_processors=1):
        self.create_control_file()
        self.save_second_order_matrix ()
        self.save_third_order_matrix ()
        if n_processors == 1:
            cmd = 'ShengBTE'
        else:
            cmd = 'mpirun -np ' + str(n_processors) + ' ShengBTE'
        return run_script (cmd, self.folder_name)
    
    
    def create_control_file_string(self):
        k_points = self.kpts
        elements = self.atoms.get_chemical_symbols ()
        unique_elements = np.unique (self.atoms.get_chemical_symbols ())
        string = ''
        string += '&allocations\n'
        string += '\tnelements=' + str(len(unique_elements)) + '\n'
        string += '\tnatoms=' + str(len(elements)) + '\n'
        string += '\tngrid(:)=' + str (k_points[0]) + ' ' + str (k_points[1]) + ' ' + str (k_points[2]) + '\n'
        string += '&end\n'
        string += '&crystal\n'
        string += '\tlfactor=0.1,\n'
        for i in range (self.atoms.cell.shape[0]):
            vector = self.atoms.cell[i]
            string += '\tlattvec(:,' + str (i + 1) + ')= ' + str (vector[0]) + ' ' + str (vector[1]) + ' ' + str (
                vector[2]) + '\n'
        string += '\telements= '
        for element in np.unique(self.atoms.get_chemical_symbols()):
            string += '\"' + element + '\",'
        string +='\n'
        string += '\ttypes='
        for element in self.atoms.get_chemical_symbols():
            string += str(ath.type_element_id(self.atoms, element) + 1) + ' '
        string += ',\n'
        for i in range (self.atoms.positions.shape[0]):
            # TODO: double check this for more complicated geometries
            cellinv = np.linalg.inv (self.atoms.cell)
            vector = cellinv.dot(self.atoms.positions[i])
            string += '\tpositions(:,' + str (i + 1) + ')= ' + str (vector[0]) + ' ' + str (vector[1]) + ' ' + str (
                vector[2]) + '\n'
        string += '\tscell(:)=' + str (self.supercell[0]) + ' ' + str (self.supercell[1]) + ' ' + str (
            self.supercell[2]) + '\n'
        # if (self.length).any():
        # 	string += '\tlength(:)=' + str(self.length[0]) + ' ' + str(self.length[1]) + ' ' + str(self.length[2]) + '\n'
        string += '&end\n'
        string += '&parameters\n'
        string += '\tT=' + str (self.temperature) + '\n'
        string += '\tscalebroad=1.0\n'
        string += '&end\n'
        string += '&flags\n'
        string += '\tespresso=.true.\n'
        
        if self.only_gamma:
            string += '\tonly_gamma=.true.\n'
            
        if self.is_classic:
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
        folder = self.folder_name
        filename = folder + '/CONTROL'
        string = self.create_control_file_string ()
        
        with open (filename, 'w') as file:
            file.write (string)

    def header(self):
    
        # this convert masses to qm masses
        mass_factor = 1.8218779 * 6.022e-4
    
        nat = len (self.atoms.get_chemical_symbols ())
    
        # TODO: The dielectric calculation is not implemented yet
        dielectric_constant = 1.
        born_eff_charge = 0.000000
    
        ntype = len (np.unique (self.atoms.get_chemical_symbols ()))
        # in quantum espresso ibrav = 0, do not use symmetry and use cartesian vectors to specify symmetries
        ibrav = 0
        header_str = ''
        header_str += str (ntype) + ' '
        header_str += str (nat) + ' '
        header_str += str (ibrav) + ' '
    
        # TODO: I'd like to have ibrav = 1 and put the actual positions here
        header_str += '0.0000000 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000 \n'
        header_str += matrix_to_string (self.atoms.cell)
    
        for i in range (ntype):
            mass = np.unique (self.replicated_atoms.get_masses ())[i] / mass_factor
            label = np.unique (self.replicated_atoms.get_chemical_symbols ())[i]
            header_str += str (i + 1) + ' \'' + label + '\' ' + str (mass) + '\n'
    
        # TODO: this needs to be changed, it works only if all the atoms in the unit cell are different species
        for i in range (nat):
            header_str += str (i + 1) + '  ' + str (i + 1) + '  ' + matrix_to_string (self.atoms.positions[i])
        header_str += 'T \n'
        header_str += matrix_to_string (np.diag (np.ones (3)) * dielectric_constant)
        for i in range (nat):
            header_str += str (i + 1) + '\n'
            header_str += matrix_to_string (np.diag (np.ones (3)) * born_eff_charge * (-1) ** i)
        header_str += str (self.supercell[0]) + ' '
        header_str += str (self.supercell[1]) + ' '
        header_str += str (self.supercell[2]) + '\n'
        return header_str

        
    def read_qpoints_mapper(self):
        q_points = pd.read_csv (self.folder_name + '/BTE.qpoints_full', header=None, delim_whitespace=True)
        self._qpoints_mapper = q_points.values
    
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
        omega = pd.read_csv (self.folder_name + '/BTE.omega', header=None, delim_whitespace=True)
        n_qpoints = self.qpoints_mapper.shape[0]
        n_branches = omega.shape[1]
        energy_data = np.zeros ((n_qpoints, n_branches))
        for index, reduced_index, q_point_x, q_point_y, q_point_z in self.qpoints_mapper:
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
        decay = pd.read_csv (self.folder_name + '/T' + temperature + 'K/' + file, header=None, delim_whitespace=True)
        # decay = pd.read_csv (self.folder_name + 'T' + temperature + 'K/BTE.w_anharmonic', header=None, delim_whitespace=True)
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
        temperature = str(int(self.temperature))
        decay = pd.read_csv (self.folder_name + '/T' + temperature + 'K/' + file, header=None, delim_whitespace=True)
        # decay = pd.read_csv (self.folder_name + 'T' + temperature + 'K/BTE.w_anharmonic', header=None, delim_whitespace=True)
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
        shenbte_folder = self.folder_name
        velocities = pd.read_csv (shenbte_folder + '/BTE.v_full', header=None, delim_whitespace=True)
        n_velocities = velocities.shape[0]
        n_qpoints = self.qpoints_mapper.shape[0]
        n_modes = int(n_velocities / n_qpoints)
        
        velocity_array = velocities.values.reshape (n_modes, n_qpoints, 3)

        velocities =  np.zeros((self.kpts[0], self.kpts[1], self.kpts[2], n_modes, 3))

        z = 0
        for k in range (self.kpts[2]):
            for j in range(self.kpts[1]):
                for i in range (self.kpts[0]):
                    velocities[i,j,k,:,:] = velocity_array[:, z, :]
                    z += 1
        return velocities
        
   
        
    def read_conductivity(self, converged=True, is_classical=False):
        folder = self.folder_name
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
            
        filename = folder + 'conductivity_'
        if is_classical:
            filename = filename + 'classical_'
        else:
            filename = filename + 'quantum_'
        filename = filename + str (self.temperature)
        
        filename = filename + '.csv'
        conductivity = conductivity_array.reshape (3, 3)
        np.savetxt(filename , conductivity, delimiter=',')
        Logger().info ('saving ' + filename)
        return conductivity
    