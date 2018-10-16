
import ballistico.atoms_helper as ath
import ase.io as io
import numpy as np
import os


class MolecularSystem (object):
    def __init__(self, configuration, temperature, replicas=(1, 1, 1)):
        self.replicas = np.array(replicas)
        self.temperature = temperature
        self.configuration = configuration
        self.configuration.cell_inv = np.linalg.inv (self.configuration.cell)

        self.folder = str (self) + '/'
        if not os.path.exists (self.folder):
            os.makedirs (self.folder)
        
        
    
    def atom_and_replica_index(self, absolute_index):
        n_replicas = np.prod(self.replicas)

        id_replica = absolute_index % n_replicas
        id_atom = absolute_index / n_replicas
        
        return int(id_atom), int(id_replica)
    

    def __str__(self):
        atoms = self.configuration
        string = ''
        unique_elements = np.unique (atoms.get_chemical_symbols ())
        for element in unique_elements:
            string += element
        volume = np.linalg.det (atoms.cell) / 1000.
        string += '_a' + str (int (volume * 1000.))
        string += '_r' + str (self.replicas[0]) + str (self.replicas[1]) + str (self.replicas[2])
        string += '_T' + str (int (self.temperature))
        return string
