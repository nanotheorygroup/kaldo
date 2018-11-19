import ballistico.calculator
from ballistico.finite_difference import FiniteDifference
from ballistico.phonons_controller import PhononsController

class BallisticoPhonons (PhononsController):
    def __init__(self, finite_difference, kpts=(1, 1, 1), is_classic=False, temperature=300, sigma_in=None, is_persistency_enabled=True):
        super(self.__class__, self).__init__(finite_difference, kpts=kpts, is_classic=is_classic, temperature=temperature, is_persistency_enabled=is_persistency_enabled, sigma_in=sigma_in)
        self.second_order = finite_difference.second_order
        self.third_order = finite_difference.third_order

    @property
    def frequencies(self):
        return super().frequencies

    @frequencies.getter
    def frequencies(self):
        if super (self.__class__, self).frequencies is not None:
            return super (self.__class__, self).frequencies
        frequencies, eigenvalues, eigenvectors, velocities = ballistico.calculator.calculate_second_all_grid(
            self.k_points,
            self.atoms,
            self.second_order,
            self.finite_difference.list_of_index,
            self.finite_difference.replicated_atoms)
        self.frequencies = frequencies
        self.eigenvalues = eigenvalues
        self.velocities = velocities
        self.eigenvectors = eigenvectors
        return frequencies

    @frequencies.setter
    def frequencies(self, new_frequencies):
        PhononsController.frequencies.fset(self, new_frequencies)
        
    @property
    def velocities(self):
        return super().velocities
    
    @velocities.getter
    def velocities(self):
        if super (self.__class__, self).velocities is not None:
            return super (self.__class__, self).velocities
        frequencies, eigenvalues, eigenvectors, velocities = ballistico.calculator.calculate_second_all_grid(
            self.k_points,
            self.atoms,
            self.second_order,
            self.finite_difference.list_of_index,
            self.finite_difference.replicated_atoms)
        self.frequencies = frequencies
        self.eigenvalues = eigenvalues
        self.velocities = velocities
        self.eigenvectors = eigenvectors
        return velocities

    @velocities.setter
    def velocities(self, new_velocities):
        PhononsController.velocities.fset(self, new_velocities)

    @property
    def eigenvalues(self):
        return super().eigenvalues
    
    @eigenvalues.getter
    def eigenvalues(self):
        if super (self.__class__, self).eigenvalues is not None:
            return super (self.__class__, self).eigenvalues
        frequencies, eigenvalues, eigenvectors, velocities = ballistico.calculator.calculate_second_all_grid(
            self.k_points,
            self.atoms,
            self.second_order,
            self.finite_difference.list_of_index,
            self.finite_difference.replicated_atoms)
        self.frequencies = frequencies
        self.eigenvalues = eigenvalues
        self.velocities = velocities
        self.eigenvectors = eigenvectors
        return eigenvalues

    @eigenvalues.setter
    def eigenvalues(self, new_eigenvalues):
        PhononsController.eigenvalues.fset(self, new_eigenvalues)

    @property
    def eigenvectors(self):
        return super().eigenvectors
    
    @eigenvectors.setter
    def eigenvectors(self, new_eigenvectors):
        PhononsController.eigenvectors.fset(self, new_eigenvectors)
        
    @eigenvectors.getter
    def eigenvectors(self):
        if super (self.__class__, self).eigenvectors is not None:
            return super (self.__class__, self).eigenvectors
        frequencies, eigenvalues, eigenvectors, velocities = ballistico.calculator.calculate_second_all_grid(
            self.k_points,
            self.atoms,
            self.second_order,
            self.finite_difference.list_of_index,
            self.finite_difference.replicated_atoms)
        self.frequencies = frequencies
        self.eigenvalues = eigenvalues
        self.velocities = velocities
        self.eigenvectors = eigenvectors
        return eigenvectors

    @property
    def dos(self):
        return super().dos
    
    @dos.setter
    def dos(self, new_dos):
        PhononsController.dos.fset(self, new_dos)

    @dos.getter
    def dos(self):
        if super (self.__class__, self).dos is not None:
            return super (self.__class__, self).dos
        # TODO: delta needs to be set by the instance
        delta = 1
        num = 100
        dos = ballistico.calculator.calculate_density_of_states(
            self.frequencies,
            self.kpts,
            delta,
            num
        )
        self.dos = dos
        return dos

    @property
    def gamma(self):
        return super ().gamma

    @gamma.setter
    def gamma(self, new_gamma):
        PhononsController.gamma.fset(self, new_gamma)
        
    @gamma.getter
    def gamma(self):
        if super (self.__class__, self).gamma is not None:
            return super (self.__class__, self).gamma
        gamma, scattering_matrix = ballistico.calculator.calculate_gamma(
            self.atoms,
            self.frequencies,
            self.velocities,
            self.occupations,
            self.kpts,
            self.eigenvectors,
            self.finite_difference.list_of_index,
            self.third_order,
            self.sigma_in
        )
        self.gamma = gamma
        self.scattering_matrix = scattering_matrix
        return gamma

    @property
    def scattering_matrix(self):
        return super ().scattering_matrix

    @scattering_matrix.setter
    def scattering_matrix(self, new_scattering_matrix):
        PhononsController.scattering_matrix.fset (self, new_scattering_matrix)

    @scattering_matrix.getter
    def scattering_matrix(self):
        if super (self.__class__, self).scattering_matrix is not None:
            return super (self.__class__, self).scattering_matrix
        gamma, scattering_matrix = ballistico.calculator.calculate_gamma (
            self.atoms,
            self.frequencies,
            self.velocities,
            self.occupations,
            self.kpts,
            self.eigenvectors,
            self.finite_difference.list_of_index,
            self.third_order,
            self.sigma_in
        )
        self.scattering_matrix = scattering_matrix
        self.gamma = gamma
        return scattering_matrix



    