import kaldo.observables.harmonic_with_q as hwq
import numpy as np
import ase.units as units
from kaldo.controllers.qhgk import generalized_heat_capacity
from kaldo.helpers.units import KB_J_PER_K
from kaldo.storable import lazy_property, Storable


# Classical statistics are implemented as a numerical hbar -> 0 limit: the
# temperature-dependent observables evaluate the same quantum formulas with
# hbar scaled by this factor, which pushes every occupation argument
# x = hbar*omega/(k_B T) deep into the classical regime, where the quantum
# expressions reduce to their classical limits with O(x^2) ~ 1e-12 error.
# Consequences of the convention:
#   * heat_capacity reduces to the Dulong-Petit k_B exactly;
#   * population becomes the equipartition occupation k_B T/(hbar*omega),
#     carrying a 1/CLASSICAL_HBAR_SCALE factor and a -1/2 offset from the
#     Bose expansion 1/(e^x - 1) = 1/x - 1/2 + O(x). Both cancel where it
#     matters: the offsets cancel in the scattering population factors
#     (n' - n'') and 1/2(1 + n' + n''), and the 1/CLASSICAL_HBAR_SCALE
#     cancels against the matching CLASSICAL_HBAR_SCALE applied to the
#     scattering potential in
#     Phonons._select_algorithm_for_phase_space_and_gamma. The population
#     and phase_space arrays a user reads therefore carry this internal
#     convention factor; bandwidth, heat capacity, and conductivity do not.
CLASSICAL_HBAR_SCALE = 1e-6


class HarmonicWithQTemp(hwq.HarmonicWithQ):

    # Define storage formats for temperature-dependent harmonic properties
    # Inherits base _store_formats from HarmonicWithQ and adds temperature-specific ones
    _store_formats = {
        **hwq.HarmonicWithQ._store_formats,  # Inherit parent formats
        'population': 'formatted',
        'heat_capacity': 'formatted',
        'heat_capacity_2d': 'formatted'
    }

    def __init__(self, temperature, is_classic, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.temperature = temperature
        self.is_classic = is_classic
        self.hbar = units._hbar
        if self.is_classic:
            self.hbar = self.hbar * CLASSICAL_HBAR_SCALE

    def _get_folder_path_components(self, label):
        components = []
        if '<temperature>' in label:
            components.append(str(int(self.temperature)))
        if '<statistics>' in label:
            components.append('classic' if self.is_classic else 'quantum')
        return components

    @lazy_property(label='<temperature>/<statistics>/<q_point>')
    def population(self):
        population = self._calculate_population()
        return population


    @lazy_property(label='<temperature>/<statistics>/<q_point>')
    def heat_capacity(self):
        heat_capacity = self._calculate_heat_capacity()
        return heat_capacity


    @lazy_property(label='<temperature>/<statistics>/<q_point>')
    def heat_capacity_2d(self):
        heat_capacity_2d = self._calculate_2d_heat_capacity()
        return heat_capacity_2d


    def _calculate_2d_heat_capacity(self):
        """Calculates the factor for the diffusivity which resembles the heat capacity.
        The array is returned in units of J/K.
        classical case: k_b
        quantum case: c_nm=hbar w_n w_m/T  * (n_n-n_m)/(w_m-w_n)

        Returns
        -------
        c_v : np.array
            (phonons.n_k_points,phonons.modes, phonons.n_modes) float
        """
        omega = 2 * np.pi * self.frequency.flatten()
        physical_mode = self.physical_mode.flatten()
        c_v = generalized_heat_capacity(omega, omega, physical_mode, physical_mode,
                                        temperature=self.temperature, is_classic=self.is_classic)
        return KB_J_PER_K * c_v


    def _calculate_population(self):
        frequency = self.frequency
        kelvintothz = units.kB / units.J / (2 * np.pi * self.hbar) * 1e-12
        temp = self.temperature * kelvintothz
        population = np.zeros_like(frequency)
        physical_mode = self.physical_mode.reshape(frequency.shape)
        population[physical_mode] = 1. / (np.exp(frequency[physical_mode] / temp) - 1.)
        return population


    def _calculate_heat_capacity(self):
        frequency = self.frequency
        c_v = np.zeros_like(frequency)
        kelvintothz = units.kB / units.J / (2 * np.pi * self.hbar) * 1e-12
        temperature = self.temperature * kelvintothz
        physical_mode = self.physical_mode.reshape(frequency.shape)
        kelvintojoule = units.kB / units.J
        f_be = self.population
        c_v[physical_mode] = kelvintojoule * f_be[physical_mode] * (f_be[physical_mode] + 1) * self.frequency[
            physical_mode] ** 2 / (temperature ** 2)
        return c_v
