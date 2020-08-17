from kaldo.observables.harmonic_with_q import HarmonicWithQ
import numpy as np
import ase.units as units
from kaldo.helpers.storage import lazy_property


KELVINTOJOULE = units.kB / units.J
KELVINTOTHZ = units.kB / units.J / (2 * np.pi * units._hbar) * 1e-12


class HarmonicWithQTemp(HarmonicWithQ):

    def __init__(self, temperature, is_classic, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.temperature = temperature
        self.is_classic = is_classic


    @lazy_property(label='<q_point>')
    def population(self):
        population = self._calculate_population()
        return population


    @lazy_property(label='<q_point>')
    def heat_capacity(self):
        heat_capacity = self._calculate_heat_capacity()
        return heat_capacity


    @lazy_property(label='<q_point>')
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
        if (self.is_classic):
            c_v = np.zeros((self.n_modes, self.n_modes))
            c_v[:, :] = KELVINTOJOULE
        else:
            frequencies = self.frequency.flatten()
            temperature = self.temperature * KELVINTOTHZ
            heat_capacity = self.heat_capacity.flatten()
            physical_mode = self.physical_mode.flatten()
            f_be = self.population.flatten()
            c_v_omega = (f_be[:, np.newaxis] - f_be[np.newaxis, :])
            diff_omega = (frequencies[:, np.newaxis] - frequencies[np.newaxis, :])
            mask_degeneracy = np.where(diff_omega == 0, True, False)

            # value to do the division
            diff_omega[mask_degeneracy] = 1
            divide_omega = -1 / diff_omega
            freq_sq = frequencies[:, np.newaxis] * frequencies[np.newaxis, :]

            # remember here f_n-f_m/ w_m-w_n index reversed
            c_v = freq_sq * c_v_omega * divide_omega
            c_v = KELVINTOJOULE * c_v / temperature

            #Degeneracy part: let us substitute the wrong elements
            heat_capacity_deg_2d = (heat_capacity[:, np.newaxis]
                                    + heat_capacity[np.newaxis, :]) / 2
            c_v = np.where(mask_degeneracy, heat_capacity_deg_2d, c_v)
            #Physical modes
            c_v = c_v * physical_mode[:, np.newaxis] * physical_mode[np.newaxis, :]
        return c_v


    def _calculate_population(self):
        frequency = self.frequency
        temp = self.temperature * KELVINTOTHZ
        density = np.zeros_like(frequency)
        physical_mode = self.physical_mode.reshape(frequency.shape)
        if self.is_classic is False:
            density[physical_mode] = 1. / (np.exp(frequency[physical_mode] / temp) - 1.)
        else:
            density[physical_mode] = temp / frequency[physical_mode]
        return density


    def _calculate_heat_capacity(self):
        frequency = self.frequency
        c_v = np.zeros_like(frequency)
        temperature = self.temperature * KELVINTOTHZ
        physical_mode = self.physical_mode.reshape(frequency.shape)
        if (self.is_classic):
            c_v[physical_mode] = KELVINTOJOULE
        else:
            f_be = self.population
            c_v[physical_mode] = KELVINTOJOULE * f_be[physical_mode] * (f_be[physical_mode] + 1) * self.frequency[
                physical_mode] ** 2 / (temperature ** 2)
        return c_v
