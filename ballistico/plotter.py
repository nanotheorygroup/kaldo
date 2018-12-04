import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import ballistico.geometry_helper as ghl
import datetime
import numpy as np
import time
from ballistico.interpolation_controller import interpolator
import os
import ballistico.constants as constants


BUFFER_PLOT = .2


class Plotter (object):
	def __init__(self, phonons, folder='plots', is_showing=True, is_persistency_enabled=True):
		self.phonons = phonons
		self.system = phonons.atoms
		self.folder = folder
		self.is_persistency_enabled = is_persistency_enabled
		self.is_showing = is_showing

		if self.is_persistency_enabled:
			if not os.path.exists (self.folder):
				os.makedirs (self.folder)

	def project_to_path(self, observable, symmetry='fcc', n_k_points=100, with_fourier=True):
		atoms = self.phonons.atoms
		k_size = self.phonons.kpts
		n_modes = atoms.positions.shape[0] * 3
		obs = observable.reshape ((k_size[0], k_size[1], k_size[2], n_modes))

		cell = atoms.cell
		k_list, q, Q, point_names = ghl.create_k_and_symmetry_space (cell, symmetry=symmetry, n_k_points=n_k_points)
		obs_plot = np.zeros ((k_list.shape[0], n_modes))

		for mode in range (n_modes):
			obs_plot[:, mode] = interpolator (k_list, obs[:, :, :, mode], with_fourier=with_fourier)
		return q, Q, point_names, obs_plot
	
	def plot_in_brillouin_zone(self, observable_name='frequency', observable=None, symmetry='fcc',  n_k_points=100, with_fourier=True):
		if observable is None:
			observable = self.phonons.frequencies
		fig = plt.figure ()
		q, Q, point_names, freqs_plot = self.project_to_path(observable, symmetry, n_k_points, with_fourier=with_fourier)
		plt.ylabel (observable_name)
		plt.xticks (Q, point_names)
		plt.xlim (q[0], q[-1])
		plt.plot (q, freqs_plot, "-")
		plt.grid ()
		plt.ylim (freqs_plot.min (), freqs_plot.max () * 1.05)
		if self.is_persistency_enabled:
			fig.savefig (self.folder + observable_name + '.pdf')
		if self.is_showing:
			plt.show ()
	
	def plot_vs_frequency(self, observable, observable_name):
		# TODO: We should check if the flattn is C-like and still compatible with Sheng 'F' like
		frequencies = self.phonons.frequencies.flatten ()
		observable = observable.flatten ()
		fig = plt.figure ()
		plt.scatter (frequencies[3:],
		             observable[3:])
		plt.ylabel (observable_name, fontsize=16, fontweight='bold')
		plt.xlabel ("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
		if self.is_persistency_enabled:
			fig.savefig (self.folder + observable_name + '.pdf')
		if self.is_showing:
			plt.show ()

	def plot_everything(self, with_dispersion=True):
		phonons = self.phonons
		if with_dispersion:
			self.plot_in_brillouin_zone (observable_name='disp_rel', with_fourier=False)
			# self.plot_in_brillouin_zone (observable_name='disp_rel_fourier', with_fourier=True)
		self.plot_vs_frequency (phonons.c_v, 'cv')
		vel = np.linalg.norm (phonons.velocities, axis=-1)
		self.plot_vs_frequency (vel, 'vel')
		self.plot_vs_frequency (phonons.gamma, 'gamma')
		self.plot_vs_frequency (phonons.gamma * constants.davide_coeff, 'gamma')