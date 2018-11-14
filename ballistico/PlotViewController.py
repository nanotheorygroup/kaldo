import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import ballistico.geometry_helper as ghl
import datetime
import numpy as np
import time
from ballistico.interpolation_controller import interpolator


BUFFER_PLOT = .2


class PlotViewController (object):
	def __init__(self, phonons):
		self.phonons = phonons
		self.system = phonons.atoms

	def project_to_path(self, observable, symmetry, n_k_points=100, with_fourier=True):

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
	
	def plot_in_brillouin_zone(self, observable, symmetry,  n_k_points=100, with_fourier=True):
		phonons = self.phonons
		fig = plt.figure ()
		q, Q, point_names, freqs_plot = self.project_to_path(observable, symmetry, n_k_points, with_fourier=with_fourier)
		plt.ylabel ("frequency ($\mathrm{Thz}$)")
		plt.xticks (Q, point_names)
		plt.xlim (q[0], q[-1])
		plt.plot (q, freqs_plot, "-")
		plt.grid ()
		plt.ylim (freqs_plot.min (), freqs_plot.max () * 1.05)
		plt.show ()
	
	def plot_dos(self, omega_e, dos_e):
		fig = plt.figure ()
		plt.plot (dos_e, omega_e, "-", color='black')
		plt.ylim (omega_e.min (), omega_e.max () * 1.05)
		plt.grid ()
		plt.xlim (0, dos_e.max () * (1. + BUFFER_PLOT))
		plt.xlabel ("DOS")
		plt.show ()


	def plot_c_v(self):
		phonons = self.phonons
		# Plot c_v
		fig = plt.figure ()
		plt.scatter (phonons.frequencies[phonons.frequencies != 0].flatten (),
		             phonons.c_v[phonons.frequencies != 0].flatten ())
		plt.ylabel ("$c_V$", fontsize=16, fontweight='bold')
		plt.xlabel ("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
		# fig.savefig ('c_v.pdf')
		plt.show ()

	def plot_velocities(self):
		phonons = self.phonons
		# Plot velocity
		fig = plt.figure ()
		plt.scatter (phonons.frequencies.flatten (), np.linalg.norm (phonons.velocities, axis=-1).flatten ())
		plt.ylabel ("$v_{rms}$ (10m/s)", fontsize=16, fontweight='bold')
		plt.xlabel ("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
		# fig.savefig ('velocities.pdf')
		plt.show ()

	def plot_gamma(self):
		phonons = self.phonons
		# Plot gamma
		fig = plt.figure ()
		plt.scatter (phonons.frequencies.flatten (), phonons.gamma.flatten ())
		plt.ylabel ("$\gamma$ (Thz)", fontsize=16, fontweight='bold')
		plt.xlabel ("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
		# fig.savefig ('gamma.pdf')
		plt.show ()
