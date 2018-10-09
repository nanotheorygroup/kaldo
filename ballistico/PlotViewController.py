import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import ballistico.geometry_helper as ghl
import datetime
import time
BUFFER_PLOT = .2


class PlotViewController (object):
	def __init__(self, system):
		self.fig = plt.figure ()
		self.system = system

		
	def plot_in_brillouin_zone(self, freqs_plot, symmetry, title=None, n_k_points=100):
		plt.figure (1, (8, 6))
		freqs_plot = freqs_plot
		k_list, q, Q, point_names = ghl.create_k_and_symmetry_space (self.system, symmetry=symmetry, n_k_points=n_k_points)
		plt.axes ([.08, .07, .67, .85])
		# plt.ylabel ("frequency ($\mathrm{Thz}$)")
		plt.ylabel ("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
		plt.xticks (Q, point_names, fontsize=16, fontweight='bold')
		plt.xlim (q[0], q[-1])
		plt.plot (q, freqs_plot, "-")
		if title:
			plt.title(title)
		plt.grid ()
		plt.ylim (freqs_plot.min (), freqs_plot.max () * 1.05)


	def plot_dos(self, omega_e, dos_e):
		plt.axes ([.8, .07, .17, .85])
		plt.fill_betweenx (x1=0., x2=dos_e, y=omega_e, color='lightgrey', edgecolor='k')
		plt.plot (dos_e, omega_e, "-", color='black')
		plt.ylim (omega_e.min (), omega_e.max () * 1.05)
		plt.xticks ([], [])
		plt.grid ()
		plt.xlim (0, dos_e.max () * (1. + BUFFER_PLOT))
		plt.xlabel ("DOS", fontsize=16, fontweight='bold')
		
	
	def show(self):
		# millis = int (round (time.time () * 1000))
		# plt.savefig ('fig-' + str (millis))
		# plt.close (self.fig)
		plt.show ()
