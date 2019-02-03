import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import ballistico.geometry_helper as geometry_helper
import datetime
import numpy as np
import time
from ballistico.interpolation_controller import interpolator
import os
import ballistico.constants as constants
from sklearn.neighbors.kde import KernelDensity

import seaborn as sns
sns.set(color_codes=True)

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

    def plot_vs_frequency(self, observable, observable_name):
        # TODO: We should check if the flattn is C-like and still compatible with Sheng 'F' like
        frequencies = self.phonons.frequencies.flatten ()
        observable = observable.flatten ()
        fig = plt.figure ()
        plt.scatter(frequencies[3:], observable[3:])
        observable[np.isnan(observable)] = 0
        # plt.ylim([observable[3:].min(), observable[3:].max()])
        # plt.xlim([frequencies[3:].min(), frequencies[3:].max()])
        plt.ylabel (observable_name, fontsize=16, fontweight='bold')
        plt.xlabel ("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
        if self.is_persistency_enabled:
            fig.savefig (self.folder + observable_name + '.pdf')
        if self.is_showing:
            plt.show ()

    def plot_dos(self, bandwidth=.5):
        phonons = self.phonons
        fig = plt.figure ()
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(phonons.frequencies.flatten().reshape(-1, 1))
        x = np.linspace(0, phonons.frequencies.max(), 200)
        y = np.exp(kde.score_samples(x.reshape(-1, 1)))
        plt.plot(x, y)
        plt.fill_between(x, y, alpha=.2)
        plt.xlabel("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
        if self.is_persistency_enabled:
            fig.savefig (self.folder + 'dos.pdf')
        if self.is_showing:
            plt.show()

    def plot_dispersion(self, symmetry='fcc', n_k_points=200):
        atoms = self.phonons.atoms
        cell = atoms.cell
        fig = plt.figure ()
        if symmetry == 'nw':
            q = np.linspace(0, 0.5, n_k_points)
            k_list = np.zeros((n_k_points, 3))
            k_list[:, 0] = q
            k_list[:, 2] = q
            Q = [0, 0.5]
            point_names = ['$\\Gamma$', 'X']
        else:
            k_list, q, Q, point_names = geometry_helper.create_k_and_symmetry_space (cell, symmetry=symmetry, n_k_points=n_k_points)

        try:
            freqs_plot, _, _, vel_plot = self.phonons.second_quantities_k_list(k_list)
        except AttributeError as err:
            print(err)
            freqs_plot = np.zeros((k_list.shape[0], self.phonons.n_modes))
            vel_plot = np.zeros((k_list.shape[0], self.phonons.n_modes, 3))
            for mode in range(self.phonons.n_modes):
                with_fourier = False

                freqs_plot[:, mode] = interpolator(k_list, self.phonons.frequencies[:, :, :, mode],
                                                   with_fourier=with_fourier)
                for alpha in range(3):
                    vel_plot[:, mode, alpha] = interpolator(k_list, self.phonons.velocities[:, :, :, mode, alpha],
                                                            with_fourier=with_fourier)

        plt.ylabel ('frequency/$THz$')
        plt.xticks (Q, point_names)
        plt.xlim (q[0], q[-1])
        plt.plot (q, freqs_plot, ".")
        plt.grid ()
        plt.ylim (freqs_plot.min (), freqs_plot.max () * 1.05)
        if self.is_persistency_enabled:
            fig.savefig (self.folder + 'dispersion' + '.pdf')
        if self.is_showing:
            plt.show()

        plt.ylabel('velocity norm/$100m/s$')
        plt.xticks(Q, point_names)
        plt.xlim(q[0], q[-1])
        plt.plot(q, np.linalg.norm(vel_plot[:, :, :], axis=2), ".")
        plt.grid()
        # plt.ylim(freqs_plot.min(), freqs_plot.max() * 1.05)
        if self.is_persistency_enabled:
            fig.savefig(self.folder + 'velocity.pdf')
        if self.is_showing:
            plt.show()
