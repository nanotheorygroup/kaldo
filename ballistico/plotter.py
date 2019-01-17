import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import ballistico.geometry_helper as geometry_helper
import datetime
import numpy as np
import time
from ballistico.interpolation_controller import interpolator
import os
import ballistico.constants as constants
import seaborn as sns

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
        plt.ylim([observable[3:].min(), observable[3:].max()])
        plt.xlim([frequencies[3:].min(), frequencies[3:].max()])
        plt.ylabel (observable_name, fontsize=16, fontweight='bold')
        plt.xlabel ("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
        if self.is_persistency_enabled:
            fig.savefig (self.folder + observable_name + '.pdf')
        if self.is_showing:
            plt.show ()

    def plot_dos(self):
        phonons = self.phonons
        fig = plt.figure ()
        # sns.set(color_codes=True)
        ax = sns.kdeplot(phonons.frequencies.flatten())
        plt.xlabel("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
        if self.is_persistency_enabled:
            fig.savefig (self.folder + 'dos.pdf')
        if self.is_showing:
            plt.show()

    def plot_dispersion(self, symmetry='fcc', n_k_points=100):
        atoms = self.phonons.atoms
        cell = atoms.cell
        fig = plt.figure ()
        k_list, q, Q, point_names = geometry_helper.create_k_and_symmetry_space (cell, symmetry=symmetry, n_k_points=n_k_points)
        freqs_plot, _, _, vel_plot = self.phonons.second_quantities_k_list(k_list)

        plt.ylabel ('frequency/$THz$')
        plt.xticks (Q, point_names)
        plt.xlim (q[0], q[-1])
        plt.plot (q, freqs_plot, "-")
        plt.grid ()
        plt.ylim (freqs_plot.min (), freqs_plot.max () * 1.05)
        if self.is_persistency_enabled:
            fig.savefig (self.folder + 'dispersion' + '.pdf')
        if self.is_showing:
            plt.show()

        plt.ylabel('velocity norm/$100m/s$')
        plt.xticks(Q, point_names)
        plt.xlim(q[0], q[-1])
        plt.plot(q, np.linalg.norm(vel_plot[:, :, :], axis=2), "-")
        plt.grid()
        # plt.ylim(freqs_plot.min(), freqs_plot.max() * 1.05)
        if self.is_persistency_enabled:
            fig.savefig(self.folder + 'velocity.pdf')
        if self.is_showing:
            plt.show()