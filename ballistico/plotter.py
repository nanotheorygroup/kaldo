import matplotlib.pyplot as plt
import numpy as np
import os
import seekpath

from sklearn.neighbors.kde import KernelDensity
from scipy.ndimage import map_coordinates

BUFFER_PLOT = .2
DEFAULT_FOLDER = 'plots/'

class Plotter (object):
    def __init__(self, phonons, folder=None, is_showing=True, is_persistency_enabled=True):
        self.phonons = phonons
        self.system = phonons.atoms
        if not folder:
            folder = phonons.folder_name + '/' + DEFAULT_FOLDER
        self.folder = folder
        self.is_persistency_enabled = is_persistency_enabled
        self.is_showing = is_showing

        if self.is_persistency_enabled:
            if not os.path.exists (self.folder):
                os.makedirs (self.folder)

    def plot_vs_frequency(self, observable, observable_name):
        frequencies = self.phonons.frequencies.flatten ()
        observable = observable.flatten ()
        fig = plt.figure ()
        plt.scatter(frequencies[3:], observable[3:],s=5)
        observable[np.isnan(observable)] = 0
        plt.ylabel (observable_name, fontsize=16, fontweight='bold')
        plt.xlabel ("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
        if self.is_persistency_enabled:
            fig.savefig (self.folder + observable_name + '.pdf')
        if self.is_showing:
            plt.show ()

    def plot_dos(self, bandwidth=.3):
        phonons = self.phonons
        fig = plt.figure ()
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(phonons.frequencies.flatten(order='C').reshape(-1, 1))
        x = np.linspace(0, phonons.frequencies.max(), 200)
        y = np.exp(kde.score_samples(x.reshape((-1, 1), order='C')))
        plt.plot(x, y)
        plt.fill_between(x, y, alpha=.2)
        plt.xlabel("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
        if self.is_persistency_enabled:
            fig.savefig (self.folder + 'dos.pdf')
        if self.is_showing:
            plt.show()

    def plot_dispersion(self, symmetry='fcc', n_k_points=100):
        atoms = self.phonons.atoms
        reference_distance = 0.5/n_k_points
        fig1 = plt.figure ()
        if symmetry == 'nw':
            q = np.linspace(0, 0.5, n_k_points)
            k_list = np.zeros((n_k_points, 3))
            k_list[:, 0] = q
            k_list[:, 2] = q
            Q = [0, 0.5]
            point_names = ['$\\Gamma$', 'X']
        else:
            k_list, q, Q, point_names = self.create_k_and_symmetry_space (atoms, reference_distance=reference_distance)
        if self.phonons.is_able_to_calculate:
            freqs_plot, _, _, vel_plot = self.phonons.second_quantities_k_list(k_list)
        else:
            freqs_plot = np.zeros((k_list.shape[0], self.phonons.n_modes))
            vel_plot = np.zeros((k_list.shape[0], self.phonons.n_modes, 3))
            for mode in range(self.phonons.n_modes):

                freqs_plot[:, mode] = self.map_interpolator(k_list, self.phonons.frequencies[:, :, :, mode])
                for alpha in range(3):
                    vel_plot[:, mode, alpha] = self.map_interpolator(k_list, self.phonons.velocities[:, :, :, mode, alpha])

        plt.ylabel ('frequency/$THz$')
        plt.xticks (Q, point_names)
        plt.xlim (q[0], q[-1])
        plt.plot (q, freqs_plot, ".")
        plt.grid ()
        plt.ylim (freqs_plot.min (), freqs_plot.max () * 1.05)
        if self.is_persistency_enabled:
            fig1.savefig (self.folder + 'dispersion' + '.pdf')
        if self.is_showing:
            plt.show()
            
        fig2 = plt.figure ()
        plt.ylabel('velocity norm/$100m/s$')
        plt.xticks(Q, point_names)
        plt.xlim(q[0], q[-1])
        plt.plot(q, np.linalg.norm(vel_plot[:, :, :], axis=2), ".")
        plt.grid()

        if self.is_persistency_enabled:
            fig2.savefig(self.folder + 'velocity.pdf')
        if self.is_showing:
            plt.show()

    def create_k_and_symmetry_space(self, atoms, reference_distance=0.02):
        cell = atoms.cell
        scaled_positions = atoms.get_positions().dot(np.linalg.inv(atoms.cell))
        inp = (cell, scaled_positions, atoms.get_atomic_numbers())
        explicit_data = seekpath.getpaths.get_explicit_k_path(inp, reference_distance=reference_distance)
        kpath = explicit_data['explicit_kpoints_rel']
        n_k_points = kpath.shape[0]
        label_positions = np.array(explicit_data['explicit_segments']).flatten()
        label_names = np.array(explicit_data['path']).flatten()
        x_label_positions = [label_positions[0]]
        x_label_names = [label_names[0]]
        for i in range(1, label_positions.size - 1, 2):
            x_label_names.append((label_names[i]))
            x_label_positions.append((label_positions[i] + label_positions[i+1]) / 2)
        x_label_positions.append(label_positions[-1])
        x_label_names.append(label_names[-1])
        point_names = x_label_names
        Q = np.array(x_label_positions)
        Q /= Q.max()
        q = np.linspace(0, 1, n_k_points)
        for i in range(len(point_names)):
            if point_names[i] == 'GAMMA':
                point_names[i] = '$\Gamma$'
        return kpath, q, Q, point_names

    def map_interpolator(self, k_list, observable):
        k_size = np.array(observable.shape)
        return map_coordinates(observable, (k_list * k_size).T, order=0, mode='wrap')
