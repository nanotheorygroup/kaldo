"""
Fourier interpolation
F. P. Russell, K. A. Wilkinson, P. H. J. Kelly, and C.-K. Skylaris,
“Optimised three-dimensional Fourier interpolation: An analysis of techniques and application to a
linear-scaling density functional theory code,” Computer Physics Communications, vol. 187, pp. 8–19, Feb. 2015.

Seekpath
Y. Hinuma, G. Pizzi, Y. Kumagai, F. Oba, I. Tanaka, Band structure diagram paths based on crystallography, Comp. Mat. Sci. 128, 140 (2017) (JOURNAL LINK, arXiv link).

Spglib
A. Togo, I. Tanaka, "Spglib: a software library for crystal symmetry search", arXiv:1808.01590 (2018) (spglib arXiv link).

"""
import matplotlib.pyplot as plt
import seekpath
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from scipy import ndimage
from kaldo.helpers.storage import get_folder_from_label
from kaldo.observables.harmonic_with_q import HarmonicWithQ
import os

BUFFER_PLOT = .2
DEFAULT_FOLDER = 'plots'


def convert_to_spg_structure(atoms):
    cell = atoms.cell
    scaled_positions = atoms.get_positions().dot(np.linalg.inv(atoms.cell))
    spg_struct = (cell, scaled_positions, atoms.get_atomic_numbers())
    return spg_struct


def resample_fourier(observable, increase_factor):
    matrix = np.fft.fftn(observable, axes=(0, 1, 2))
    bigger_matrix = np.zeros((increase_factor * matrix.shape[0], increase_factor * matrix.shape[1],
                              increase_factor * matrix.shape[2])).astype(complex)
    half = int(matrix.shape[0] / 2)
    bigger_matrix[0:half, 0:half, 0:half] = matrix[0:half, 0:half, 0:half]
    bigger_matrix[-half:, 0:half, 0:half] = matrix[-half:, 0:half, 0:half]
    bigger_matrix[0:half, -half:, 0:half] = matrix[0:half, -half:, 0:half]
    bigger_matrix[-half:, -half:, 0:half] = matrix[-half:, -half:, 0:half]
    bigger_matrix[0:half, 0:half, -half:] = matrix[0:half, 0:half, -half:]
    bigger_matrix[-half:, 0:half, -half:] = matrix[-half:, 0:half, -half:]
    bigger_matrix[0:half, -half:, -half:] = matrix[0:half, -half:, -half:]
    bigger_matrix[-half:, -half:, -half:] = matrix[-half:, -half:, -half:]
    bigger_matrix = (np.fft.ifftn(bigger_matrix, axes=(0, 1, 2)))
    bigger_matrix *= increase_factor ** 3
    return bigger_matrix


def interpolator(k_list, observable, fourier_order=0, interpolation_order=0, is_wrapping=True):
    # Here we can put a pipeline of several interpolator
    if fourier_order:
        observable = resample_fourier(observable, increase_factor=fourier_order).real

    k_size = np.array(observable.shape)
    if is_wrapping:
        out = ndimage.map_coordinates(observable, (k_list * k_size).T, order=interpolation_order, mode='wrap')
    else:
        out = ndimage.map_coordinates(observable, (k_list * k_size).T, order=interpolation_order)

    return out


def create_k_and_symmetry_space(atoms, n_k_points=300, symprec=1e-05):
    spg_struct = convert_to_spg_structure(atoms)
    autopath = seekpath.get_path(spg_struct, symprec=symprec)
    path_cleaned = []
    for edge in autopath['path']:
        edge_cleaned = []
        for point in edge:
            if point == 'GAMMA':
                edge_cleaned.append('G')
            else:
                edge_cleaned.append(point.replace('_', ''))
        path_cleaned.append(edge_cleaned)
    point_coords_cleaned = {}
    for key in autopath['point_coords'].keys():
        if key == 'GAMMA':
            point_coords_cleaned['G'] = autopath['point_coords'][key]
        else:
            point_coords_cleaned[key.replace('_', '')] = autopath['point_coords'][key]

    density = n_k_points / 5
    bandpath = atoms.cell.bandpath(path=path_cleaned,
                                   density=density,
                                   special_points=point_coords_cleaned)

    previous_point_position = -1.
    kpath = bandpath.kpts
    points_positions = []
    points_names = []
    kpoint_axis = bandpath.get_linear_kpoint_axis()
    for i in range(len(kpoint_axis[-2])):
        point_position = kpoint_axis[-2][i]
        point_name = kpoint_axis[-1][i]
        if point_position != previous_point_position:
            points_positions.append(point_position)
            points_names.append(point_name)
        previous_point_position = point_position

    points_positions = np.array(points_positions)
    points_positions /= points_positions.max()
    for i in range(len(points_names)):
        if points_names[i] == 'GAMMA':
            points_names[i] = '$\\Gamma$'
    return kpath, points_positions, points_names


def plot_vs_frequency(phonons, observable, observable_name, is_showing=True):
    physical_mode = phonons.physical_mode.flatten()
    frequency = phonons.frequency.flatten()
    observable = observable.flatten()
    fig = plt.figure()
    plt.scatter(frequency[physical_mode], observable[physical_mode], s=5)
    observable[np.isnan(observable)] = 0
    plt.ylabel(observable_name, fontsize=16)
    plt.xlabel("$\\nu$ (THz)", fontsize=16)
    plt.ylim(observable.min(), observable.max())
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=16)
    folder = get_folder_from_label(phonons, base_folder=DEFAULT_FOLDER)
    if not os.path.exists(folder):
        os.makedirs(folder)
    fig.savefig(folder + '/' + observable_name + '.png')
    if is_showing:
        plt.show()
    else:
        plt.close()


def plot_dos(phonons, bandwidth=.05,n_points=200, is_showing=True):
    
    fig = plt.figure()
    physical_mode = phonons.physical_mode.flatten(order='C')
    frequency = phonons.frequency.flatten(order='C')
    frequency = frequency[physical_mode]
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(frequency.reshape(-1, 1))
    x = np.linspace(frequency.min(), phonons.frequency.max(), n_points)
    y = np.exp(kde.score_samples(x.reshape((-1, 1))))
    plt.plot(x, y)
    plt.fill_between(x, y, alpha=.2)
    plt.xlabel("$\\nu$ (THz)", fontsize=16)
    plt.ylabel('DOS',fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=16)
    folder = get_folder_from_label(phonons, base_folder=DEFAULT_FOLDER)
    if not os.path.exists(folder):
        os.makedirs(folder)
    fig.savefig(folder + '/' + 'dos.png')
    if is_showing:
        plt.show()
    else:
        plt.close()

def plot_dispersion(phonons, n_k_points=300, is_showing=True, symprec=1e-3, is_nw=None, with_velocity=True, color='b', is_unfolding=False):
    # TODO: remove useless symmetry flag
    atoms = phonons.atoms

    if is_nw is None and phonons.is_nw:
        is_nw = phonons.is_nw
    if is_nw:
        q = np.linspace(0, 0.5, n_k_points)
        k_list = np.zeros((n_k_points, 3))
        k_list[:, 0] = q
        k_list[:, 2] = q
        Q = [0, 0.5]
        point_names = ['$\\Gamma$', 'X']
    else:
        try:
            k_list, Q, point_names = create_k_and_symmetry_space(atoms, n_k_points=n_k_points, symprec=symprec)
            q = np.linspace(0, 1, k_list.shape[0])
        except seekpath.hpkot.SymmetryDetectionError as err:
            print(err)
            q = np.linspace(0, 0.5, n_k_points)
            k_list = np.zeros((n_k_points, 3))
            k_list[:, 0] = q
            k_list[:, 2] = q
            Q = [0, 0.5]
            point_names = ['$\\Gamma$', 'X']
    freqs_plot = []
    vel_plot = []
    vel_norm = []
    for q_point in k_list:
        phonon = HarmonicWithQ(q_point, phonons.forceconstants.second,
                               distance_threshold=phonons.forceconstants.distance_threshold,
                               storage='memory',
                               is_unfolding=is_unfolding)
        freqs_plot.append(phonon.frequency.flatten())

        if with_velocity:
            val_value = phonon.velocity[0]
            vel_plot.append(val_value)
            vel_norm.append(np.linalg.norm(val_value, axis=-1))
    freqs_plot = np.array(freqs_plot)
    if with_velocity:
        vel_plot = np.array(vel_plot)
        vel_norm = np.array(vel_norm)

    fig1, ax1 = plt.subplots()

    plt.tick_params(axis='both', which='minor', labelsize=16)
    plt.ylabel("$\\nu$ (THz)", fontsize=16)
    plt.xlabel('$q$', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)

    plt.xticks(Q, point_names)
    plt.xlim(q[0], q[-1])
    if color is not None:
        plt.plot(q, freqs_plot, '.', color=color, linewidth=4, markersize=4)
    else:
        plt.plot(q, freqs_plot, '.', linewidth=4, markersize=4)
    plt.grid()
    plt.ylim(freqs_plot.min(), freqs_plot.max() * 1.05)
    folder = get_folder_from_label(phonons, base_folder=DEFAULT_FOLDER)
    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    fig1.savefig(folder + '/' + 'dispersion' + '.png')
    np.savetxt(folder + '/' + 'q', q)
    np.savetxt(folder + '/' + 'dispersion', freqs_plot)

    if is_showing:
        plt.show()
    else:
        plt.close()
    if with_velocity:
        for alpha in range(3):
            np.savetxt(folder + '/' + 'velocity_' + str(alpha), vel_plot[:, :, alpha])

        fig2, ax2 = plt.subplots()
        plt.ylabel('$|v|(\AA/ps)$', fontsize=16)
        plt.xlabel('$q$', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.tick_params(axis='both', which='minor', labelsize=20)
        plt.xticks(Q, point_names)
        plt.xlim(q[0], q[-1])
        if color is not None:
            plt.plot(q, vel_norm[:, :], '.', linewidth=4, markersize=4, color=color)
        else:
            plt.plot(q, vel_norm[:, :], '.', linewidth=4, markersize=4)

        plt.grid()
        plt.tick_params(axis='both', which='major', labelsize=16)

        fig2.savefig(folder + '/' + 'velocity.png')
        np.savetxt(folder + '/' + 'velocity_norm',  vel_norm)

        if is_showing:
            plt.show()
        else:
            plt.close()

def cumulative_cond_cal(freq,full_cond,n_phonons):

  conductivity = np.einsum('maa->m', 1/3 * full_cond)
  conductivity = conductivity.reshape(n_phonons)
  cumulative_cond = np.zeros_like(conductivity)
  freq_reshaped = freq.reshape(n_phonons)

  for mu in range(cumulative_cond.size):
      single_cumulative_cond = conductivity[(freq_reshaped < freq_reshaped[mu])].sum()
      cumulative_cond[mu] = single_cumulative_cond
  
  return cumulative_cond


def plot_crystal(phonons):
    plot_vs_frequency(phonons, phonons.diffusivity, 'diffusivity mm2overs')
    plot_dispersion(phonons)
    plot_dos(phonons)
    # heat_capacity in 10-23 m2 kg s-2 K-1
    plot_vs_frequency(phonons, phonons.heat_capacity, 'heat_capacity')
    plot_vs_frequency(phonons, phonons.bandwidth, 'gamma_THz')
    plot_vs_frequency(phonons, phonons.phase_space, 'phase_space')
