"""
Ballistico
Anharmonic Lattice Dynamics
"""
import matplotlib.pyplot as plt
import seekpath
from sklearn.neighbors.kde import KernelDensity
import numpy as np
from scipy import ndimage
from ballistico.helpers.tools import convert_to_spg_structure
from ase.dft.kpoints import BandPath
BUFFER_PLOT = .2
DEFAULT_FOLDER = 'plots/'


def resample_fourier(observable, increase_factor):
    matrix = np.fft.fftn (observable, axes=(0, 1, 2))
    bigger_matrix = np.zeros ((increase_factor * matrix.shape[0], increase_factor * matrix.shape[1],increase_factor * matrix.shape[2])).astype (complex)
    half = int(matrix.shape[0] / 2)
    bigger_matrix[0:half, 0:half, 0:half] = matrix[0:half, 0:half, 0:half]
    bigger_matrix[-half:, 0:half, 0:half] = matrix[-half:, 0:half, 0:half]
    bigger_matrix[0:half, -half:, 0:half] = matrix[0:half, -half:, 0:half]
    bigger_matrix[-half:, -half:, 0:half] = matrix[-half:, -half:, 0:half]
    bigger_matrix[0:half, 0:half, -half:] = matrix[0:half, 0:half, -half:]
    bigger_matrix[-half:, 0:half, -half:] = matrix[-half:, 0:half, -half:]
    bigger_matrix[0:half, -half:, -half:] = matrix[0:half, -half:, -half:]
    bigger_matrix[-half:, -half:, -half:] = matrix[-half:, -half:, -half:]
    bigger_matrix = (np.fft.ifftn (bigger_matrix, axes=(0, 1, 2)))
    bigger_matrix *= increase_factor ** 3
    return bigger_matrix


def interpolator(k_list, observable, fourier_order=0, interpolation_order=0, is_wrapping=True):
    # Here we can put a pipeline of several interpolator
    if fourier_order:
        observable = resample_fourier (observable, increase_factor=fourier_order).real

    k_size = np.array(observable.shape)
    if is_wrapping:
        out = ndimage.map_coordinates (observable, (k_list * k_size).T, order=interpolation_order, mode='wrap')
    else:
        out = ndimage.map_coordinates (observable, (k_list * k_size).T, order=interpolation_order)

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
    frequencies = phonons.frequencies.flatten ()
    observable = observable.flatten ()
    fig = plt.figure ()
    plt.scatter(frequencies[3:], observable[3:],s=5)
    observable[np.isnan(observable)] = 0
    plt.ylabel (observable_name, fontsize=16, fontweight='bold')
    plt.xlabel ("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
    plt.ylim(observable.min(), observable.max())
    fig.savefig (phonons.folder + '/' + observable_name + '.pdf')
    if is_showing:
        plt.show ()

def plot_dos(phonons, bandwidth=.3, is_showing=True):
    fig = plt.figure ()
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(phonons.frequencies.flatten(order='C').reshape(-1, 1))
    x = np.linspace(phonons.frequencies.min(), phonons.frequencies.max(), 200)
    y = np.exp(kde.score_samples(x.reshape((-1, 1), order='C')))
    plt.plot(x, y)
    plt.fill_between(x, y, alpha=.2)
    plt.xlabel("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
    fig.savefig (phonons.folder + '/' + 'dos.pdf')
    if is_showing:
        plt.show()

def plot_dispersion(phonons, n_k_points=300, is_showing=True, symprec=1e-5):
    #TODO: remove useless symmetry flag
    atoms = phonons.atoms
    fig1 = plt.figure ()
    try:
        k_list, Q, point_names = create_k_and_symmetry_space (atoms, n_k_points=n_k_points, symprec=symprec)
        q = np.linspace(0, 1, k_list.shape[0])
    except seekpath.hpkot.SymmetryDetectionError as err:
        print(err)
        q = np.linspace(0, 0.5, n_k_points)
        k_list = np.zeros((n_k_points, 3))
        k_list[:, 0] = q
        k_list[:, 2] = q
        Q = [0, 0.5]
        point_names = ['$\\Gamma$', 'X']

    if phonons.is_able_to_calculate:
        freqs_plot = phonons.calculate_second_order_observable('frequencies', k_list)
        vel_plot = phonons.calculate_second_order_observable('velocities', k_list)
        vel_norm = np.linalg.norm(vel_plot, axis=-1)
        # print(vel_plot)
    else:
        freqs_plot = np.zeros((k_list.shape[0], phonons.n_modes))
        vel_plot = np.zeros((k_list.shape[0], phonons.n_modes, 3))
        vel_norm = np.zeros((k_list.shape[0], phonons.n_modes))

        frequencies = phonons.frequencies.reshape((phonons.kpts[0], phonons.kpts[1],
                                                        phonons.kpts[2], phonons.n_modes), order='C')
        velocities = phonons.velocities.reshape((phonons.kpts[0], phonons.kpts[1],
                                                       phonons.kpts[2], phonons.n_modes, 3), order='C')
        for mode in range(phonons.n_modes):
            freqs_plot[:, mode] = interpolator(k_list, frequencies[..., mode], fourier_order=5, interpolation_order=2)

            for alpha in range(3):
                vel_plot[:, mode, alpha] = interpolator(k_list, velocities[..., mode, alpha], interpolation_order=0, is_wrapping=False)
            vel_norm[:, mode] = interpolator(k_list, np.linalg.norm(velocities[..., mode, :],axis=-1), interpolation_order=0, is_wrapping=False)

    plt.ylabel ('frequency/$THz$', fontsize=25, fontweight='bold')
    plt.xlabel('$\mathbf{q}$', fontsize=25, fontweight='bold')
    plt.xticks (Q, point_names)
    plt.xlim (q[0], q[-1])
    plt.plot (q, freqs_plot, '.', linewidth=1, markersize=4)
    plt.grid ()
    plt.ylim (freqs_plot.min (), freqs_plot.max () * 1.05)
    fig1.savefig (phonons.folder + '/' + 'dispersion' + '.pdf')
    if is_showing:
        plt.show()

    for alpha in range(3):
        fig2 = plt.figure ()
        plt.ylabel('v_$' + str(alpha) + '$/($100m/s$)', fontsize=25, fontweight='bold')
        plt.xlabel('$\mathbf{q}$', fontsize=25, fontweight='bold')
        plt.xticks(Q, point_names)
        plt.xlim(q[0], q[-1])
        plt.plot(q, vel_plot[:, :, alpha], '.', linewidth=1, markersize=4)
        plt.grid()
        fig2.savefig(phonons.folder + '/' + 'velocity.pdf')
        if is_showing:
            plt.show()

    fig2 = plt.figure ()
    plt.ylabel('$|v|$/($100m/s$)', fontsize=25, fontweight='bold')
    plt.xlabel('$\mathbf{q}$', fontsize=25, fontweight='bold')

    plt.xticks(Q, point_names)
    plt.xlim(q[0], q[-1])
    plt.plot(q, vel_norm[:, :], '.', linewidth=1, markersize=4)
    plt.grid()
    fig2.savefig(phonons.folder + '/' + 'velocity.pdf')
    if is_showing:
        plt.show()
