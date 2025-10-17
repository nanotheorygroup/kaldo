"""
_Plotter module for visualizing phonon properties.

References
----------
Fourier interpolation:
    F. P. Russell, K. A. Wilkinson, P. H. J. Kelly, and C.-K. Skylaris,
    "Optimised three-dimensional Fourier interpolation: An analysis of techniques and application to a
    linear-scaling density functional theory code," Computer Physics Communications, vol. 187, pp. 8–19, Feb. 2015.

Seekpath:
    Y. Hinuma, G. Pizzi, Y. Kumagai, F. Oba, I. Tanaka,
    Band structure diagram paths based on crystallography, Comp. Mat. Sci. 128, 140 (2017).

Spglib:
    A. Togo, I. Tanaka, "Spglib: a software library for crystal symmetry search", arXiv:1808.01590 (2018).
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seekpath
from scipy import ndimage
from kaldo.observables.harmonic_with_q import HarmonicWithQ
from kaldo.conductivity import Conductivity
from kaldo.helpers.logger import get_logger

logging = get_logger()

BUFFER_PLOT = .2
DEFAULT_FOLDER = 'plots'

# Public API - use these functions directly
__all__ = [
    'plot_vs_frequency',
    'plot_dos',
    'plot_dispersion',
    'plot_crystal',
    'plot_amorphous',
]


class _Plotter:
    """Internal visualization toolkit for phonon properties.

    Note: This class is internal. Users should use the module-level functions
    like plot_crystal(), plot_amorphous(), plot_dos(), etc. instead of
    instantiating this class directly.

    This class provides comprehensive plotting capabilities for phonon dispersion,
    density of states, thermal conductivity, and other phonon-related properties.

    Parameters
    ----------
    phonons : Phonons
        Phonons object containing the phonon properties to visualize
    backend : str, optional
        Matplotlib backend to use. Default: None (uses matplotlib default)
    style : dict, optional
        Custom matplotlib style settings. Default: None

    Examples
    --------
    This class is internal - users should use module-level functions:
    >>> from kaldo.controllers.plotter import plot_dispersion, plot_dos, plot_crystal
    >>> plot_dispersion(phonons, is_showing=False)
    >>> plot_dos(phonons, is_showing=False)
    >>> plot_crystal(phonons, figsize=(8, 6))
    """

    def __init__(self, phonons, backend=None, style=None):
        """Initialize _Plotter with phonons object.

        Parameters
        ----------
        phonons : Phonons
            Phonons object containing the phonon properties to visualize
        backend : str, optional
            Matplotlib backend to use. Default: None
        style : dict, optional
            Custom matplotlib style settings. Default: None
        """
        self.phonons = phonons
        if backend is not None:
            matplotlib.use(backend)
        if style is not None:
            plt.rcParams.update(style)

    @staticmethod
    def _convert_to_spg_structure(atoms):
        """Convert ASE atoms to spglib structure format."""
        cell = atoms.cell
        scaled_positions = atoms.get_positions().dot(np.linalg.inv(atoms.cell))
        spg_struct = (cell, scaled_positions, atoms.get_atomic_numbers())
        return spg_struct

    @staticmethod
    def _resample_fourier(observable, increase_factor):
        """Resample observable using Fourier interpolation."""
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

    @staticmethod
    def _interpolator(k_list, observable, fourier_order=0, interpolation_order=0, is_wrapping=True):
        """Interpolate observable on k-point list.

        Can use Fourier and/or spline interpolation.
        """
        if fourier_order:
            observable = _Plotter._resample_fourier(observable, increase_factor=fourier_order).real

        k_size = np.array(observable.shape)
        if is_wrapping:
            out = ndimage.map_coordinates(observable, (k_list * k_size).T, order=interpolation_order, mode='wrap')
        else:
            out = ndimage.map_coordinates(observable, (k_list * k_size).T, order=interpolation_order)
        return out

    @staticmethod
    def _create_k_and_symmetry_space(atoms, n_k_points=300, symprec=1e-05, manually_defined_path=None):
        """Create k-point path and symmetry labels for band structure plotting.

        Parameters
        ----------
        atoms : ase.Atoms
            Atomic structure
        n_k_points : int
            Number of k-points along the path
        symprec : float
            Symmetry precision for seekpath
        manually_defined_path : ase.dft.kpoints.BandPath, optional
            Manually defined band path. If None, use automatic path detection

        Returns
        -------
        kpath : ndarray
            K-point coordinates
        points_positions : ndarray
            Normalized positions of high-symmetry points
        points_names : list
            Names of high-symmetry points
        """
        if manually_defined_path is not None:
            bandpath = manually_defined_path
        else:
            # Use auto-detect scheme
            spg_struct = _Plotter._convert_to_spg_structure(atoms)
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

    @staticmethod
    def _set_fig_properties(ax_list, panel_color_str='black', line_width=2):
        """Apply consistent formatting to matplotlib axes.

        Parameters
        ----------
        ax_list : list
            List of matplotlib axes objects
        panel_color_str : str
            Color for panel borders and ticks. Default: 'black'
        line_width : int
            Width of panel borders and tick marks. Default: 2
        """
        tl = 4  # major tick length
        tw = 2  # tick width
        tlm = 2  # minor tick length

        for ax in ax_list:
            ax.tick_params(which='major', length=tl, width=tw)
            ax.tick_params(which='minor', length=tlm, width=tw)
            ax.tick_params(which='both', axis='both', direction='in',
                           right=True, top=True)
            ax.spines['bottom'].set_color(panel_color_str)
            ax.spines['top'].set_color(panel_color_str)
            ax.spines['left'].set_color(panel_color_str)
            ax.spines['right'].set_color(panel_color_str)

            ax.spines['bottom'].set_linewidth(line_width)
            ax.spines['top'].set_linewidth(line_width)
            ax.spines['left'].set_linewidth(line_width)
            ax.spines['right'].set_linewidth(line_width)

            for t in ax.xaxis.get_ticklines():
                t.set_color(panel_color_str)
                t.set_linewidth(line_width)
            for t in ax.yaxis.get_ticklines():
                t.set_color(panel_color_str)
                t.set_linewidth(line_width)

    def _calculate_dispersion_data(self, n_k_points=300, symprec=1e-3, manually_defined_path=None):
        """Calculate dispersion and velocity data along high-symmetry path.
        
        This is a shared helper method used by both plot_dispersion and plot_crystal.
        
        Parameters
        ----------
        n_k_points : int
            Number of k-points along the path. Default: 300
        symprec : float
            Symmetry precision for automatic path detection. Default: 1e-3
        manually_defined_path : ase.dft.kpoints.BandPath, optional
            Manually defined band path. If None, uses automatic detection.
            
        Returns
        -------
        q : ndarray
            Normalized q-point coordinates (0 to 1)
        Q : ndarray
            Positions of high-symmetry points
        point_names : list
            Names of high-symmetry points
        freqs_plot : ndarray
            Phonon frequencies along path
        vel_norm_plot : ndarray
            Group velocity norms along path
        vel_plot : ndarray
            Full velocity vectors along path
        """
        atoms = self.phonons.atoms
        if self.phonons.is_nw:
            q = np.linspace(0, 0.5, n_k_points)
            k_list = np.zeros((n_k_points, 3))
            k_list[:, 0] = q
            k_list[:, 2] = q
            Q = [0, 0.5]
            point_names = ['$\\Gamma$', 'X']
        else:
            try:
                k_list, Q, point_names = self._create_k_and_symmetry_space(
                    atoms, n_k_points=n_k_points, symprec=symprec, manually_defined_path=manually_defined_path)
                q = np.linspace(0, 1, k_list.shape[0])
            except seekpath.hpkot.SymmetryDetectionError as err:
                logging.warning(f"Symmetry detection error: {err}. Using default path.")
                q = np.linspace(0, 0.5, n_k_points)
                k_list = np.zeros((n_k_points, 3))
                k_list[:, 0] = q
                k_list[:, 2] = q
                Q = [0, 0.5]
                point_names = ['$\\Gamma$', 'X']

        freqs_plot = []
        vel_plot = []
        vel_norm_plot = []
        for q_point in k_list:
            phonon = HarmonicWithQ(q_point, self.phonons.forceconstants.second,
                                   distance_threshold=self.phonons.forceconstants.distance_threshold,
                                   storage='memory',
                                   is_nw=self.phonons.is_nw,
                                   is_unfolding=self.phonons.is_unfolding)
            freqs_plot.append(phonon.frequency.flatten())
            vel_value = phonon.velocity[0]
            vel_plot.append(vel_value)
            vel_norm_plot.append(np.linalg.norm(vel_value, axis=-1))

        freqs_plot = np.array(freqs_plot)
        vel_plot = np.array(vel_plot)
        vel_norm_plot = np.array(vel_norm_plot)
        
        return q, Q, point_names, freqs_plot, vel_norm_plot, vel_plot
    def _calculate_dos(self, p_atoms_list=None, p_atoms_labels=None, direction=None, 
                       bandwidth=0.05, n_points=200):
        """Calculate density of states (DOS) or projected DOS (PDOS) for multiple atom sets.

        This is the single unified method for all DOS calculations in the plotter.
        Uses the phonons.pdos() method internally.

        Parameters
        ----------
        p_atoms_list : list of lists, optional
            List of atom index sets for projected DOS. If None, automatically groups 
            by chemical species for multi-element systems, or uses all atoms for 
            single-element systems.
        p_atoms_labels : list of str, optional
            Labels for each atom set in p_atoms_list. If None, uses chemical symbols
            for multi-element systems or 'Total' for single-element systems.
        direction : array_like, optional
            3-vector direction for DOS projection. If None, sums over all Cartesian directions.
        bandwidth : float
            Gaussian smearing width. Default: 0.05
            Units: THz
        n_points : int
            Number of frequency points for DOS calculation. Default: 200

        Returns
        -------
        dos_data : list of tuples
            List of (fgrid, pdos) tuples, one for each atom set
        labels : list of str
            List of labels corresponding to each DOS dataset
        """
        # Automatically infer p_atoms_list and labels from atoms object if not provided
        if p_atoms_list is None:
            symbols = self.phonons.atoms.get_chemical_symbols()
            unique_species = list(dict.fromkeys(symbols))  # Preserve order
            
            # For multi-element systems, group by species
            if len(unique_species) > 1:
                p_atoms_list = []
                p_atoms_labels = []
                for species in unique_species:
                    indices = [i for i, s in enumerate(symbols) if s == species]
                    p_atoms_list.append(indices)
                    p_atoms_labels.append(species)
            else:
                # Single-element system: use all atoms
                p_atoms_list = [list(range(self.phonons.n_atoms))]
                if p_atoms_labels is None:
                    p_atoms_labels = ['Total']
        
        # Set default labels if not provided
        if p_atoms_labels is None:
            p_atoms_labels = ['Total'] * len(p_atoms_list)

        # Calculate DOS for each atom set
        dos_data = []
        for p_atoms in p_atoms_list:
            try:
                fgrid, pdos = self.phonons.pdos(p_atoms, direction=direction, 
                                                bandwidth=bandwidth, n_points=n_points)
                dos_data.append((fgrid, pdos))
            except Exception as e:
                logging.warning(f"Failed to calculate DOS for atoms {p_atoms}: {e}")

        return dos_data, p_atoms_labels

    def plot_vs_frequency(self, observable, observable_name, is_showing=True):
        """Create scatter plot of observable vs phonon frequency.

        Parameters
        ----------
        observable : ndarray
            Observable to plot (same shape as phonons)
        observable_name : str
            Name of observable for axis label and filename
        is_showing : bool
            Whether to display the plot. Default: True
        """
        physical_mode = self.phonons.physical_mode.flatten()
        frequency = self.phonons.frequency.flatten()
        observable = observable.flatten()
        fig = plt.figure()
        plt.scatter(frequency[physical_mode], observable[physical_mode], s=5)
        observable[np.isnan(observable)] = 0
        plt.ylabel(observable_name, fontsize=16)
        plt.xlabel("$\\nu$ (THz)", fontsize=16)
        plt.ylim(observable.min(), observable.max())
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.tick_params(axis='both', which='minor', labelsize=16)
        folder = self.phonons.get_folder_from_label(base_folder=DEFAULT_FOLDER)
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig.savefig(folder + '/' + observable_name + '.png')
        if is_showing:
            plt.show()
        else:
            plt.close()

    def plot_dos(self, p_atoms=None, p_atoms_labels=None, direction=None, bandwidth=.05, n_points=200, 
                 is_showing=True, filename='dos', figsize=(8, 6)):
        """Produce a plot of phonon density of states (DOS) or projected phonon DOS (PDOS).

        Parameters
        ----------
        p_atoms : list or list of lists, optional
            Atom indices for projected DOS. If None, automatically groups by chemical 
            species for multi-element systems, or calculates total DOS for single-element systems.
            Providing a list of atom indices returns single PDOS summed over those atoms.
            Providing a list of lists returns one PDOS for each set of indices.
        p_atoms_labels : list of str, optional
            Labels for each atom set in p_atoms for DOS legend. If None, uses chemical 
            symbols for multi-element systems.
        direction : array_like, optional
            3-vector direction for DOS projection. If None, sums over all Cartesian directions.
        bandwidth : float
            Gaussian smearing width. Default: 0.05
            Units: THz
        n_points : int
            Number of frequency points for DOS calculation. Default: 200
        is_showing : bool
            Whether to display the plot. Default: True
        filename : str
            Output filename (without extension). Default: 'dos'
        figsize : tuple
            Figure size (width, height) in inches. Default: (8, 6) for publication.
            Use (8, 6) for larger presentations.
        """
        # Convert single list to list of lists for unified handling
        if p_atoms is None:
            p_atoms_list = None
        elif isinstance(p_atoms[0], (int, np.integer)):
            # Single list of atom indices
            p_atoms_list = [p_atoms]
        else:
            # Already a list of lists
            p_atoms_list = p_atoms

        folder = self.phonons.get_folder_from_label(base_folder=DEFAULT_FOLDER)
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Load kaldo style guide if available
        style_file = os.path.join(os.path.dirname(__file__), 'kaldo_style_guide.mpl')
        if os.path.exists(style_file):
            plt.style.use(style_file)
        
        # Additional style settings
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['mathtext.fontset'] = 'cm'

        # Use unified DOS calculation method
        dos_data, p_atoms_labels = self._calculate_dos(p_atoms_list, p_atoms_labels=p_atoms_labels,
                                                        direction=direction, bandwidth=bandwidth,
                                                        n_points=n_points)

        if not dos_data:
            logging.error('Failed to calculate DOS')
            return

        # Save and plot the DOS
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        self._set_fig_properties([ax])
        
        colors = ['#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E', '#8EBA42']
        for idx, (fgrid, pdos) in enumerate(dos_data):
            np.save(folder + f'/{filename}.npy', np.vstack((fgrid, pdos)))
            color = colors[idx % len(colors)]
            label = p_atoms_labels[idx] if idx < len(p_atoms_labels) else None
            for p in np.expand_dims(pdos, 0) if pdos.ndim == 1 else pdos:
                plt.plot(fgrid, p, c=color, label=label)
                label = None

        plt.xlabel("Frequency (THz)")
        plt.ylabel('DOS')
        if p_atoms_labels is not None and len(dos_data) > 1:
            plt.legend(loc='best')
        fig.savefig(folder + f'/{filename}.png', dpi=300, bbox_inches='tight')

        if is_showing:
            plt.show()
        else:
            plt.close()

    def plot_dispersion(self, p_atoms_list=None, p_atoms_labels=None, bandwidth=.05, n_points=200,
                        n_k_points=300, is_showing=True, symprec=1e-3, with_velocity=True,
                        manually_defined_path=None, folder=None, figsize=(8, 6)):
        """Plot phonon dispersion relation and optionally group velocity with DOS panel.

        Parameters
        ----------
        p_atoms_list : list of lists, optional
            List of atom index sets for projected DOS. If None, plots total DOS.
        p_atoms_labels : list of str, optional
            Labels for each atom set in p_atoms_list for DOS legend. If None, uses 'Total'.
        bandwidth : float
            Gaussian smearing width for DOS calculation. Default: 0.05
            Units: THz
        n_points : int
            Number of frequency points for DOS. Default: 200
        n_k_points : int
            Number of k-points along the path. Default: 300
        is_showing : bool
            Whether to display the plot. Default: True
        symprec : float
            Symmetry precision for automatic path detection. Default: 1e-3
        with_velocity : bool
            Whether to also plot group velocity. Default: True
        manually_defined_path : ase.dft.kpoints.BandPath, optional
            Manually defined band path. If None, uses automatic detection.
        folder : str, optional
            Output folder. If None, uses default from phonons object.
        figsize : tuple
            Figure size (width, height) in inches. Default: (8, 6) for publication.
            Use (8, 6) for larger presentations.
        """
        if not folder:
            folder = self.phonons.get_folder_from_label(base_folder=DEFAULT_FOLDER)
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Load kaldo style guide if available
        style_file = os.path.join(os.path.dirname(__file__), 'kaldo_style_guide.mpl')
        if os.path.exists(style_file):
            plt.style.use(style_file)
        
        # Additional style settings
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['mathtext.fontset'] = 'cm'

        # Calculate dispersion data using shared method
        q, Q, point_names, freqs_plot, vel_norm_plot, vel_plot = self._calculate_dispersion_data(
            n_k_points=n_k_points, symprec=symprec, manually_defined_path=manually_defined_path)

        # By default, plot total DOS (all atoms)
        if p_atoms_list is None:
            p_atoms_list = [list(range(self.phonons.n_atoms))]
            if p_atoms_labels is None:
                p_atoms_labels = ['Total']

        # Calculate DOS using unified method
        dos_data, p_atoms_labels = self._calculate_dos(p_atoms_list, p_atoms_labels,
                                                        direction=None, bandwidth=bandwidth,
                                                        n_points=n_points)

        # Plot dispersion with DOS panel
        fig = plt.figure(figsize=figsize)
        # Create main axes that leave room for DOS panel on the right
        ax = fig.add_axes([0.125, 0.11, 0.7, 0.77])
        self._set_fig_properties([ax])
        plt.sca(ax)
        plt.plot(q[0], freqs_plot[0, 0], 'r.', ms=2)
        plt.plot(q, freqs_plot, 'r.', ms=2)
        plt.axhline(y=0, color='k', ls='-', lw=1)
        for i in range(1, len(Q)-1):
            plt.axvline(x=Q[i], ymin=0, ymax=1, ls='--', lw=2, c='k')
        plt.ylabel('Frequency (THz)')
        plt.xlabel(r'Wave vector ($\frac{2\pi}{a}$)')
        plt.xticks(Q, point_names)
        plt.xlim([Q[0], Q[-1]])

        # Add DOS panel on the right
        if dos_data:
            dosax = fig.add_axes([0.85, .11, .13, .77])
            self._set_fig_properties([dosax])
            colors = ['#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E', '#8EBA42']
            for idx, (fgrid, pdos) in enumerate(dos_data):
                color = colors[idx % len(colors)]
                label = p_atoms_labels[idx] if idx < len(p_atoms_labels) else None
                for p in np.expand_dims(pdos, 0) if pdos.ndim == 1 else pdos:
                    dosax.plot(p, fgrid, c=color, label=label)
                    label = None
            dosax.set_yticks([])
            dosax.set_xticks([])
            dosax.set_xlabel("DOS")
            dosax.set_ylim(ax.get_ylim())
            if p_atoms_labels is not None and len(dos_data) > 1:
                dosax.legend(fontsize=6.5, loc='best')

        plt.savefig(folder + '/dispersion_dos.png', dpi=300, bbox_inches='tight')
        
        np.savetxt(folder + '/q', q)
        np.savetxt(folder + '/dispersion', freqs_plot)
        np.savetxt(folder + '/Q_val', Q)
        np.savetxt(folder + '/point_names', point_names, fmt='%s')

        if is_showing:
            plt.show()
        else:
            plt.close()

        # Plot velocity if requested
        if with_velocity:
            for alpha in range(3):
                np.savetxt(folder + '/velocity_' + str(alpha), vel_plot[:, :, alpha])
            
            fig = plt.figure(figsize=figsize)
            ax = fig.gca()
            self._set_fig_properties([ax])
            plt.plot(q, vel_norm_plot / 10.0, 'r.', ms=2)
            for i in range(1, len(Q)-1):
                plt.axvline(x=Q[i], ymin=0, ymax=1, ls='--', lw=2, c='k')
            plt.ylabel(r'$|v|$ (km/s)')
            plt.xlabel(r'Wave vector ($\frac{2\pi}{a}$)')
            plt.xticks(Q, point_names)
            plt.xlim([Q[0], Q[-1]])
            plt.savefig(folder + '/velocity.png', dpi=300, bbox_inches='tight')
            np.savetxt(folder + '/velocity_norm', vel_norm_plot)
            
            if is_showing:
                plt.show()
            else:
                plt.close()

    def plot_crystal(self, p_atoms_list=None, p_atoms_labels=None, bandwidth=.05, n_points=200,
                     is_showing=True, n_k_points=300, symprec=1e-3, method='inverse', figsize=(8, 6)):
        """Create comprehensive plots for crystal phonon properties.

        Generates a complete set of publication-quality plots including:
        - Dispersion with DOS
        - Velocity vs q (along high-symmetry path)
        - Heat capacity vs frequency
        - Group velocity vs frequency
        - Phase space vs frequency
        - Lifetime vs frequency
        - Scattering rate vs frequency
        - Mean free path vs frequency
        - Per-mode thermal conductivity
        - Cumulative thermal conductivity vs frequency
        - Cumulative thermal conductivity vs mean free path

        Parameters
        ----------
        p_atoms_list : list of lists, optional
            List of atom index sets for projected DOS. If None, automatically groups 
            by chemical species for multi-element systems, or uses all atoms for 
            single-element systems.
        p_atoms_labels : list of str, optional
            Labels for each atom set in p_atoms_list for DOS legend. If None, uses 
            chemical symbols for multi-element systems.
        bandwidth : float
            Gaussian smearing width for DOS calculation. Default: 0.05
            Units: THz
        n_points : int
            Number of frequency points for DOS. Default: 200
        is_showing : bool
            Whether to display plots interactively. Default: True
        n_k_points : int
            Number of k-points for dispersion. Default: 300
        symprec : float
            Symmetry precision for dispersion calculation. Default: 1e-3
        method : str
            Method for conductivity calculation ('rta', 'sc', 'inverse'). Default: 'inverse'
        figsize : tuple
            Figure size (width, height) in inches. Default: (8, 6) for publication.
            Use (8, 6) for larger presentations.
        """
        folder = self.phonons.get_folder_from_label(base_folder=DEFAULT_FOLDER)
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Load kaldo style guide if available
        style_file = os.path.join(os.path.dirname(__file__), 'kaldo_style_guide.mpl')
        if os.path.exists(style_file):
            plt.style.use(style_file)
        
        # Additional style settings
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['mathtext.fontset'] = 'cm'

        # Calculate conductivity for mean free path and kappa plots
        conductivity = Conductivity(phonons=self.phonons, method=method, storage='memory')

        # Get flattened data
        physical_mode = self.phonons.physical_mode.flatten()
        frequency = self.phonons.frequency.flatten()

        # Calculate dispersion and velocity along high-symmetry path using shared method
        if self.phonons.is_nw:
            logging.warning("Comprehensive plotting for nanowires not yet fully supported")
            return

        try:
            q, Q, point_names, freqs_plot, vel_norm_plot, _ = self._calculate_dispersion_data(
                n_k_points=n_k_points, symprec=symprec)
        except Exception as e:
            logging.error(f"Failed to create k-space: {e}")
            return

        # Calculate DOS using unified method
        dos_data, p_atoms_labels = self._calculate_dos(p_atoms_list, p_atoms_labels,
                                                        direction=None, bandwidth=bandwidth,
                                                        n_points=n_points)

        # Plot dispersion with DOS panel
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        self._set_fig_properties([ax])
        plt.plot(q[0], freqs_plot[0, 0], 'r.', ms=2)
        plt.plot(q, freqs_plot, 'r.', ms=2)
        plt.axhline(y=0, color='k', ls='-', lw=1)
        for i in range(1, len(Q)-1):
            plt.axvline(x=Q[i], ymin=0, ymax=1, ls='--', lw=2, c='k')
        plt.ylabel('Frequency (THz)')
        plt.xlabel(r'Wave vector ($\frac{2\pi}{a}$)')
        plt.xticks(Q, point_names)
        plt.xlim([Q[0], Q[-1]])

        # Add DOS panel on the right
        if dos_data:
            dosax = fig.add_axes([0.91, .11, .17, .77])
            self._set_fig_properties([dosax])
            colors = ['#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E', '#8EBA42']
            for idx, (fgrid, pdos) in enumerate(dos_data):
                color = colors[idx % len(colors)]
                label = p_atoms_labels[idx] if idx < len(p_atoms_labels) else None
                for p in np.expand_dims(pdos, 0) if pdos.ndim == 1 else pdos:
                    dosax.plot(p, fgrid, c=color, label=label)
                    label = None
            dosax.set_yticks([])
            dosax.set_xticks([])
            dosax.set_xlabel("DOS")
            dosax.set_ylim(plt.gca().get_ylim())
            if p_atoms_labels is not None and len(dos_data) > 1:
                dosax.legend(fontsize=6.5, loc='best')

        plt.savefig(folder + '/dispersion_dos.png', dpi=300, bbox_inches='tight')
        if is_showing:
            plt.show()
        else:
            plt.close()

        # Velocity vs q (along high-symmetry path)
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        self._set_fig_properties([ax])
        plt.plot(q, vel_norm_plot / 10.0, 'r.', ms=2)
        for i in range(1, len(Q)-1):
            plt.axvline(x=Q[i], ymin=0, ymax=1, ls='--', lw=2, c='k')
        plt.ylabel(r'$|v|$ (km/s)')
        plt.xlabel(r'Wave vector ($\frac{2\pi}{a}$)')
        plt.xticks(Q, point_names)
        plt.xlim([Q[0], Q[-1]])
        plt.savefig(folder + '/velocity_vs_q.png', dpi=300, bbox_inches='tight')
        if is_showing:
            plt.show()
        else:
            plt.close()

        # Heat Capacity
        heat_capacity = self.phonons.heat_capacity.flatten()
        
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        self._set_fig_properties([ax])
        plt.scatter(frequency[physical_mode], 1e23*heat_capacity[physical_mode],
                    facecolor='w', edgecolor='#E24A33', s=10, marker='8')
        plt.ylabel(r"$C_{v}$ ($10^{23}$ J/K)")
        plt.xlabel('Frequency (THz)')
        y_data = 1e23*heat_capacity[physical_mode]
        y_min, y_max = y_data.min(), y_data.max()
        plt.ylim(0.9*y_min, 1.05*y_max)
        plt.savefig(folder + '/cv.png', dpi=300, bbox_inches='tight')
        if is_showing:
            plt.show()
        else:
            plt.close()

        # Group Velocity
        velocity = self.phonons.velocity.real.reshape(-1, 3)
        velocity_norm = np.linalg.norm(velocity, axis=1) / 10.0

        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        self._set_fig_properties([ax])
        plt.scatter(frequency[physical_mode], velocity_norm[physical_mode],
                    facecolor='w', edgecolor='#E24A33', s=10, marker='^')
        plt.xlabel('Frequency (THz)')
        plt.ylabel(r'$|v|$ (km/s)')
        plt.savefig(folder + '/velocity.png', dpi=300, bbox_inches='tight')
        if is_showing:
            plt.show()
        else:
            plt.close()

        # Phase Space
        phase_space = self.phonons.phase_space.flatten()

        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        self._set_fig_properties([ax])
        plt.scatter(frequency[physical_mode], phase_space[physical_mode],
                    facecolor='w', edgecolor='#E24A33', s=10, marker='o')
        plt.xlabel('Frequency (THz)')
        plt.ylabel('Phase space')
        plt.savefig(folder + '/phasespace.png', dpi=300, bbox_inches='tight')
        if is_showing:
            plt.show()
        else:
            plt.close()

        # Lifetime
        scattering_rate = self.phonons.bandwidth.flatten()
        lifetime = scattering_rate ** (-1)

        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        self._set_fig_properties([ax])
        plt.scatter(frequency[physical_mode], lifetime[physical_mode],
                    facecolor='w', edgecolor='#E24A33', s=10, marker='s')
        plt.yscale('log')
        plt.ylabel(r'$\tau$ (ps)')
        plt.xlabel('Frequency (THz)')
        plt.savefig(folder + '/lifetime.png', dpi=300, bbox_inches='tight')
        if is_showing:
            plt.show()
        else:
            plt.close()

        # Scattering Rate
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        self._set_fig_properties([ax])
        plt.scatter(frequency[physical_mode], scattering_rate[physical_mode],
                    facecolor='w', edgecolor='#E24A33', s=10, marker='d')
        plt.ylabel(r'$\Gamma$ (THz)')
        plt.xlabel('Frequency (THz)')
        plt.savefig(folder + '/gamma.png', dpi=300, bbox_inches='tight')
        if is_showing:
            plt.show()
        else:
            plt.close()

        # Mean Free Path
        mean_free_path = conductivity.mean_free_path.reshape(-1, 3) / 10.0
        mean_free_path_norm = np.linalg.norm(mean_free_path, axis=1)

        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        self._set_fig_properties([ax])
        plt.scatter(frequency[physical_mode], mean_free_path_norm[physical_mode],
                    facecolor='w', edgecolor='#E24A33', s=10, marker='8')
        plt.ylabel(r'$\lambda$ (nm)')
        plt.xlabel('Frequency (THz)')
        plt.yscale('log')
        plt.savefig(folder + '/mfp.png', dpi=300, bbox_inches='tight')
        if is_showing:
            plt.show()
        else:
            plt.close()

        # Per-mode Conductivity
        kappa_tensor = conductivity.conductivity.reshape(self.phonons.n_k_points, self.phonons.n_modes, 3, 3)
        kappa_per_mode = kappa_tensor.sum(axis=-1).sum(axis=-1).flatten()

        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        self._set_fig_properties([ax])
        plt.scatter(frequency[physical_mode], kappa_per_mode[physical_mode],
                    facecolor='w', edgecolor='#E24A33', s=10, marker='>')
        plt.axhline(y=0, color='k', ls='--', lw=1)
        plt.ylabel(r'$\kappa_{per \ mode}$ $\left(\frac{\rm{W}}{\rm{m}\cdot\rm{K}}\right)$')
        plt.xlabel('Frequency (THz)')
        plt.savefig(folder + '/kappa_per_mode.png', dpi=300, bbox_inches='tight')
        if is_showing:
            plt.show()
        else:
            plt.close()

        # Cumulative Conductivity vs Frequency
        # Compute cumulative conductivity vs frequency
        kappa_freq = np.einsum('maa->m', 1/3 * kappa_tensor.reshape(-1, 3, 3))
        freq_argsort = np.argsort(frequency)
        freq_sorted = frequency[freq_argsort]
        kappa_cum_wrt_freq = np.cumsum(kappa_freq[freq_argsort])

        kappa_matrix = kappa_tensor.sum(axis=0).sum(axis=0)
        kappa_total = np.mean(np.diag(kappa_matrix))

        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        self._set_fig_properties([ax])
        plt.plot(freq_sorted, kappa_cum_wrt_freq, c='#E24A33',
                 label=r'$\kappa \approx %.0f$ $\frac{\rm{W}}{\rm{m}\cdot\rm{K}}$' % kappa_total)
        plt.ylabel(r'$\kappa_{cumulative, \omega}$ $\left(\frac{\rm{W}}{\rm{m}\cdot\rm{K}}\right)$')
        plt.xlabel('Frequency (THz)')
        plt.legend(loc='best')
        plt.savefig(folder + '/kappa_cumulative_freq.png', dpi=300, bbox_inches='tight')
        if is_showing:
            plt.show()
        else:
            plt.close()

        # Cumulative Conductivity vs Mean Free Path
        # Compute cumulative conductivity vs mean free path
        kappa_mfp = np.einsum('maa->m', 1/3 * kappa_tensor.reshape(-1, 3, 3))
        lambda_argsort = np.argsort(mean_free_path_norm)
        lambda_sorted = mean_free_path_norm[lambda_argsort]
        kappa_cum_wrt_lambda = np.cumsum(kappa_mfp[lambda_argsort])

        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        self._set_fig_properties([ax])
        plt.plot(lambda_sorted, kappa_cum_wrt_lambda, c='#E24A33')
        plt.xlabel(r'$\lambda$ (nm)')
        plt.ylabel(r'$\kappa_{cumulative, \lambda}$ $\left(\frac{\rm{W}}{\rm{m}\cdot\rm{K}}\right)$')
        plt.xscale('log')
        plt.savefig(folder + '/kappa_cumulative_mfp.png', dpi=300, bbox_inches='tight')
        if is_showing:
            plt.show()
        else:
            plt.close()

        logging.info(f"Plots saved to {folder} (κ = {kappa_total:.1f} W/m·K)")

    def plot_amorphous(self, p_atoms_list=None, p_atoms_labels=None, bandwidth=.05, n_points=200,
                       is_showing=True, method='qhgk', figsize=(8, 6)):
        """Create comprehensive plots for amorphous phonon properties.

        Generates a complete set of publication-quality plots including:
        - Density of states (DOS)
        - Heat capacity vs frequency
        - Diffusivity vs frequency
        - Phase space vs frequency
        - Participation ratio vs frequency
        - Lifetime vs frequency
        - Scattering rate vs frequency
        - Mean free path vs frequency (if available for method)
        - Per-mode thermal conductivity
        - Cumulative thermal conductivity vs frequency
        - Cumulative thermal conductivity vs mean free path (if available)

        Parameters
        ----------
        p_atoms_list : list of lists, optional
            List of atom index sets for projected DOS. If None, automatically groups 
            by chemical species for multi-element systems, or uses all atoms for 
            single-element systems.
        p_atoms_labels : list of str, optional
            Labels for each atom set in p_atoms_list for DOS legend. If None, uses 
            chemical symbols for multi-element systems.
        bandwidth : float
            Gaussian smearing width for DOS calculation. Default: 0.05
            Units: THz
        n_points : int
            Number of frequency points for DOS. Default: 200
        is_showing : bool
            Whether to display plots interactively. Default: True
        method : str
            Method for conductivity calculation. Default: 'qhgk' (appropriate for amorphous)
        figsize : tuple
            Figure size (width, height) in inches. Default: (8, 6) for publication.
            Use (8, 6) for larger presentations.
        """
        folder = self.phonons.get_folder_from_label(base_folder=DEFAULT_FOLDER)
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Load kaldo style guide if available
        style_file = os.path.join(os.path.dirname(__file__), 'kaldo_style_guide.mpl')
        if os.path.exists(style_file):
            plt.style.use(style_file)
        
        # Additional style settings
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['mathtext.fontset'] = 'cm'

        # Calculate conductivity for mean free path, diffusivity, and kappa plots
        # For QHGK method, enable diffusivity calculation by passing bandwidth
        if method == 'qhgk':
            conductivity = Conductivity(phonons=self.phonons, method=method, storage='memory',
                                      diffusivity_bandwidth=self.phonons.bandwidth)
        else:
            conductivity = Conductivity(phonons=self.phonons, method=method, storage='memory')

        # Get flattened data
        physical_mode = self.phonons.physical_mode.flatten()
        frequency = self.phonons.frequency.flatten()

        # Calculate conductivity first to ensure diffusivity is available
        kappa_tensor = conductivity.conductivity.reshape(self.phonons.n_k_points, self.phonons.n_modes, 3, 3)
        kappa_per_mode = kappa_tensor.sum(axis=-1).sum(axis=-1).flatten()
        
        # Now diffusivity should be available
        diffusivity_data = conductivity.diffusivity
        if diffusivity_data is not None:
            diffusivity = diffusivity_data.flatten(order='C')
        else:
            diffusivity = None

        # DOS
        dos_data, p_atoms_labels = self._calculate_dos(p_atoms_list, p_atoms_labels,
                                                        direction=None, bandwidth=bandwidth,
                                                        n_points=n_points)

        if dos_data:
            fig = plt.figure(figsize=figsize)
            ax = fig.gca()
            self._set_fig_properties([ax])
            colors = ['#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E', '#8EBA42']
            for idx, (fgrid, pdos) in enumerate(dos_data):
                color = colors[idx % len(colors)]
                label = p_atoms_labels[idx] if idx < len(p_atoms_labels) else None
                for p in np.expand_dims(pdos, 0) if pdos.ndim == 1 else pdos:
                    plt.plot(fgrid, p, c=color, label=label)
                    label = None
            plt.xlabel("Frequency (THz)")
            plt.ylabel('DOS')
            if p_atoms_labels is not None and len(dos_data) > 1:
                plt.legend(loc='best')
            plt.savefig(folder + '/dos.png', dpi=300, bbox_inches='tight')
            if is_showing:
                plt.show()
            else:
                plt.close()

        # Heat Capacity
        heat_capacity = self.phonons.heat_capacity.flatten()

        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        self._set_fig_properties([ax])
        plt.scatter(frequency[physical_mode], 1e23*heat_capacity[physical_mode],
                    facecolor='w', edgecolor='#E24A33', s=10, marker='8')
        plt.ylabel(r"$C_{v}$ ($10^{23}$ J/K)")
        plt.xlabel('Frequency (THz)')
        y_data = 1e23*heat_capacity[physical_mode]
        y_min, y_max = y_data.min(), y_data.max()
        plt.ylim(0.9*y_min, 1.05*y_max)
        plt.savefig(folder + '/cv.png', dpi=300, bbox_inches='tight')
        if is_showing:
            plt.show()
        else:
            plt.close()

        # Diffusivity
        if diffusivity is not None:
            fig = plt.figure(figsize=figsize)
            ax = fig.gca()
            self._set_fig_properties([ax])
            plt.scatter(frequency[3:], diffusivity[3:], s=5, c='#E24A33')
            plt.xlabel(r'$\nu$ (THz)')
            plt.ylabel(r'$D$ (mm$^2$/s)')
            plt.xlim([0, 25])
            plt.savefig(folder + '/diffusivity.png', dpi=300, bbox_inches='tight')
            if is_showing:
                plt.show()
            else:
                plt.close()

        # Phase Space
        phase_space = self.phonons.phase_space.flatten()

        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        self._set_fig_properties([ax])
        plt.scatter(frequency[physical_mode], phase_space[physical_mode],
                    facecolor='w', edgecolor='#E24A33', s=10, marker='o')
        plt.xlabel('Frequency (THz)')
        plt.ylabel('Phase space')
        plt.savefig(folder + '/phasespace.png', dpi=300, bbox_inches='tight')
        if is_showing:
            plt.show()
        else:
            plt.close()

        # Participation Ratio
        try:
            participation_ratio = self.phonons.participation_ratio.flatten()
            fig = plt.figure(figsize=figsize)
            ax = fig.gca()
            self._set_fig_properties([ax])
            plt.scatter(frequency[physical_mode], participation_ratio[physical_mode],
                        facecolor='w', edgecolor='#E24A33', s=10, marker='o')
            plt.ylabel('Participation ratio')
            plt.xlabel('Frequency (THz)')
            plt.ylim(0, 1.05)
            plt.savefig(folder + '/participation_ratio.png', dpi=300, bbox_inches='tight')
            if is_showing:
                plt.show()
            else:
                plt.close()
        except Exception as e:
            logging.warning(f"Failed to plot participation ratio: {e}")

        # Lifetime
        scattering_rate = self.phonons.bandwidth.flatten()
        lifetime = scattering_rate ** (-1)

        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        self._set_fig_properties([ax])
        plt.scatter(frequency[physical_mode], lifetime[physical_mode],
                    facecolor='w', edgecolor='#E24A33', s=10, marker='s')
        plt.yscale('log')
        plt.ylabel(r'$\tau$ (ps)')
        plt.xlabel('Frequency (THz)')
        plt.savefig(folder + '/lifetime.png', dpi=300, bbox_inches='tight')
        if is_showing:
            plt.show()
        else:
            plt.close()

        # Scattering Rate
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        self._set_fig_properties([ax])
        plt.scatter(frequency[physical_mode], scattering_rate[physical_mode],
                    facecolor='w', edgecolor='#E24A33', s=10, marker='d')
        plt.ylabel(r'$\Gamma$ (THz)')
        plt.xlabel('Frequency (THz)')
        plt.savefig(folder + '/gamma.png', dpi=300, bbox_inches='tight')
        if is_showing:
            plt.show()
        else:
            plt.close()

        # Mean Free Path (if available)
        try:
            mean_free_path = conductivity.mean_free_path.reshape(-1, 3) / 10.0
            mean_free_path_norm = np.linalg.norm(mean_free_path, axis=1)
            has_mfp = True
            
            fig = plt.figure(figsize=figsize)
            ax = fig.gca()
            self._set_fig_properties([ax])
            plt.scatter(frequency[physical_mode], mean_free_path_norm[physical_mode],
                        facecolor='w', edgecolor='#E24A33', s=10, marker='8')
            plt.ylabel(r'$\lambda$ (nm)')
            plt.xlabel('Frequency (THz)')
            plt.yscale('log')
            plt.savefig(folder + '/mfp.png', dpi=300, bbox_inches='tight')
            if is_showing:
                plt.show()
            else:
                plt.close()
        except:
            has_mfp = False

        # Per-mode Conductivity
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        self._set_fig_properties([ax])
        plt.scatter(frequency[physical_mode], kappa_per_mode[physical_mode],
                    facecolor='w', edgecolor='#E24A33', s=10, marker='>')
        plt.axhline(y=0, color='k', ls='--', lw=1)
        plt.ylabel(r'$\kappa_{per \ mode}$ $\left(\frac{\rm{W}}{\rm{m}\cdot\rm{K}}\right)$')
        plt.xlabel('Frequency (THz)')
        plt.savefig(folder + '/kappa_per_mode.png', dpi=300, bbox_inches='tight')
        if is_showing:
            plt.show()
        else:
            plt.close()

        # Cumulative Conductivity vs Frequency
        # Compute cumulative conductivity vs frequency
        kappa_freq = np.einsum('maa->m', 1/3 * kappa_tensor.reshape(-1, 3, 3))
        freq_argsort = np.argsort(frequency)
        freq_sorted = frequency[freq_argsort]
        kappa_cum_wrt_freq = np.cumsum(kappa_freq[freq_argsort])

        kappa_matrix = kappa_tensor.sum(axis=0).sum(axis=0)
        kappa_total = np.mean(np.diag(kappa_matrix))

        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        self._set_fig_properties([ax])
        plt.plot(freq_sorted, kappa_cum_wrt_freq, c='#E24A33',
                 label=r'$\kappa \approx %.0f$ $\frac{\rm{W}}{\rm{m}\cdot\rm{K}}$' % kappa_total)
        plt.ylabel(r'$\kappa_{cumulative, \omega}$ $\left(\frac{\rm{W}}{\rm{m}\cdot\rm{K}}\right)$')
        plt.xlabel('Frequency (THz)')
        plt.legend(loc='best')
        plt.savefig(folder + '/kappa_cumulative_freq.png', dpi=300, bbox_inches='tight')
        if is_showing:
            plt.show()
        else:
            plt.close()

        # Cumulative Conductivity vs Mean Free Path (if available)
        if has_mfp:
            # Compute cumulative conductivity vs mean free path
            kappa_mfp = np.einsum('maa->m', 1/3 * kappa_tensor.reshape(-1, 3, 3))
            lambda_argsort = np.argsort(mean_free_path_norm)
            lambda_sorted = mean_free_path_norm[lambda_argsort]
            kappa_cum_wrt_lambda = np.cumsum(kappa_mfp[lambda_argsort])

            fig = plt.figure(figsize=figsize)
            ax = fig.gca()
            self._set_fig_properties([ax])
            plt.plot(lambda_sorted, kappa_cum_wrt_lambda, c='#E24A33')
            plt.xlabel(r'$\lambda$ (nm)')
            plt.ylabel(r'$\kappa_{cumulative, \lambda}$ $\left(\frac{\rm{W}}{\rm{m}\cdot\rm{K}}\right)$')
            plt.xscale('log')
            plt.savefig(folder + '/kappa_cumulative_mfp.png', dpi=300, bbox_inches='tight')
            if is_showing:
                plt.show()
            else:
                plt.close()

        logging.info(f"Plots saved to {folder} (κ = {kappa_total:.1f} W/m·K)")


def plot_vs_frequency(phonons, observable, observable_name, is_showing=True):
    """Create scatter plot of observable vs phonon frequency.

    Parameters
    ----------
    phonons : Phonons
        Phonons object
    observable : ndarray
        Observable to plot
    observable_name : str
        Name of observable for axis label and filename
    is_showing : bool
        Whether to display the plot. Default: True
    """
    plotter = _Plotter(phonons)
    plotter.plot_vs_frequency(observable, observable_name, is_showing)


def plot_dos(phonons, p_atoms=None, p_atoms_labels=None, direction=None, bandwidth=.05, n_points=200,
             is_showing=True, filename='dos', figsize=(8, 6)):
    """Produce a plot of phonon density of states (DOS) or projected phonon DOS (PDOS).

    Parameters
    ----------
    phonons : Phonons
        Phonons object
    p_atoms : list or list of lists, optional
        Atom indices for projected DOS
    p_atoms_labels : list of str, optional
        Labels for each atom set in p_atoms for DOS legend
    direction : array_like, optional
        3-vector direction for DOS projection
    bandwidth : float
        Gaussian smearing width. Default: 0.05
    n_points : int
        Number of frequency points. Default: 200
    is_showing : bool
        Whether to display the plot. Default: True
    filename : str
        Output filename. Default: 'dos'
    figsize : tuple
        Figure size (width, height) in inches. Default: (8, 6)
    """
    plotter = _Plotter(phonons)
    plotter.plot_dos(p_atoms, p_atoms_labels, direction, bandwidth, n_points, is_showing, filename, figsize)


def plot_dispersion(phonons, p_atoms_list=None, p_atoms_labels=None, bandwidth=.05, n_points=200,
                    n_k_points=300, is_showing=True, symprec=1e-3, with_velocity=True,
                    manually_defined_path=None, folder=None, figsize=(8, 6)):
    """Plot phonon dispersion relation and optionally group velocity with DOS panel.

    Parameters
    ----------
    phonons : Phonons
        Phonons object
    p_atoms_list : list of lists, optional
        List of atom index sets for projected DOS. If None, automatically groups
        by chemical species for multi-element systems.
    p_atoms_labels : list of str, optional
        Labels for each atom set in p_atoms_list for DOS legend
    bandwidth : float
        Gaussian smearing width for DOS calculation. Default: 0.05
    n_points : int
        Number of frequency points for DOS. Default: 200
    n_k_points : int
        Number of k-points along the path. Default: 300
    is_showing : bool
        Whether to display the plot. Default: True
    symprec : float
        Symmetry precision. Default: 1e-3
    with_velocity : bool
        Whether to also plot group velocity. Default: True
    manually_defined_path : ase.dft.kpoints.BandPath, optional
        Manually defined band path
    folder : str, optional
        Output folder
    figsize : tuple
        Figure size (width, height) in inches. Default: (8, 6)
    """
    plotter = _Plotter(phonons)
    plotter.plot_dispersion(p_atoms_list, p_atoms_labels, bandwidth, n_points,
                           n_k_points, is_showing, symprec, with_velocity,
                           manually_defined_path, folder, figsize)


def plot_crystal(phonons, p_atoms_list=None, p_atoms_labels=None, bandwidth=.05, n_points=200,
                 is_showing=True, n_k_points=300, symprec=1e-3, method='inverse', figsize=(8, 6)):
    """Create comprehensive plots for crystal phonon properties.

    Parameters
    ----------
    phonons : Phonons
        Phonons object
    p_atoms_list : list of lists, optional
        List of atom index sets for projected DOS. If None, automatically groups
        by chemical species for multi-element systems.
    p_atoms_labels : list of str, optional
        Labels for each atom set in p_atoms_list for DOS legend. If None, uses
        chemical symbols for multi-element systems.
    bandwidth : float
        Gaussian smearing width. Default: 0.05
    n_points : int
        Number of frequency points for DOS. Default: 200
    is_showing : bool
        Whether to display plots. Default: True
    n_k_points : int
        Number of k-points for dispersion. Default: 300
    symprec : float
        Symmetry precision. Default: 1e-3
    method : str
        Method for conductivity calculation. Default: 'inverse'
    figsize : tuple
        Figure size (width, height) in inches. Default: (8, 6)
    """
    plotter = _Plotter(phonons)
    plotter.plot_crystal(p_atoms_list, p_atoms_labels, bandwidth, n_points, is_showing, n_k_points, symprec, method, figsize)


def plot_amorphous(phonons, p_atoms_list=None, p_atoms_labels=None, bandwidth=.05, n_points=200, is_showing=True,
                   method='qhgk', figsize=(8, 6)):
    """Create comprehensive plots for amorphous phonon properties.

    Parameters
    ----------
    phonons : Phonons
        Phonons object
    p_atoms_list : list of lists, optional
        List of atom index sets for projected DOS. If None, automatically groups
        by chemical species for multi-element systems.
    p_atoms_labels : list of str, optional
        Labels for each atom set in p_atoms_list for DOS legend. If None, uses
        chemical symbols for multi-element systems.
    bandwidth : float
        Gaussian smearing width. Default: 0.05
    n_points : int
        Number of frequency points for DOS. Default: 200
    is_showing : bool
        Whether to display plots. Default: True
    method : str
        Method for conductivity calculation. Default: 'qhgk'
    figsize : tuple
        Figure size (width, height) in inches. Default: (8, 6)
    """
    plotter = _Plotter(phonons)
    plotter.plot_amorphous(p_atoms_list, p_atoms_labels, bandwidth, n_points, is_showing, method, figsize)


# Legacy helper functions for backward compatibility
def convert_to_spg_structure(atoms):
    """Convert ASE atoms to spglib structure format."""
    return _Plotter._convert_to_spg_structure(atoms)


def resample_fourier(observable, increase_factor):
    """Resample observable using Fourier interpolation."""
    return _Plotter._resample_fourier(observable, increase_factor)


def interpolator(k_list, observable, fourier_order=0, interpolation_order=0, is_wrapping=True):
    """Interpolate observable on k-point list."""
    return _Plotter._interpolator(k_list, observable, fourier_order, interpolation_order, is_wrapping)


def create_k_and_symmetry_space(atoms, n_k_points=300, symprec=1e-05, manually_defined_path=None):
    """Create k-point path and symmetry labels for band structure plotting."""
    return _Plotter._create_k_and_symmetry_space(atoms, n_k_points, symprec, manually_defined_path)


def set_fig_properties(ax_list, panel_color_str='black', line_width=2):
    """Apply consistent formatting to matplotlib axes."""
    _Plotter._set_fig_properties(ax_list, panel_color_str, line_width)
