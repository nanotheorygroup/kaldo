# Visualize phonon eigenmodes as animated trajectories
#
# This script loads force constants, builds a HarmonicWithQ object at Gamma,
# and writes extended-XYZ trajectory files and HTML viewers for selected
# phonon modes. The .xyz files can be viewed with ASE GUI or OVITO.
# The .html files open in any browser.

from kaldo.forceconstants import ForceConstants
from kaldo.observables.harmonic_with_q import HarmonicWithQ
import kaldo.controllers.plotter as plotter
import numpy as np

# --- Load force constants (adjust path and format for your system) ---
forceconstants = ForceConstants.from_folder(
    folder='fc_folder',
    supercell=np.array([3, 3, 3]),
    format='lammps',
    only_second=True,
)

# --- Create HarmonicWithQ at the Gamma point ---
q_point = np.array([0, 0, 0])
harmonic = HarmonicWithQ(
    q_point=q_point,
    second=forceconstants.second,
)

# --- Print frequencies to pick modes of interest ---
freqs = harmonic.frequency[0]
print("Frequencies (THz):")
for i, f in enumerate(freqs):
    print(f"  mode {i:3d}: {f:.4f}")

# --- Write XYZ trajectory files ---
for mode in [0, 3, 6]:
    fname = f'mode_{mode}_traj.xyz'
    plotter.write_phonon_mode_xyz(harmonic, mode_index=mode, filename=fname)
    print(f"Wrote {fname}")

# --- Generate standalone HTML viewers ---
for mode in [0, 3, 6]:
    html_fname = f'mode_{mode}.html'
    plotter.write_phonon_mode_html(harmonic, mode_index=mode, html_filename=html_fname)
    print(f"Wrote {html_fname}")

# --- Viewing ---
# ASE GUI:   ase gui mode_0_traj.xyz
# OVITO:     ovito mode_0_traj.xyz
# Browser:   open mode_0.html
