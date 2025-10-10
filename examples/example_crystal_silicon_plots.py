"""
Crystal Silicon - Phonon Property Visualization

Demonstrates plot_crystal() for generating comprehensive phonon property plots.
"""

from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
from kaldo.controllers.plotter import plot_crystal

forceconstants = ForceConstants.from_folder(
    folder="kaldo/tests/si-crystal",
    supercell=[3, 3, 3],
    format="eskm"
)

phonons = Phonons(
    forceconstants=forceconstants,
    kpts=[5, 5, 5],
    is_classic=False,
    temperature=300,
    storage="memory",
)

# plot_crystal(phonons, is_showing=False, figsize=(4, 3))
plot_crystal(phonons, is_showing=False, figsize=(8, 6))
