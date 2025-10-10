"""
Amorphous Silicon - Phonon Property Visualization

Demonstrates plot_amorphous() for generating comprehensive phonon property plots.
"""

from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
from kaldo.controllers.plotter import plot_amorphous

forceconstants = ForceConstants.from_folder(
    folder="kaldo/tests/si-amorphous",
    format="eskm"
)

phonons = Phonons(
    forceconstants=forceconstants,
    is_classic=False,
    temperature=300,
    third_bandwidth=0.05 / 4.135,
    broadening_shape="triangle",
    storage="memory",
)

# plot_amorphous(phonons, is_showing=False, figsize=(4, 3))
plot_amorphous(phonons, is_showing=False, figsize=(8, 6))

