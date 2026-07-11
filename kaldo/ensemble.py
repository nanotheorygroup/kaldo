"""Ensemble uncertainty quantification for harmonic phonon properties.

Aggregates a scalar per-mode Phonons property (e.g. frequency) across N committee
members into a mean and standard deviation. Model-agnostic: members are ordinary
Phonons objects, however they were produced.
"""
import numpy as np

from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
from kaldo.helpers.logger import get_logger

logging = get_logger()


class PhononsEnsemble:
    """A collection of Phonons members with mean/std aggregation of properties.

    Parameters
    ----------
    phonons_list : list of Phonons
        One Phonons object per committee member. Must be non-empty.
    """

    def __init__(self, phonons_list):
        members = list(phonons_list)
        if len(members) == 0:
            raise ValueError("PhononsEnsemble needs at least one member.")
        self._members = members

    @property
    def members(self):
        """List of Phonons members."""
        return self._members

    @property
    def n_members(self):
        """Number of members in the ensemble."""
        return len(self._members)
