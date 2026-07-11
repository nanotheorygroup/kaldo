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

    def _stack(self, observable):
        """Stack the named per-mode property across members: shape (n_members, ...).

        Aggregation is sorted-stack: kaldo returns frequencies sorted ascending per
        q-point, so member axis m tracks the same branch away from band crossings.
        At crossings and in degenerate subspaces the ordering can swap branches,
        which slightly overestimates the std there.
        """
        arrays = []
        for i, member in enumerate(self._members):
            try:
                value = getattr(member, observable)
            except AttributeError as exc:
                raise AttributeError(
                    f"Ensemble member has no property {observable!r}. "
                    "mean_std aggregates scalar per-mode Phonons properties such as "
                    "'frequency', 'heat_capacity', 'participation_ratio', 'bandwidth'."
                ) from exc
            arr = np.asarray(value)
            if i == 0:
                ref_shape = arr.shape
            elif arr.shape != ref_shape:
                raise ValueError(
                    f"Ensemble members disagree on the shape of {observable!r}: "
                    f"member 0 has {ref_shape}, member {i} has {arr.shape}. "
                    "All members must share kpts and system size."
                )
            arrays.append(arr)
        return np.stack(arrays, axis=0)

    def mean(self, observable):
        """Mean of the named per-mode property across members."""
        return self._stack(observable).mean(axis=0)

    def std(self, observable):
        """Standard deviation of the named per-mode property across members."""
        return self._stack(observable).std(axis=0)

    def mean_std(self, observable):
        """Return (mean, std) of the named per-mode property across members."""
        stacked = self._stack(observable)
        return stacked.mean(axis=0), stacked.std(axis=0)
