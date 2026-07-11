"""Ensemble uncertainty quantification for harmonic phonon properties.

Aggregates a scalar per-mode Phonons property (e.g. frequency) across N committee
members into a mean and standard deviation. Model-agnostic: members are ordinary
Phonons objects, however they were produced.
"""
import numpy as np

from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons


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

    @classmethod
    def from_calculators(cls, atoms, supercell, calculators, *,
                         delta_shift=1e-2, symmetrize=True, **phonons_kwargs):
        """Build an ensemble from N independent ASE calculators.

        For each calculator, run a finite-difference second-order calculation,
        optionally symmetrize the force constants (kaldo 2.2.0), and build a
        Phonons member. Suited to a handful of independent committee members;
        get_forces() is called serially per member.

        Parameters
        ----------
        atoms : ase.Atoms
            Unit cell (same for every member).
        supercell : tuple of int, length 3
            Supercell for the finite-difference second-order calculation.
        calculators : list of ASE calculators
            One calculator per committee member. Must be non-empty.
        delta_shift : float, optional
            Finite-difference displacement in Angstrom. Default 1e-2.
        symmetrize : bool, optional
            Project each member's force constants onto the space-group-invariant
            subspace. Default True.
        **phonons_kwargs
            Forwarded to each Phonons (kpts, temperature, is_classic, storage,
            folder, ...).
        """
        calculators = list(calculators)
        if len(calculators) == 0:
            raise ValueError("from_calculators needs at least one calculator.")

        base_folder = phonons_kwargs.get('folder', None)
        members = []
        for i, calc in enumerate(calculators):
            member_folder = f"{base_folder}/member_{i}" if base_folder else f"ensemble_member_{i}"
            fc = ForceConstants(atoms=atoms, supercell=supercell,
                                folder=member_folder)
            fc.second.calculate(calculator=calc, delta_shift=delta_shift,
                                is_storing=False, symmetrize=False)
            # Give each member a distinct Phonons folder as well, so on-disk
            # storage does not have the members clobber each other's output.
            member_kwargs = dict(phonons_kwargs, folder=member_folder)
            members.append(cls._member_from_second(
                atoms, supercell, fc.second, symmetrize=symmetrize,
                phonons_kwargs=member_kwargs))
        return cls(members)

    @staticmethod
    def _member_from_second(atoms, supercell, second_order, symmetrize, phonons_kwargs):
        """Build one Phonons member from a computed SecondOrder.

        If symmetrize is True, projects the force constants onto the
        space-group-invariant subspace (kaldo 2.2.0). Wraps the SecondOrder in a
        ForceConstants and constructs a Phonons with phonons_kwargs.
        """
        if symmetrize:
            second_order.symmetrize()
        fc = ForceConstants(atoms=atoms, supercell=supercell, second_order=second_order)
        return Phonons(forceconstants=fc, **phonons_kwargs)

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
            if not hasattr(type(member), observable):
                raise AttributeError(
                    f"Ensemble member has no property {observable!r}. "
                    "mean_std aggregates scalar per-mode Phonons properties such as "
                    "'frequency', 'heat_capacity', 'participation_ratio', 'bandwidth'."
                )
            arr = np.asarray(getattr(member, observable))
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
