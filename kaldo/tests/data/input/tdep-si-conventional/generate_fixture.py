"""Regenerate the compact non-diagonal TDEP IFC2/IFC3 test fixture.

The structure is silicon in the diamond primitive cell.  The force constants
come from an analytic nearest-neighbour bond model, not from a TDEP fit or a
DFT calculation.  This makes the translation gauge, acoustic sum rules, and
expected tensor values exact and independently inspectable.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import ase.io
import numpy as np
from ase.build import bulk, make_supercell

ROOT = Path(__file__).resolve().parent
SUPERCELL_MATRIX = np.array([[1, -1, 1], [1, 1, -1], [-1, 1, 1]], dtype=int)
BOND_TRANSLATIONS = np.array([[0, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=int)
HARMONIC_STIFFNESS = 5.0  # eV / angstrom**2
CUBIC_STIFFNESS = 1.0  # eV / angstrom**3


def _key(atom_2, translation_2, atom_3=None, translation_3=None):
    """Return one sortable IFC record key made only of Python integers."""
    values = [atom_2, *translation_2]
    if atom_3 is not None:
        values.extend([atom_3, *translation_3])
    return tuple(int(value) for value in values)


def _bond_tensors(primitive):
    """Build exact IFC2/IFC3 blocks from translationally invariant bonds."""
    cell = np.asarray(primitive.cell)
    positions = np.asarray(primitive.positions)
    identity = np.eye(3)
    ifc2 = [defaultdict(lambda: np.zeros((3, 3))) for _ in primitive]
    ifc3 = [defaultdict(lambda: np.zeros((3, 3, 3))) for _ in primitive]

    for translation in BOND_TRANSLATIONS:
        displacement = positions[1] + translation @ cell - positions[0]
        direction = displacement / np.linalg.norm(displacement)
        cubic = CUBIC_STIFFNESS * np.einsum(
            "a,b,c->abc", direction, direction, direction
        )

        # E2 = k |u_B - u_A|^2 / 2.  Off-site blocks are -k I and
        # each on-site block is their negative sum.
        ifc2[0][_key(0, (0, 0, 0))] += HARMONIC_STIFFNESS * identity
        ifc2[0][_key(1, translation)] -= HARMONIC_STIFFNESS * identity
        ifc2[1][_key(1, (0, 0, 0))] += HARMONIC_STIFFNESS * identity
        ifc2[1][_key(0, -translation)] -= HARMONIC_STIFFNESS * identity

        # E3 = g [e . (u_B - u_A)]^3 / 6.  The eight sign products of
        # derivatives with respect to A/B are emitted in a central-cell
        # gauge.  Translating the B-centred records changes R to -R.
        zero = (0, 0, 0)
        ifc3[0][_key(0, zero, 0, zero)] -= cubic
        ifc3[0][_key(0, zero, 1, translation)] += cubic
        ifc3[0][_key(1, translation, 0, zero)] += cubic
        ifc3[0][_key(1, translation, 1, translation)] -= cubic

        opposite = -translation
        ifc3[1][_key(1, zero, 1, zero)] += cubic
        ifc3[1][_key(1, zero, 0, opposite)] -= cubic
        ifc3[1][_key(0, opposite, 1, zero)] -= cubic
        ifc3[1][_key(0, opposite, 0, opposite)] += cubic

    return ifc2, ifc3


def _write_ifc2(path, records):
    """Write analytic harmonic blocks using the documented TDEP layout."""
    lines = [str(len(records)), "2.351258971274751"]
    for central_records in records:
        lines.append(str(len(central_records)))
        for key, tensor in sorted(central_records.items()):
            atom_2, *translation = key
            lines.extend(
                [
                    str(atom_2 + 1),
                    " ".join(str(value) for value in translation),
                    *(" ".join(f"{value:.16e}" for value in row) for row in tensor),
                ]
            )
    path.write_text("\n".join(lines) + "\n")


def _write_ifc3(path, records):
    """Write analytic cubic blocks using the documented TDEP layout."""
    lines = [str(len(records)), "2.351258971274751"]
    for central, central_records in enumerate(records):
        lines.append(str(len(central_records)))
        for key, tensor in sorted(central_records.items()):
            atom_2 = key[0]
            translation_2 = key[1:4]
            atom_3 = key[4]
            translation_3 = key[5:8]
            lines.extend(
                [
                    str(central + 1),
                    str(atom_2 + 1),
                    str(atom_3 + 1),
                    "0 0 0",
                    " ".join(str(value) for value in translation_2),
                    " ".join(str(value) for value in translation_3),
                ]
            )
            flat = tensor.reshape(-1)
            lines.extend(
                " ".join(f"{value:.16e}" for value in flat[start : start + 9])
                for start in range(0, 27, 9)
            )
    path.write_text("\n".join(lines) + "\n")


def _validate_sum_rules(ifc2, ifc3):
    """Fail generation unless every analytic translation sum is exact."""
    for central_records in ifc2:
        np.testing.assert_allclose(sum(central_records.values()), 0.0, atol=1e-14)
    for central_records in ifc3:
        by_second = defaultdict(lambda: np.zeros((3, 3, 3)))
        by_third = defaultdict(lambda: np.zeros((3, 3, 3)))
        for key, tensor in central_records.items():
            atom_2 = key[0]
            translation_2 = key[1:4]
            atom_3 = key[4]
            translation_3 = key[5:8]
            by_third[(atom_3, *translation_3)] += tensor
            by_second[(atom_2, *translation_2)] += tensor
        np.testing.assert_allclose(sum(by_second.values()), 0.0, atol=1e-14)
        np.testing.assert_allclose(sum(by_third.values()), 0.0, atol=1e-14)


def main():
    """Write the structures, force constants, and human-readable manifest."""
    primitive = bulk("Si", "diamond", a=5.43)
    supercell = make_supercell(primitive, SUPERCELL_MATRIX)
    ifc2, ifc3 = _bond_tensors(primitive)
    _validate_sum_rules(ifc2, ifc3)

    ase.io.write(ROOT / "infile.ucposcar", primitive, format="vasp")
    ase.io.write(ROOT / "infile.ssposcar", supercell, format="vasp")
    _write_ifc2(ROOT / "infile.forceconstant", ifc2)
    _write_ifc3(ROOT / "infile.forceconstant_thirdorder", ifc3)

    support_set = {
        (0, 0, 0),
        *(tuple(row) for row in BOND_TRANSLATIONS),
        *(tuple(-row) for row in BOND_TRANSLATIONS),
    }
    support = [(0, 0, 0), *sorted(support_set - {(0, 0, 0)})]
    manifest = {
        "description": "Diamond-Si geometry with analytic nearest-neighbour IFC2/IFC3",
        "reference_kind": "exact analytic test model; not a material-property reference",
        "supercell_matrix": SUPERCELL_MATRIX.tolist(),
        "determinant": int(round(abs(np.linalg.det(SUPERCELL_MATRIX)))),
        "primitive_atoms": len(primitive),
        "supercell_atoms": len(supercell),
        "ifc2_records": sum(len(records) for records in ifc2),
        "ifc3_records": sum(len(records) for records in ifc3),
        "translation_support": [list(map(int, translation)) for translation in support],
        "harmonic_stiffness_eV_per_A2": HARMONIC_STIFFNESS,
        "cubic_stiffness_eV_per_A3": CUBIC_STIFFNESS,
        "nearest_neighbor_translations": BOND_TRANSLATIONS.tolist(),
    }
    (ROOT / "expected.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
