"""Verify that MOL2 and PDBQT retain an audited 3D SDF heavy-atom geometry.

This is a conversion check, not a new structure or stereochemistry assignment.
The caller must first validate source-defined stereochemistry in the reference
3D SDF, and separately check the converted molecular connectivity.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
import re


@dataclass(frozen=True)
class HeavyAtom:
    element: str
    xyz: tuple[float, float, float]
    label: str
    atom_type: str | None = None
    formal_charge: int | None = None


def _checked_atom(element, xyz, label, atom_type=None, formal_charge=None):
    position = tuple(float(value) for value in xyz)
    if len(position) != 3 or not all(math.isfinite(value) for value in position):
        raise ValueError(f"Nonfinite or malformed coordinates for {label}")
    return HeavyAtom(element, position, str(label), atom_type, formal_charge)


def _read_reference_graph(path):
    """Read one SDF molecule without deriving stereo from conversion formats."""
    from rdkit import Chem

    blocks = [block for block in Path(path).read_text(encoding="utf-8-sig").split("$$$$")
              if block.strip()]
    if len(blocks) != 1:
        raise ValueError("Reference SDF must contain exactly one readable molecule")
    # An empty molecule title is a valid first header line; preserve it.
    mol = Chem.MolFromMolBlock(blocks[0], removeHs=False)
    if mol is None:
        raise ValueError("Reference SDF must contain exactly one readable molecule")
    if mol.GetNumConformers() != 1:
        raise ValueError("Reference SDF must contain one coordinate conformer")
    conformer = mol.GetConformer()
    atoms = []
    for atom in mol.GetAtoms():
        point = conformer.GetAtomPosition(atom.GetIdx())
        atoms.append(_checked_atom(atom.GetSymbol(), (point.x, point.y, point.z),
                                   f"SDF atom {atom.GetIdx() + 1}",
                                   formal_charge=atom.GetFormalCharge()))
    bonds = {tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))):
             bond.GetBondTypeAsDouble() for bond in mol.GetBonds()}
    return atoms, bonds, Chem.GetFormalCharge(mol)


def read_reference_heavy_atoms(path):
    atoms, _, _ = _read_reference_graph(path)
    return [atom for atom in atoms if atom.element != "H"]


def _read_mol2_graph(path):
    atoms = []
    atom_ids = {}
    bonds_by_id = []
    partial_charges = []
    section = None
    found_block = False
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("@<TRIPOS>"):
            section = line.strip()
            if section == "@<TRIPOS>ATOM":
                if found_block:
                    raise ValueError("MOL2 contains multiple atom blocks")
                found_block = True
            continue
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if section == "@<TRIPOS>BOND":
            if len(fields) < 4:
                raise ValueError("Malformed MOL2 bond row")
            order = {"1": 1.0, "2": 2.0, "3": 3.0, "ar": 1.5, "am": 1.0}.get(fields[3])
            if order is None:
                raise ValueError(f"Unsupported MOL2 bond type {fields[3]!r}")
            bonds_by_id.append((fields[1], fields[2], order))
            continue
        if section != "@<TRIPOS>ATOM":
            continue
        if len(fields) < 6:
            raise ValueError("Malformed MOL2 atom row")
        # Standard SYBYL types identify the element before the dot: C.3,
        # N.am, O.co2, etc. Bare element types such as Cl are also accepted.
        atom_type = fields[5]
        element = atom_type.split(".", 1)[0]
        if not re.fullmatch(r"[A-Z][a-z]?", element):
            raise ValueError(f"Unrecognized MOL2 atom type {atom_type}")
        if fields[0] in atom_ids:
            raise ValueError(f"Duplicate MOL2 atom ID {fields[0]}")
        atom_ids[fields[0]] = len(atoms)
        atoms.append(_checked_atom(element, fields[2:5], f"MOL2 atom {fields[0]}", atom_type=atom_type))
        if len(fields) >= 9:
            charge = float(fields[8])
            if not math.isfinite(charge):
                raise ValueError("Nonfinite MOL2 partial charge")
            partial_charges.append(charge)
    if not found_block:
        raise ValueError("MOL2 has no atom block")
    bonds = {}
    for first, second, order in bonds_by_id:
        if first not in atom_ids or second not in atom_ids:
            raise ValueError("MOL2 bond references an absent atom")
        edge = tuple(sorted((atom_ids[first], atom_ids[second])))
        if edge[0] == edge[1] or edge in bonds:
            raise ValueError("MOL2 has a self bond or duplicate bond")
        bonds[edge] = order
    charge_sum = sum(partial_charges) if len(partial_charges) == len(atoms) else None
    return atoms, bonds, charge_sum


def read_mol2_heavy_atoms(path):
    atoms, _, _ = _read_mol2_graph(path)
    return [atom for atom in atoms if atom.element != "H"]


_PDBQT_ELEMENTS = {
    "C": "C", "A": "C", "N": "N", "NA": "N", "NS": "N",
    "OA": "O", "OS": "O", "O": "O", "S": "S", "SA": "S",
    "P": "P", "F": "F", "Cl": "Cl", "Br": "Br", "I": "I",
    "H": "H", "HD": "H", "HS": "H",
    "Mg": "Mg", "Mn": "Mn", "Zn": "Zn", "Ca": "Ca", "Fe": "Fe",
}


def read_pdbqt_heavy_atoms(path):
    atoms = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atom_type = line[77:].strip()
        if atom_type not in _PDBQT_ELEMENTS:
            raise ValueError(f"Unrecognized PDBQT atom type {atom_type!r}")
        element = _PDBQT_ELEMENTS[atom_type]
        if element == "H":
            continue
        atoms.append(_checked_atom(element, (line[30:38], line[38:46], line[46:54]),
                                   f"PDBQT atom {line[6:11].strip()}"))
    return atoms


def _distance(first, second):
    return math.dist(first.xyz, second.xyz)


def _reject_clashes(atoms, tolerance_A, format_name):
    for atom in atoms:
        if not all(math.isfinite(value) for value in atom.xyz):
            raise ValueError(f"Nonfinite {format_name} coordinates for {atom.label}")
    for index, first in enumerate(atoms):
        for second in atoms[index + 1:]:
            if _distance(first, second) <= tolerance_A:
                raise ValueError(f"{format_name} heavy-atom coordinate clash: "
                                 f"{first.label}, {second.label}")


def _match_geometry(reference, converted, format_name, tolerance_A):
    if not math.isfinite(tolerance_A) or tolerance_A <= 0:
        raise ValueError("Geometry tolerance must be positive and finite")
    if not reference:
        raise ValueError("Reference has no heavy atoms")
    if Counter(atom.element for atom in reference) != Counter(atom.element for atom in converted):
        raise ValueError(f"{format_name} heavy-atom element counts differ from reference")
    _reject_clashes(reference, tolerance_A, "Reference")
    _reject_clashes(converted, tolerance_A, format_name)
    matched = set()
    atom_mapping = {}
    displacements = []
    for reference_index, atom in enumerate(reference):
        candidates = [(index, _distance(atom, other))
                      for index, other in enumerate(converted)
                      if other.element == atom.element and _distance(atom, other) <= tolerance_A]
        if not candidates:
            raise ValueError(f"{format_name} lacks matching geometry for {atom.label}")
        if len(candidates) != 1:
            raise ValueError(f"Ambiguous {format_name} geometry match for {atom.label}")
        index, displacement = candidates[0]
        if index in matched:
            raise ValueError(f"{format_name} atom matched to multiple reference atoms")
        matched.add(index)
        atom_mapping[reference_index] = index
        displacements.append(displacement)
    if len(matched) != len(converted):
        raise ValueError(f"{format_name} has unmatched heavy atoms")
    return {"atoms": len(reference), "maximum_displacement_A": max(displacements),
            "one_to_one_geometry_preserved": True}, atom_mapping


def match_heavy_geometry(reference, converted, format_name, tolerance_A=0.003):
    """Match every heavy atom by element and coordinates, allowing reordering."""
    result, _ = _match_geometry(reference, converted, format_name, tolerance_A)
    result["heavy_atoms"] = result.pop("atoms")
    return result


def audit_mapped_mol2_bonds(reference_atoms, reference_bonds, mol2_atoms, mol2_bonds,
                          atom_mapping, partial_charge_sum):
    """Allow only a charged carboxylate's explicit 1/2 -> ar/ar convention."""
    mapped = {tuple(sorted((atom_mapping[first], atom_mapping[second]))): order
              for (first, second), order in reference_bonds.items()}
    if mapped.keys() != mol2_bonds.keys():
        raise ValueError("MOL2 atom-mapped bond connectivity or orders differ from reference SDF")
    differences = {edge for edge in mapped if mapped[edge] != mol2_bonds[edge]}
    if not differences:
        return []
    reference_by_mol2 = {converted: reference_atoms[source] for source, converted in atom_mapping.items()}
    neighbours = {index: [] for index in range(len(mol2_atoms))}
    for (first, second), order in mapped.items():
        neighbours[first].append((second, order))
        neighbours[second].append((first, order))
    accepted = set()
    exceptions = []
    for carbon, atom in enumerate(mol2_atoms):
        if atom.element != "C" or atom.atom_type != "C.2":
            continue
        adjacent = neighbours[carbon]
        oxygen_bonds = [(index, order) for index, order in adjacent if mol2_atoms[index].element == "O"]
        if len(adjacent) != 3 or len(oxygen_bonds) != 2:
            continue
        if sorted(order for _, order in oxygen_bonds) != [1.0, 2.0]:
            continue
        if reference_by_mol2[carbon].formal_charge != 0:
            continue
        oxygens = [index for index, _ in oxygen_bonds]
        if sorted(reference_by_mol2[index].formal_charge for index in oxygens) != [-1, 0]:
            continue
        # A terminal anionic O has no attached H or other substituent. This
        # deliberately excludes protonated acids, esters, and arbitrary aromatics.
        if any(len(neighbours[index]) != 1 or mol2_atoms[index].atom_type != "O.co2" for index in oxygens):
            continue
        other = [(index, order) for index, order in adjacent if index not in oxygens]
        if len(other) != 1 or other[0][1] != 1.0 or mol2_atoms[other[0][0]].element not in ("C", "H"):
            continue
        edges = {tuple(sorted((carbon, oxygen))) for oxygen in oxygens}
        if not edges <= differences or any(mol2_bonds[edge] != 1.5 for edge in edges):
            continue
        accepted.update(edges)
        exceptions.append({"convention": "terminal_anionic_carboxylate_1_2_to_ar_ar",
                           "carbon_mol2_atom": atom.label,
                           "oxygen_mol2_atoms": [mol2_atoms[index].label for index in oxygens],
                           "source_orders": [order for _, order in oxygen_bonds],
                           "mol2_orders": [1.5, 1.5]})
    formal_charge = sum(atom.formal_charge or 0 for atom in reference_atoms)
    if accepted != differences or partial_charge_sum is None or abs(partial_charge_sum - formal_charge) > 0.05:
        raise ValueError("MOL2 atom-mapped bond connectivity or orders differ from reference SDF")
    return exceptions


def audit_conversion_geometry(reference_3d_sdf, mol2_path, pdbqt_path, tolerance_A=0.003):
    """Check both converted formats against the same source-audited geometry."""
    reference_all, reference_bonds, formal_charge = _read_reference_graph(reference_3d_sdf)
    mol2_all, mol2_bonds, partial_charge_sum = _read_mol2_graph(mol2_path)
    all_atom_result, atom_mapping = _match_geometry(reference_all, mol2_all, "MOL2", tolerance_A)
    resonance_exceptions = audit_mapped_mol2_bonds(reference_all, reference_bonds, mol2_all,
                                                  mol2_bonds, atom_mapping, partial_charge_sum)
    reference = [atom for atom in reference_all if atom.element != "H"]
    mol2 = [atom for atom in mol2_all if atom.element != "H"]
    pdbqt = read_pdbqt_heavy_atoms(pdbqt_path)
    return {"tolerance_A": tolerance_A,
            "mol2": match_heavy_geometry(reference, mol2, "MOL2", tolerance_A),
            "pdbqt": match_heavy_geometry(reference, pdbqt, "PDBQT", tolerance_A),
            "mol2_all_atoms": all_atom_result,
            "hydrogen_count": sum(atom.element == "H" for atom in reference_all),
            "mol2_bonds": len(mol2_bonds), "mol2_atom_mapped_bonds_preserved": True,
            "mol2_bond_order_convention_exceptions": resonance_exceptions,
            "source_formal_charge": formal_charge,
            "mol2_partial_charge_sum": partial_charge_sum,
            "heavy_geometry_preserved_through_conversion": True}
