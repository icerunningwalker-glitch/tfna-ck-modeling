"""Small conversion checks; run after adding the isolated RDKit tools to sys.path."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from rdkit import Chem

from chemistry_geometry_checks import (HeavyAtom, audit_conversion_geometry,
                                       audit_mapped_mol2_bonds, match_heavy_geometry)


class GeometryChecks(unittest.TestCase):
    def test_reordering_and_rounding_are_preserved(self):
        reference = [HeavyAtom("C", (0.1234, 0.2345, 0.3456), "1"),
                     HeavyAtom("C", (1.5, 0.0, 0.0), "2"),
                     HeavyAtom("O", (2.6, 0.7, 0.2), "3")]
        converted = [reference[2], HeavyAtom("C", (0.123, 0.235, 0.346), "a"), reference[1]]
        result = match_heavy_geometry(reference, converted, "mock")
        self.assertTrue(result["one_to_one_geometry_preserved"])
        self.assertLess(result["maximum_displacement_A"], 0.003)

    def test_changed_geometry_is_rejected(self):
        reference = [HeavyAtom("C", (0, 0, 0), "1"), HeavyAtom("O", (1.4, 0, 0), "2")]
        converted = [reference[0], HeavyAtom("O", (1.4, 0, 0.1), "2")]
        with self.assertRaisesRegex(ValueError, "lacks matching geometry"):
            match_heavy_geometry(reference, converted, "mock")

    def test_changed_element_or_missing_atom_is_rejected(self):
        reference = [HeavyAtom("C", (0, 0, 0), "1"), HeavyAtom("O", (1.4, 0, 0), "2")]
        with self.assertRaisesRegex(ValueError, "element counts"):
            match_heavy_geometry(reference, [reference[0]], "mock")
        with self.assertRaisesRegex(ValueError, "element counts"):
            match_heavy_geometry(reference, [reference[0], HeavyAtom("N", (1.4, 0, 0), "2")], "mock")

    def test_coordinate_clash_is_rejected(self):
        reference = [HeavyAtom("C", (0, 0, 0), "1"), HeavyAtom("O", (1.4, 0, 0), "2")]
        converted = [reference[0], HeavyAtom("O", (0, 0, 0), "2")]
        with self.assertRaisesRegex(ValueError, "coordinate clash"):
            match_heavy_geometry(reference, converted, "mock")

    def test_ambiguous_matching_is_rejected(self):
        reference = [HeavyAtom("C", (0, 0, 0), "1"), HeavyAtom("C", (0.004, 0, 0), "2")]
        converted = [HeavyAtom("C", (-0.002, 0, 0), "a"), HeavyAtom("C", (0.002, 0, 0), "b")]
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            match_heavy_geometry(reference, converted, "mock")

    def test_nonfinite_and_invalid_tolerance_are_rejected(self):
        reference = [HeavyAtom("C", (0, 0, 0), "1")]
        for tolerance in (0, -1, float("nan")):
            with self.assertRaisesRegex(ValueError, "tolerance"):
                match_heavy_geometry(reference, reference, "mock", tolerance)

    def test_charged_terminal_carboxylate_convention_is_bounded(self):
        reference = [HeavyAtom("C", (0, 0, 0), "1", formal_charge=0),
                     HeavyAtom("C", (1.5, 0, 0), "2", formal_charge=0),
                     HeavyAtom("O", (2.1, 1.1, 0), "3", formal_charge=0),
                     HeavyAtom("O", (2.1, -1.1, 0), "4", formal_charge=-1)]
        converted = [HeavyAtom(atom.element, atom.xyz, atom.label, atom_type=kind)
                     for atom, kind in zip(reference, ("C.3", "C.2", "O.co2", "O.co2"))]
        source_bonds = {(0, 1): 1.0, (1, 2): 2.0, (1, 3): 1.0}
        converted_bonds = {(0, 1): 1.0, (1, 2): 1.5, (1, 3): 1.5}
        mapping = {index: index for index in range(4)}
        exceptions = audit_mapped_mol2_bonds(reference, source_bonds, converted,
                                           converted_bonds, mapping, -1.0005)
        self.assertEqual(len(exceptions), 1)
        # Neutralized charge, untyped oxygen, or a protonated acid cannot use it.
        with self.assertRaisesRegex(ValueError, "bond connectivity or orders"):
            audit_mapped_mol2_bonds(reference, source_bonds, converted, converted_bonds, mapping, 0.0)
        untyped = converted.copy()
        untyped[3] = HeavyAtom("O", reference[3].xyz, "4", atom_type="O.3")
        with self.assertRaisesRegex(ValueError, "bond connectivity or orders"):
            audit_mapped_mol2_bonds(reference, source_bonds, untyped, converted_bonds, mapping, -1.0)
        neutral_reference = reference.copy()
        neutral_reference[3] = HeavyAtom("O", reference[3].xyz, "4", formal_charge=0)
        with self.assertRaisesRegex(ValueError, "bond connectivity or orders"):
            audit_mapped_mol2_bonds(neutral_reference, source_bonds, converted, converted_bonds, mapping, 0.0)

    def test_rdkit_sdf_mol2_pdbqt_roundtrip_with_planar_geometry(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            sdf, mol2, pdbqt = (folder / name for name in ("reference.sdf", "ligand.mol2", "ligand.pdbqt"))
            mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
            coordinates = [(0.1234, 0.2345, 0.0), (1.5234, 0.2345, 0.0), (2.6234, 0.9345, 0.0)]
            coordinates.extend((index * 0.33, 2 + index * 0.1, 1 + index * 0.23)
                               for index in range(3, mol.GetNumAtoms()))
            conformer = Chem.Conformer(mol.GetNumAtoms())
            conformer.Set3D(True)
            for index, xyz in enumerate(coordinates):
                conformer.SetAtomPosition(index, xyz)
            mol.AddConformer(conformer)
            writer = Chem.SDWriter(str(sdf))
            writer.write(mol)
            writer.close()
            # Permute the conversion order and include a hydrogen that is removed
            # by PDBQT preparation. All reference heavy positions remain unchanged.
            rows = []
            pdbqt_rows = ["ROOT"]
            converted_order = (2, 0, 1, *range(3, mol.GetNumAtoms()))
            source_to_serial = {index: serial for serial, index in enumerate(converted_order, 1)}
            for serial, index in enumerate(converted_order, 1):
                x, y, z = coordinates[index]
                element = mol.GetAtomWithIdx(index).GetSymbol()
                sybyl = "O.3" if element == "O" else "C.3" if element == "C" else "H"
                rows.append(f"{serial} {element}{serial} {x:.4f} {y:.4f} {z:.4f} {sybyl} 1 LIG 0")
                if element == "H":
                    continue
                atom_type = "OA" if element == "O" else "A" if serial == 2 else "C"
                line = f"ATOM  {serial:5d} {element+str(serial):<4} LIG A   1    {x:8.3f}{y:8.3f}{z:8.3f}"
                pdbqt_rows.append(line.ljust(70) + f"{0.0:6.3f} " + atom_type)
            bonds = [f"{index + 1} {source_to_serial[bond.GetBeginAtomIdx()]} "
                     f"{source_to_serial[bond.GetEndAtomIdx()]} 1"
                     for index, bond in enumerate(mol.GetBonds())]
            mol2_text = "@<TRIPOS>MOLECULE\nfixture\n9 8\nSMALL\nNO_CHARGES\n@<TRIPOS>ATOM\n"
            mol2_text += "\n".join(rows) + "\n@<TRIPOS>BOND\n" + "\n".join(bonds) + "\n"
            mol2.write_text(mol2_text, encoding="ascii")
            pdbqt.write_text("\n".join(pdbqt_rows + ["ENDROOT", "TORSDOF 0"]) + "\n", encoding="ascii")
            result = audit_conversion_geometry(sdf, mol2, pdbqt)
            self.assertTrue(result["heavy_geometry_preserved_through_conversion"])
            self.assertEqual(result["pdbqt"]["heavy_atoms"], 3)
            self.assertTrue(result["mol2_atom_mapped_bonds_preserved"])
            self.assertEqual(result["hydrogen_count"], 6)
            self.assertEqual(result["mol2_all_atoms"]["atoms"], 9)
            # Geometry and element counts alone must not permit a bond change.
            altered_bonds = bonds.copy()
            altered_bonds[0] = altered_bonds[0][:-1] + "2"
            mol2.write_text(mol2_text.replace("\n".join(bonds), "\n".join(altered_bonds)), encoding="ascii")
            with self.assertRaisesRegex(ValueError, "bond connectivity or orders"):
                audit_conversion_geometry(sdf, mol2, pdbqt)
            # Nor may a converter silently remove or add an explicit hydrogen.
            mol2.write_text(mol2_text.replace(rows[-1] + "\n", ""), encoding="ascii")
            with self.assertRaisesRegex(ValueError, "absent atom|element counts"):
                audit_conversion_geometry(sdf, mol2, pdbqt)


if __name__ == "__main__":
    unittest.main()
