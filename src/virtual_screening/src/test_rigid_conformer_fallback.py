"""Tests for the fallback's source identity and timeout-sentinel safeguards."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rdkit import Chem
from rdkit.Chem import AllChem, rdCIPLabeler

from rigid_conformer_fallback import (_atom_signature, _graph_smiles,
                                      _validated_geometry, build_rigid_conformer,
                                      conformer_quality_diagnostics)


class RigidFallbackChecks(unittest.TestCase):
    def write_source(self, path, smiles):
        mol = Chem.MolFromSmiles(smiles)
        mol.SetProp("_Name", "test source")
        AllChem.Compute2DCoords(mol)
        writer = Chem.SDWriter(str(path))
        writer.write(mol)
        writer.close()

    def test_timeout_sentinel_is_not_optimized_or_written(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = (Path(directory) / name for name in ("source.sdf", "result.sdf"))
            self.write_source(source, "C[C@H](O)F")
            with patch("rdkit.Chem.AllChem.EmbedMultipleConfs", return_value=[-1]):
                with self.assertRaisesRegex(ValueError, "No converged fallback conformer"):
                    build_rigid_conformer(source, output)
            self.assertFalse(output.exists())

    def test_mirrored_geometry_and_changed_graph_are_rejected(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H](O)F"))
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        rdCIPLabeler.AssignCIPLabels(mol)
        expected_cip = {atom.GetIdx(): atom.GetProp("_CIPCode") for atom in mol.GetAtoms()
                        if atom.HasProp("_CIPCode")}
        canonical = Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
        graph, signature = _graph_smiles(mol), _atom_signature(mol)
        cid = AllChem.EmbedMolecule(mol, randomSeed=42)
        self.assertGreaterEqual(cid, 0)
        self.assertIsNotNone(_validated_geometry(mol, cid, canonical, expected_cip, graph, signature))
        mirrored = Chem.Mol(mol)
        conformer = mirrored.GetConformer(cid)
        for index in range(mirrored.GetNumAtoms()):
            point = conformer.GetAtomPosition(index)
            conformer.SetAtomPosition(index, (-point.x, point.y, point.z))
        self.assertIsNone(_validated_geometry(mirrored, cid, canonical, expected_cip, graph, signature))
        self.assertIsNone(_validated_geometry(mol, cid, canonical, expected_cip, graph + "C", signature))

    def test_artemisinin_is_accepted_only_with_exact_source_stereo(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = (Path(directory) / name for name in ("source.sdf", "result.sdf"))
            smiles = "C[C@@H]1CC[C@H]2[C@@H](C)C(=O)O[C@@H]3O[C@]4(C)CC[C@@H]1[C@]32OO4"
            self.write_source(source, smiles)
            audit = build_rigid_conformer(source, output)
            self.assertTrue(audit["fallback_acceptance_exact_source_stereo"])
            self.assertTrue(audit["complete_explicit_hydrogen_graph_preserved"])
            result = Chem.SDMolSupplier(str(output), removeHs=False)[0]
            Chem.AssignStereochemistryFrom3D(result, replaceExistingTags=True)
            Chem.AssignStereochemistry(result, cleanIt=True, force=True)
            self.assertEqual(Chem.MolToSmiles(Chem.RemoveHs(result), isomericSmiles=True),
                             Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
            self.assertEqual(result.GetNumHeavyAtoms(), 20)
            self.assertEqual(result.GetNumAtoms(), 42)

    def test_geometry_warnings_exclude_short_graph_paths(self):
        mol = Chem.MolFromSmiles("CCCC")
        conformer = Chem.Conformer(4)
        conformer.Set3D(True)
        for index, x in enumerate((0.0, 1.5, 3.0, 4.5)):
            conformer.SetAtomPosition(index, (x, 0.0, 0.0))
        cid = mol.AddConformer(conformer)
        normal = conformer_quality_diagnostics(mol, cid)
        self.assertAlmostEqual(normal["minimum_nonbonded_heavy_distance_A"], 4.5)
        self.assertFalse(normal["geometry_quality_warning"])
        mol.GetConformer(cid).SetAtomPosition(3, (0.5, 0.1, 0.0))
        abnormal = conformer_quality_diagnostics(mol, cid)
        self.assertIn("severe_bond_stretch", abnormal["severe_geometry_flags"])
        self.assertIn("severe_nonbonded_heavy_clash", abnormal["severe_geometry_flags"])


if __name__ == "__main__":
    unittest.main()
