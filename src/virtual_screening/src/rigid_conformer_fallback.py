"""Bounded candidate sampling with strict SDF-derived structure acceptance.

Only candidate generation omits RDKit's chirality constraints. A minimized
candidate is accepted only if its coordinate-derived stereochemistry exactly
matches the source isomeric SMILES and true CIP labels, including the complete
explicit-hydrogen graph. This does not correct or enumerate source structures.
"""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path


def _atom_signature(mol):
    return Counter((atom.GetAtomicNum(), atom.GetIsotope(), atom.GetFormalCharge())
                   for atom in mol.GetAtoms())


def _graph_smiles(mol):
    from rdkit import Chem
    return Chem.MolToSmiles(mol, isomericSmiles=False, allHsExplicit=True)


def _validated_geometry(mol, cid, expected_canonical, expected_cip, expected_graph,
                        expected_atom_signature):
    from rdkit import Chem
    from rdkit.Chem import rdCIPLabeler

    check = Chem.Mol(mol)
    conformer = check.GetConformer(cid)
    if not conformer.Is3D():
        return None
    for index in range(check.GetNumAtoms()):
        point = conformer.GetAtomPosition(index)
        if not all(math.isfinite(value) for value in (point.x, point.y, point.z)):
            return None
    Chem.AssignStereochemistryFrom3D(check, confId=cid, replaceExistingTags=True)
    Chem.AssignStereochemistry(check, cleanIt=True, force=True)
    rdCIPLabeler.AssignCIPLabels(check)
    if any(not check.GetAtomWithIdx(index).HasProp("_CIPCode") or
           check.GetAtomWithIdx(index).GetProp("_CIPCode") != cip
           for index, cip in expected_cip.items()):
        return None
    if Chem.MolToSmiles(Chem.RemoveHs(check), isomericSmiles=True) != expected_canonical:
        return None
    if _graph_smiles(check) != expected_graph or _atom_signature(check) != expected_atom_signature:
        return None
    return check


def conformer_quality_diagnostics(mol, cid=-1):
    """Report gross geometry warnings without changing candidate acceptance.

    Ratios use the RDKit periodic table's covalent radii. They are diagnostic
    heuristics, not strain-energy estimates or a physical validation standard.
    Nonbonded heavy pairs exclude direct neighbours and neighbours two bonds
    apart, including short paths through any element.
    """
    from rdkit import Chem

    conformer = mol.GetConformer(cid)
    periodic_table = Chem.GetPeriodicTable()
    coordinates = [tuple(conformer.GetAtomPosition(index)) for index in range(mol.GetNumAtoms())]
    adjacency = [set() for _ in range(mol.GetNumAtoms())]
    bonded = []
    for bond in mol.GetBonds():
        first, second = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[first].add(second)
        adjacency[second].add(first)
        atoms = [mol.GetAtomWithIdx(index) for index in (first, second)]
        length = math.dist(coordinates[first], coordinates[second])
        radii_sum = sum(periodic_table.GetRcovalent(atom.GetAtomicNum()) for atom in atoms)
        bonded.append({"atom_indices_one_based": [first + 1, second + 1],
                       "elements": [atom.GetSymbol() for atom in atoms],
                       "bond_order": bond.GetBondTypeAsDouble(), "distance_A": length,
                       "covalent_radius_ratio": length / radii_sum})
    heavy_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1]
    nonbonded = []
    for offset, first in enumerate(heavy_indices):
        excluded = {first} | adjacency[first]
        for neighbour in adjacency[first]:
            excluded.update(adjacency[neighbour])
        for second in heavy_indices[offset + 1:]:
            if second in excluded:
                continue
            atoms = [mol.GetAtomWithIdx(index) for index in (first, second)]
            distance = math.dist(coordinates[first], coordinates[second])
            radii_sum = sum(periodic_table.GetRcovalent(atom.GetAtomicNum()) for atom in atoms)
            nonbonded.append({"atom_indices_one_based": [first + 1, second + 1],
                              "elements": [atom.GetSymbol() for atom in atoms],
                              "distance_A": distance, "covalent_radius_ratio": distance / radii_sum})
    longest = max(bonded, key=lambda value: value["distance_A"]) if bonded else None
    stretched = max(bonded, key=lambda value: value["covalent_radius_ratio"]) if bonded else None
    compressed = min(bonded, key=lambda value: value["covalent_radius_ratio"]) if bonded else None
    closest = min(nonbonded, key=lambda value: value["distance_A"]) if nonbonded else None
    closest_ratio = min(nonbonded, key=lambda value: value["covalent_radius_ratio"]) if nonbonded else None
    flags = []
    warnings = []
    if stretched and stretched["covalent_radius_ratio"] > 1.2:
        warnings.append("bond_stretch")
    if closest and closest["distance_A"] < 2.0:
        warnings.append("close_nonbonded_heavy_pair")
    if stretched and stretched["covalent_radius_ratio"] > 1.35:
        flags.append("severe_bond_stretch")
    if compressed and compressed["covalent_radius_ratio"] < 0.65:
        flags.append("severe_bond_compression")
    if closest and (closest["distance_A"] < 1.2 or closest_ratio["covalent_radius_ratio"] < 0.85):
        flags.append("severe_nonbonded_heavy_clash")
    return {"max_bond_length_A": longest["distance_A"] if longest else None,
            "max_bond_covalent_radius_ratio": stretched["covalent_radius_ratio"] if stretched else None,
            "min_bond_covalent_radius_ratio": compressed["covalent_radius_ratio"] if compressed else None,
            "minimum_nonbonded_heavy_distance_A": closest["distance_A"] if closest else None,
            "minimum_nonbonded_heavy_covalent_radius_ratio":
                closest_ratio["covalent_radius_ratio"] if closest_ratio else None,
            "longest_bond": longest, "most_stretched_bond": stretched,
            "most_compressed_bond": compressed, "closest_nonbonded_heavy_pair": closest,
            "geometry_warning_flags": warnings,
            "severe_geometry_flags": flags, "geometry_quality_warning": bool(warnings or flags),
            "warning_thresholds": {"bond_radius_ratio_warning_above": 1.2,
                                   "bond_radius_ratio_severe_above": 1.35,
                                   "bond_radius_ratio_below": 0.65,
                                   "nonbonded_heavy_distance_warning_below_A": 2.0,
                                   "nonbonded_heavy_distance_below_A": 1.2,
                                   "nonbonded_heavy_radius_ratio_below": 0.85},
            "interpretation": "Geometry heuristics only; unchanged structure/stereo acceptance. "
                              "Convergence and identity do not establish a low-strain physical conformer."}


def build_rigid_conformer(source_path, output_path, seed=20260930, timeout_seconds=4):
    """Write one validated, converged conformer or raise without writing output.

    At most three pools are sampled: 32 ordinary candidates, then two pools of
    64 random-coordinate candidates with distinct deterministic seeds. Stop
    after the first pool containing accepted candidates and choose its lowest
    force-field energy. Each embedding pool has a wall-time limit.
    """
    from rdkit import Chem, rdBase
    from rdkit.Chem import AllChem, rdCIPLabeler

    if not isinstance(timeout_seconds, int) or timeout_seconds < 1:
        raise ValueError("Embedding timeout must be a positive integer number of seconds")
    source_text = Path(source_path).read_text(encoding="utf-8-sig")
    blocks = [block for block in source_text.split("$$$$") if block.strip()]
    if len(blocks) != 1:
        raise ValueError("Fallback requires one source SDF structure")
    record = Chem.MolFromMolBlock(blocks[0], removeHs=False)
    if record is None:
        raise ValueError("Fallback could not read source SDF structure")
    source = max(Chem.GetMolFrags(record, asMols=True), key=lambda mol: mol.GetNumHeavyAtoms())
    Chem.AssignStereochemistry(source, cleanIt=True, force=True)
    canonical = Chem.MolToSmiles(Chem.RemoveHs(source), isomericSmiles=True)
    reordered = Chem.MolFromSmiles(canonical)
    if reordered is None or Chem.MolToSmiles(reordered, isomericSmiles=True) != canonical:
        raise ValueError("SDF-derived canonical round trip changed source structure")
    template = Chem.AddHs(reordered)
    original_all_h = Chem.AddHs(source)
    expected_graph = _graph_smiles(original_all_h)
    signature = _atom_signature(original_all_h)
    if _graph_smiles(template) != expected_graph or _atom_signature(template) != signature:
        raise ValueError("Canonical round trip changed complete source hydrogen graph")
    Chem.AssignStereochemistry(template, cleanIt=True, force=True)
    rdCIPLabeler.AssignCIPLabels(template)
    expected_cip = {atom.GetIdx(): atom.GetProp("_CIPCode") for atom in template.GetAtoms()
                    if atom.HasProp("_CIPCode")}
    formal_charge = Chem.GetFormalCharge(template)
    if AllChem.MMFFHasAllMoleculeParams(template):
        forcefield_name = "MMFF94s"
    elif AllChem.UFFHasAllMoleculeParams(template):
        forcefield_name = "UFF"
    else:
        raise ValueError("Fallback lacks complete MMFF/UFF parameters")

    pools = []
    selected = None
    total_sampled = 0
    for count, random_coords, pool_seed in ((32, False, seed + 1),
                                            (64, True, seed + 2),
                                            (64, True, seed + 3)):
        mol = Chem.Mol(template)
        params = AllChem.ETKDGv3()
        params.randomSeed = pool_seed
        params.enforceChirality = False
        params.numThreads = 1
        params.maxIterations = 200
        params.timeout = timeout_seconds
        params.useRandomCoords = random_coords
        returned = list(AllChem.EmbedMultipleConfs(mol, numConfs=count, params=params))
        actual_ids = {conf.GetId() for conf in mol.GetConformers()}
        # RDKit can return [-1] on timeout; it is not a conformer.
        cids = [cid for cid in returned if cid >= 0 and cid in actual_ids]
        total_sampled += len(cids)
        accepted = []
        converged_count = 0
        properties = (AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
                      if forcefield_name == "MMFF94s" and cids else None)
        for cid in cids:
            forcefield = (AllChem.MMFFGetMoleculeForceField(mol, properties, confId=cid)
                          if forcefield_name == "MMFF94s" else
                          AllChem.UFFGetMoleculeForceField(mol, confId=cid))
            if forcefield is None:
                continue
            flag = forcefield.Minimize(maxIts=1000)
            energy = forcefield.CalcEnergy()
            if flag != 0 or not math.isfinite(energy):
                continue
            converged_count += 1
            checked = _validated_geometry(mol, cid, canonical, expected_cip,
                                           expected_graph, signature)
            if checked is not None:
                accepted.append((energy, cid, checked))
        pools.append({"seed": pool_seed, "requested_conformers": count,
                      "random_coordinates": random_coords,
                      "embedding_timeout_seconds": timeout_seconds,
                      "actual_conformers": len(cids), "converged_conformers": converged_count,
                      "strictly_valid_conformers": len(accepted)})
        if accepted:
            selected = min(accepted, key=lambda value: (value[0], value[1]))
            break
    if selected is None:
        raise ValueError("No converged fallback conformer preserves exact source graph and stereochemistry")
    energy, cid, checked = selected
    # Retain catalog annotations without using vendor SMILES as structure input.
    for name in record.GetPropNames(includePrivate=False, includeComputed=False):
        checked.SetProp(name, record.GetProp(name))
    if record.HasProp("_Name"):
        checked.SetProp("_Name", record.GetProp("_Name"))
    writer = Chem.SDWriter(str(output_path))
    try:
        writer.write(checked, confId=cid)
    finally:
        writer.close()
    return {"rdkit_version": rdBase.rdkitVersion,
            "embedding": "ETKDGv3 bounded sampling with strict source postvalidation",
            "fallback_generation_enforce_chirality": False,
            "fallback_acceptance_exact_source_stereo": True,
            "canonical_source": "SDF mol block; canonical round trip verified before embedding",
            "random_seed": seed, "embedding_seeds": [pool["seed"] for pool in pools],
            "sampled_conformers": total_sampled,
            "stereochemically_valid_conformers": pools[-1]["strictly_valid_conformers"],
            "fallback_pools": pools, "forcefield": forcefield_name,
            "energy_kcal_mol": energy, "minimization_converged": True,
            "source_rdkit_smiles": canonical, "generated_rdkit_smiles": canonical,
            "specified_stereocenters": len(expected_cip), "cip_algorithm": "rdCIPLabeler",
            "defined_stereochemistry_preserved": True,
            "complete_explicit_hydrogen_graph_preserved": True,
            "conformer_quality": conformer_quality_diagnostics(checked, cid),
            "formal_charge": formal_charge,
            "heavy_atoms": template.GetNumHeavyAtoms(), "all_atoms": template.GetNumAtoms()}
