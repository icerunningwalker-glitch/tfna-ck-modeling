# Virtual screening in the unified project

Code is in src/virtual_screening/, templates and synthetic inputs in data/virtual_screening/, the method card in models/virtual_screening/, provenance and verification in logs/virtual_screening/. This module contains no original compound library or scientific docking results.

## Software checks

From the unified project root:

```bash
python -B src/virtual_screening/self_test.py
python -B src/virtual_screening/demo.py --output-dir results/virtual_screening/new_synthetic_demo
```

The demo also works from any current directory: its default output resolves from the script location to project/results/virtual_screening/synthetic_demo. Two synthetic ligands, a toy receptor and deterministic artificial scores test the complete eight-box software path. It does not launch real docking or establish biological findings. A durable synthetic marker, evidence_type=synthetic_software_test and scientific_candidate=False are retained. Demo exports use run/results/synthetic_results.csv, never results.csv, and must be excluded from scientific final candidates. Existing outputs are not overwritten.

## Real input interface

```bash
python -B src/virtual_screening/screen.py --manifest data/virtual_screening/ligands.csv --receptor data/virtual_screening/receptor.pdbqt --vina tools/vina.exe --output-dir results/virtual_screening/run01 --reference-id YOUR_REFERENCE_ID --execute
python -B src/virtual_screening/predict.py --root results/virtual_screening/run01 --reference-id YOUR_REFERENCE_ID
```

Root screen.py can delegate the same arguments. CLI paths resolve from the caller's current directory; absolute paths work too. Relative ligand PDBQT paths inside a manifest always resolve from its directory. Required manifest fields: identifier,name,pdbqt,status. Optional: smiles,cas,formula,molecular_weight,detail. Prepared/reused ligands enter screening; failed/not_prepared records remain exclusions.

Without --execute, files are checked but no outputs or jobs are created. --limit-jobs 2 --execute runs a partial preflight; partial results are not labelled final. Resume the same frozen inputs and parameters; changed protocols are rejected. Valid cached successful jobs are reused. No external binary, installation or download is included.

## Method and output

Preserved defaults: AutoDock Vina 1.1.2, CPU, seed 104729, exhaustiveness 8, 20 modes, energy range 4 kcal/mol, 5 workers and 4 CPU threads per job. Eight equal-sized overlapping 2x2x2 tiles cover receptor heavy-atom bounds with 8 Angstrom margin. These boxes are not independent binding sites or biological domains. Predicted-hit threshold: score <= -8 kcal/mol; lower scores are more favorable. All eight boxes must succeed before a ligand enters final ranking. Missing results are never negative hits.

Primary rank: hit-box fraction descending, best score ascending, mean score ascending, then ID. Secondary rank: best score, mean score, ID. Hit-box count is not measured tFNA loading, affinity or MD stability. The module is conventional docking, not an AI model; it performs no spatial pose deduplication or MD.

Real scientific exports: selected_run/results/results.csv and structures/*.pdb. Fields:

candidate_id,track,smiles,structure_file,rank,best_vina_score_kcal_mol,predicted_hit_box_count_8,predicted_hit_box_fraction_8,model_version,run_version,protocol_sha256,evidence_type,scientific_candidate,notes.

Structure_file is relative to the candidate-table directory. PDB snapshots are first-mode coordinates only; scored PDBQT remains in screen/raw. Coverage, exclusions, quality warnings, attempts, protocol hashes and reference rank are kept separately.

## Dependencies and evidence boundary

Core orchestration, tests and default demo use Python standard library only. Optional chemical preparation needs RDKit 2025.9.6, OpenBabel 2.3.2 and AutoDockTools 4.2.6 with compatible interpreters, all separately supplied. Entry: src/virtual_screening/src/prepare_library.py --help. No actual receptor, real screening or external-tool installation was validated during unification. Optional chemistry tests run only when already-installed RDKit is available.

Logs in the submitted example have runtime paths replaced by anonymous placeholders. New user runs can record local executable/input paths, so inspect a shareable copy separately. Source ancestry and actual software validation are in logs/virtual_screening/. External software/data rights and code-license decisions remain separate; no new license is invented.
