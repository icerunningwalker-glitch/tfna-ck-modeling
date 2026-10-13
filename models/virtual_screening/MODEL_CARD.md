# Method card

Method: AutoDock Vina 1.1.2 single-seed, eight-box virtual screening. No self-trained model or learned weights are used in this isolated module. External Vina executable is supplied by the recipient and recorded by SHA-256 in the run protocol.

Inputs: one prepared receptor PDBQT, prepared ligand PDBQTs and eligibility manifest. Optional SDF preparation uses the separately documented chemistry stack.

Outputs: predicted docking scores, complete-coverage rankings, retained quality warnings and reference position. The framework supports candidate discovery, not quantitative loading prediction or experimental efficacy claims. Single-seed rigid-receptor scoring does not capture full receptor flexibility, solvent/ions or many-cargo competition. Larger/flexible ligands may have score bias and long runtime. No statistical uncertainty is inferred from one seed; sensitivity thresholds are descriptive only.

Contribution of the packaged workflow: immutable protocol and input hashes, parallel execution, validated resume caches, explicit preparation exclusions, complete-coverage ranking and documented exports. Vina and chemistry tools are third-party methods, not claimed as original algorithms. The separate AI-design project requires its own training code, data provenance, model versions, splits, weights and logs to support an AI submission.

Third-party sources: https://vina.scripps.edu/ ; https://github.com/ccsb-scripps/AutoDock-Vina ; https://www.rdkit.org/ ; https://openbabel.org/ ; https://autodock.scripps.edu/ . Licenses and redistribution terms must be checked for each installed version. No external binary is redistributed here.
