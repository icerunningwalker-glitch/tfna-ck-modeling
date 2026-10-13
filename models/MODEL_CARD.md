# Unified Model Card

Version: `submission_unified_20261004`. This project contains three computational modules with different inputs and output meanings.

| Module | Computational role | Model/version | Input | Output |
|---|---|---|---|---|
| AI sequence optimization | Frozen sequence representation and project-trained MLP; pair-preserving design | HyenaDNA tiny pinned revision; historical MLP20261001; separate comparison20261003 | Four63-nt strands decomposed into six20-nt edges | Local continuous count estimates and their dimensionless sum |
| Virtual screening | Classical Vina docking, no neural-network training | Protocol-specific Vina executable hash/version | Prepared receptor/ligand PDBQT and eight spatial boxes | Predicted energy and full-coverage hit-box rankings; relative PDB references |
| Transport | Mechanistic finite-volume forward model, no trained weights | TH21 reference parameter snapshot | Literature-derived/estimated physical parameters and exposure design | Intact-carrier distribution, exposure, retention and physical checks |

The historical MLP and newer benchmark are separate, not a silently updated single model. The pretrained encoder was frozen; no pretraining or fine-tuning was conducted in this project. Original historical full fitting is provided as a training entry. Current benchmark is library-level fivefold development evaluation; related families can cross outer folds. Inner folds tune parameters and are not an external test set. Only two sequences have count2. No hidden event evaluation set is included.

Candidate generation preserves complementary strands and excludes linkers; mutation burden counts paired sites, twice that many actual bases. The local sum does not resolve global steric competition or simultaneous whole-carrier occupancy. WT and Opt predictions can overlap training sequences. Scores do not establish experimental loading, affinity, efficacy or safety. Individual-prediction intervals have not been independently validated.

Virtual-screening demonstration uses artificial scores solely to exercise software. Synthetic rows are rejected from the unified scientific `results.csv`. Real docking requires recipient-supplied permitted materials/tools. Transport describes the reference carrier and specified loading-equivalent condition; it is not a fitted per-Opt prediction and is not attached to each optimized sequence as measured delivery.

Bundled model assets and detailed module cards are under module-specific folders in `models/`. Fitted ML assets include joblib and a pure numerical NPZ export; only trusted joblib files should be loaded. TabPFN V2 weight is acquired separately under its own terms; its saved evaluation/context records remain available. Physical-model parameters are under `data/transport/` rather than misrepresented as ML weights.

Actual verification environment, runtime, numerical comparisons, hashes and untested operations are recorded in `logs/`. This integration does not retrain, redock or run MD. Original training hardware/runtime and some source-acquisition timestamps were not retained. The owner confirmed public release of original labels and designs and selected MIT. The configured competition track is Track 4: delivery carrier design; third-party terms are recorded separately.
