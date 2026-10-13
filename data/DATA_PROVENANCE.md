# Data sources and processing

| Asset | Source/version and use | Processing and split | Permission/acquisition record |
|---|---|---|---|
| Historical123 local sequence labels/features | Project local CK docking snapshot20261001; historical MLP training |546 deduplicated poses summarized into counts0/1/2(31/90/2); tight threshold−8kcal/mol; RC canonicalization; historical5×5 presence-stratified evaluation, final full fit | Scientific source tables retained; owner confirms team-produced docking data and MIT release without confidentiality restrictions; exact original acquisition timestamp unknown |
| Current comparison | Uniform docking seed104729 label snapshot20261003, same123sequence IDs and cached features | Labels aligned byID; count-stratified5fold outer splits; grouped3fold inner search; train-only transforms/PCA8 | Source and fold records retained; previously analysed library, not independent external testing |
| Parent/Opt and candidate pool | User's existing four-strand designs and saved20261002 candidates |1850 sampled designs+4 comparators; constraints, duplicate/hash exclusions and score/diversity selection;84 selected constructs | Original sequence values preserved; no team identities attached |
| Pretrained HyenaDNA | Public pinned Hugging Face model snapshot | Frozen forward/RC128-vector representations; no pretraining/fine-tuning | Official repository/revision/hash preserved; declaration and license status in notice; exact original acquisition time unknown |
| TabPFN | Prior Labs V2 pretrained model | Saved training contexts/predictions; no gradient fine-tuning | Weight not redistributed; recipient obtains under official terms |
| Virtual screening | Recipient-supplied real library/receptor; empty source templates in data/virtual_screening | Preparation status, completeness, protocol hashes/seed and exclusions retained | No commercial source library is uploaded; recipient supplies provenance/permission |
| Synthetic screening example | Generated toy software fixtures | Two toy ligands×eight boxes with explicitly artificial energies; no biological inference | Only software-test output, excluded from formal candidate CSV |
| Transport | Literature-derived processed numeric data, references/DOI and model design inputs | Fitted summaries/parameter tables; finite-volume forward simulation; no training/test split | Original papers/source workbooks excluded; citations retained; cited numeric observations and derived parameters retain their literature provenance |

Detailed transport sources are in `logs/transport/Data_Sources.csv`, parameter tables and fit summaries. Module manifests pin original weights and input values. `SOURCE_MAP.json` records the handoff's source ancestry without local account paths. Module code/logs and final results do not include personal information.

No clinical/patient-identifying data, API credential or hidden competition test set is supplied. Scientific identifiers and publicly cited authors are distinct from private team identities. Unknown dates, licenses or raw data cannot be reconstructed from a result snapshot and are marked unknown rather than invented.

## Publication authorization

The project owner confirmed that docking labels and candidate sequences were produced by the team and may be published without institutional or partner confidentiality restrictions. The owner selected MIT for original code, skills and original project data on 2026-10-13. Third-party assets retain their own notices.
