# Data provenance

| Asset | Source and purpose | Processing/version |
|---|---|---|
| Historical local labels and features | Project-generated local CK docking; 123 unique 20-nt sequences, 546 deduplicated poses summarized into 0/1/2 counts | Historical 20261001 label snapshot; forward/RC canonicalization; tight threshold Vina <= -8 kcal/mol |
| Current benchmark labels | Project uniform single-seed docking snapshot, seed 104729 | Current 20261003; same sequence IDs, fixed five-fold assignments; cached X aligned to current labels rather than silently using historical y |
| Parent and optimized sequences | Existing WT, Opt1, Opt2, Opt3 and sampled pair-preserving candidate pool | Four63-nt strands, six20-bp edges; original values retained; local file metadata removed |
| Historical candidate pool /84 selection | Existing 20261002 generation/scoring output | 1850 designs +4 comparators; top-score/diversity filtering,20 per burden1–4; optional generator retains seed and prior-pool exclusion hashes |
| Frozen HyenaDNA encoder | Official Hugging Face repository `LongSafari/hyenadna-tiny-1k-seqlen-hf` | Pinned revision in `design/checkpoint/source_manifest.json`; not fine-tuned |
| TabPFN V2 | Prior Labs pretrained checkpoint | External weight; original evaluation results/context retained; license/acquisition described separately |

No raw docking preparation, poses, MD trajectories or wet-lab data are required by this sequence-only package. The label tables permit model reproduction, but do not reproduce docking from raw coordinates. No hidden competition evaluation set is included or claimed.

Historical and current label snapshots are kept separately, even where counts coincide. Both describe local sampled docking, not a measured loading dataset. Source table processing preserved scientific columns and removed machine/account metadata. Canonical sequence deduplication precedes splitting; preprocessing is fitted within training folds. Outer folds in current evaluation allow family overlap, so this is within-library development evaluation.

The original project data are provided for the user's requested reuse. No separate signed redistribution license, raw-docking source license or exact acquisition timestamp was present in the inspected records. This document does not invent one. Formal public submission should confirm those permissions. Public pretrained data details should be obtained from upstream model cards; this project did not retrain the HyenaDNA encoder or reproduce its pretraining corpus.
