# Third-party sources and licenses

Public upstream attribution below identifies software/model providers, not the project's private team, institution or accounts.

**HyenaDNA**: Nguyen et al., *HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution* (2023), https://arxiv.org/abs/2306.15794 . Official pinned Hugging Face snapshot: https://huggingface.co/LongSafari/hyenadna-tiny-1k-seqlen-hf/tree/e8c1effa8673814e257e627d2e1eda9ea5a373f6 . The snapshot's model card declares `bsd-3-clause`; the packaged manifest preserves repository/revision/file hashes. Its historical snapshot has no separate LICENSE or copyright header. `licenses/BSD-3-Clause-reference.txt` is the standard license reference, not an invented copyright notice. Consult the pinned upstream model and retain applicable original notices when redistributing. The separate HazyResearch/hyena-dna GitHub repository currently carries Apache-2.0; do not silently substitute that repository license for the pinned Hugging Face model declaration.

**TabPFN**: Prior Labs, https://github.com/PriorLabs/TabPFN . Built with PriorLabs-TabPFN. Version9.0.0 code and V2 pretrained weights have distinct terms. The V2 checkpoint is intentionally not redistributed in this package; obtain it from the official provider and comply with its model license. Saved predictions and training contexts are retained for review.

**Libraries**: NumPy, SciPy, scikit-learn, joblib, PyTorch, Transformers, tokenizers, safetensors, einops, threadpoolctl and matplotlib are required as listed, not vendored. Their installed distributions carry their respective licenses.

Original project-specific scripts, documentation, and owner-produced docking labels and DNA sequence designs are released under the MIT license chosen by the project owner. Third-party code, software packages, and pretrained model assets retain their own terms and attribution described above. No private names, email addresses, account credentials or local project paths are needed by these instructions.
