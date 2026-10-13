"""Frozen HyenaDNA tiny: mean pooling over real bases and RC averaging."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
from common import DATA_ROOT, MODEL_ROOT, RESULTS_ROOT, LOGS_ROOT, ROOT, normalize, reverse_complement


class FrozenHyena:
    def __init__(self, checkpoint=None):
        checkpoint = Path(checkpoint) if checkpoint else MODEL_ROOT / "checkpoint"
        os.environ.setdefault("HF_HOME", str(RESULTS_ROOT / "runs" / "runtime_cache"))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from transformers import AutoModel, AutoTokenizer
        manifest = json.loads((checkpoint / "source_manifest.json").read_text())
        for item in manifest["files"]:
            path = checkpoint / item["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("Checkpoint integrity failure: " + item["file"])
        torch.set_num_threads(2)
        torch.manual_seed(20261001)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
        self.model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True,
                                              torch_dtype=torch.float32).to("cpu").eval()
        self.model.requires_grad_(False)
        self.revision = manifest["revision"]
        self.dimension = self.model.config.d_model
        if self.dimension != 128:
            raise ValueError("Expected the pinned 128-dimensional HyenaDNA tiny model.")

    def token_embeddings(self, sequences):
        sequences = [normalize(s) for s in sequences]
        encoded = self.tokenizer(sequences, add_special_tokens=False, padding=False, return_tensors="pt")
        ids = encoded["input_ids"]
        if tuple(ids.shape) != (len(sequences), 20) or self.torch.any(ids == self.tokenizer.pad_token_id):
            raise ValueError("Expected one real-base token per nucleotide without padding.")
        with self.torch.inference_mode():
            hidden = self.model(input_ids=ids).last_hidden_state
        if tuple(hidden.shape) != (len(sequences), 20, 128):
            raise ValueError("Unexpected embedding shape.")
        return hidden.detach().cpu().numpy()

    def featurize(self, sequences):
        sequences = [normalize(s) for s in sequences]
        forward = self.token_embeddings(sequences)
        reverse = self.token_embeddings([reverse_complement(s) for s in sequences])
        features = (forward.mean(axis=1) + reverse.mean(axis=1)) * np.float32(.5)
        if not np.isfinite(features).all():
            raise ValueError("Nonfinite embedding.")
        return features
