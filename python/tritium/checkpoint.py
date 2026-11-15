import json
from pathlib import Path

import tritium
from tritium.models import TernaryTransformer, TransformerConfig


def load_run(directory):
    directory = Path(directory)
    checkpoint = json.loads((directory / "checkpoint.json").read_text())
    model = TernaryTransformer(TransformerConfig(**checkpoint["config"]))
    model.load_state_dict(tritium.load(directory / "model.bin"))
    tokenizer = tritium.Tokenizer.load(directory / "tokenizer.txt")
    return model, tokenizer, checkpoint["step"]
