import json

import tritium
from tritium.train import main


def test_training_writes_a_resumable_checkpoint(tmp_path):
    stories = "Once upon a time, a small dog found a red ball. The dog was happy."
    text = tmp_path / "stories.txt"
    text.write_text("<|endoftext|>".join([stories] * 200))
    arguments = [
        "--text",
        str(text),
        "--output",
        str(tmp_path / "run"),
        "--vocab-size",
        "300",
        "--width",
        "32",
        "--layers",
        "1",
        "--heads",
        "2",
        "--hidden",
        "64",
        "--context",
        "16",
        "--batch",
        "4",
        "--steps",
        "6",
        "--warmup",
        "2",
        "--log-every",
        "3",
        "--save-every",
        "3",
    ]
    main(arguments)
    checkpoint = json.loads((tmp_path / "run" / "checkpoint.json").read_text())
    assert checkpoint["step"] == 6
    assert "head.weight" in tritium.load(tmp_path / "run" / "model.bin")
    main([*arguments[:-6], "--steps", "8", "--log-every", "4", "--save-every", "4"])
    assert json.loads((tmp_path / "run" / "checkpoint.json").read_text())["step"] == 8
