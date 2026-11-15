import pytest

import tritium
from tritium.models import TernaryTransformer, TransformerConfig

CONFIG = TransformerConfig(vocab_size=64, width=64, layers=1, heads=2, hidden=64, context=8)


def test_backends_generate_identical_tokens():
    model = TernaryTransformer(CONFIG)
    prompt = [1, 5, 9]
    outputs = {
        backend: model.to("tritium", backend=backend).generate(prompt, 4, temperature=0.8, seed=3)
        for backend in ("reference", "simulator", "rtl")
    }
    assert outputs["reference"] == outputs["simulator"] == outputs["rtl"]
    assert len(outputs["reference"]) == 4


def test_generation_stops_at_context():
    device_model = TernaryTransformer(CONFIG).to("tritium", backend="reference")
    assert len(device_model.generate([1, 2, 3], 50)) == CONFIG.context - 3 + 1


def test_rtl_backend_reports_cycles():
    device_model = TernaryTransformer(CONFIG).to("tritium", backend="rtl")
    device_model.decode(7)
    statistics = device_model.statistics()
    assert statistics["tokens"] == 1 and statistics["cycles"] > 0


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError):
        TernaryTransformer(CONFIG).to("tritium", backend="gpu")


def test_unsupported_shape_is_rejected():
    config = TransformerConfig(vocab_size=64, width=96, layers=1, heads=2, hidden=64, context=8)
    with pytest.raises(ValueError):
        TernaryTransformer(config).to("tritium")


def test_cpu_device_returns_the_model():
    model = TernaryTransformer(CONFIG)
    assert model.to("cpu") is model
    assert isinstance(model.to("tritium", backend="reference"), tritium.DeviceModel)
