import tritium
from tritium import functional as F
from tritium.models import TernaryTransformer, TransformerConfig

CONFIG = TransformerConfig(vocab_size=300, width=32, layers=2, heads=2, hidden=64, context=16)


def test_model_learns_a_repeating_sequence():
    model = TernaryTransformer(CONFIG)
    tokens = [(index * 7) % 300 for index in range(33)]
    optimizer = tritium.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    losses = []
    for _ in range(40):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(tokens[:32], batch=2), tokens[1:33])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] / 3


def test_state_dict_restores_outputs(tmp_path):
    model = TernaryTransformer(CONFIG)
    tritium.save(model.state_dict(), tmp_path / "model.bin")
    restored = TernaryTransformer(CONFIG)
    for parameter in restored.parameters():
        parameter.copy_(parameter * 0.0)
    restored.load_state_dict(tritium.load(tmp_path / "model.bin"))
    tokens = list(range(10))
    assert model(tokens).tolist() == restored(tokens).tolist()


def test_parameters_have_unique_names():
    names = [name for name, _ in TernaryTransformer(CONFIG).named_parameters()]
    assert len(names) == len(set(names)) == 3 + 6 * CONFIG.layers
    assert "blocks.1.down.weight" in names


def test_tokenizer_round_trips_stories(tmp_path):
    text = "Once upon a time, there was a cat.<|endoftext|>The cat was happy. " * 30
    tokenizer = tritium.Tokenizer.train(text, 320)
    tokens = tokenizer.encode(text[:80])
    assert tokenizer.decode(tokens) == text[:80]
    tokenizer.save(tmp_path / "tokenizer.txt")
    assert tritium.Tokenizer.load(tmp_path / "tokenizer.txt").encode(text) == tokenizer.encode(text)
