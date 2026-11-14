import argparse
import json
import math
import time
from pathlib import Path

import tritium
from tritium import functional as F
from tritium.models import TernaryTransformer, TransformerConfig


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(prog="python -m tritium.train")
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--tokenizer-bytes", type=int, default=20_000_000)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(arguments)


def prepare_tokens(arguments):
    arguments.output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = arguments.output / "tokenizer.txt"
    if tokenizer_path.exists():
        tokenizer = tritium.Tokenizer.load(tokenizer_path)
    else:
        with arguments.text.open(encoding="utf-8", errors="replace") as file:
            sample = file.read(arguments.tokenizer_bytes)
        tokenizer = tritium.Tokenizer.train(sample, arguments.vocab_size)
        tokenizer.save(tokenizer_path)
    datasets = {}
    for name, path in (("train", arguments.text), ("validation", arguments.validation)):
        if path is None:
            continue
        token_path = arguments.output / f"{name}.tokens"
        if not token_path.exists():
            count = tokenizer.encode_file(path, token_path)
            print(f"{name}: {count:,} tokens")
        datasets[name] = tritium.TokenDataset(token_path, seed=arguments.seed + len(datasets))
    return tokenizer, datasets


def learning_rate(step, arguments):
    if step < arguments.warmup:
        return arguments.lr * (step + 1) / arguments.warmup
    progress = (step - arguments.warmup) / max(1, arguments.steps - arguments.warmup)
    return arguments.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def evaluate(model, dataset, arguments, batches=8):
    with tritium.no_grad():
        total = 0.0
        for _ in range(batches):
            inputs, targets = dataset.batch(arguments.batch, arguments.context)
            total += F.cross_entropy(model(inputs, batch=arguments.batch), targets).item()
    return total / batches


def save_checkpoint(model, arguments, step):
    tritium.save(model.state_dict(), arguments.output / "model.bin")
    checkpoint = {"config": model.config.to_dict(), "step": step}
    (arguments.output / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2) + "\n")


def main(arguments=None):
    arguments = parse_arguments(arguments)
    tokenizer, datasets = prepare_tokens(arguments)
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        width=arguments.width,
        layers=arguments.layers,
        heads=arguments.heads,
        hidden=arguments.hidden,
        context=arguments.context,
    )
    model = TernaryTransformer(config)
    start = 0
    checkpoint_path = arguments.output / "checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        if TransformerConfig(**checkpoint["config"]) != config:
            raise SystemExit("the saved checkpoint has a different model shape")
        model.load_state_dict(tritium.load(arguments.output / "model.bin"))
        start = checkpoint["step"]
    parameters = sum(len(parameter.tolist()) for parameter in model.parameters())
    print(f"model: {parameters:,} parameters, from step {start}")
    optimizer = tritium.optim.AdamW(
        model.parameters(), lr=arguments.lr, weight_decay=arguments.weight_decay
    )
    began = time.perf_counter()
    for step in range(start, arguments.steps):
        optimizer.lr = learning_rate(step, arguments)
        inputs, targets = datasets["train"].batch(arguments.batch, arguments.context)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(inputs, batch=arguments.batch), targets)
        loss.backward()
        optimizer.step()
        done = step + 1
        if done % arguments.log_every == 0 or done == arguments.steps:
            seconds = (time.perf_counter() - began) / (done - start)
            line = f"step {done}: loss {loss.item():.3f}, {seconds:.2f}s/step"
            if "validation" in datasets:
                line += f", validation {evaluate(model, datasets['validation'], arguments):.3f}"
            print(line, flush=True)
        if done % arguments.save_every == 0 or done == arguments.steps:
            save_checkpoint(model, arguments, done)


if __name__ == "__main__":
    main()
