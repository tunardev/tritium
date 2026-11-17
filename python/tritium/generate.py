import argparse
import time
from pathlib import Path

from tritium.checkpoint import load_run


def main(arguments=None):
    parser = argparse.ArgumentParser(prog="python -m tritium.generate")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--tokens", type=int, default=80)
    parser.add_argument("--backend", default="reference", choices=["reference", "simulator", "rtl"])
    parser.add_argument("--weights", default="ternary", choices=["ternary", "int8"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args(arguments)
    model, tokenizer, _ = load_run(arguments.run)
    device_model = model.to("tritium", backend=arguments.backend, weights=arguments.weights)
    prompt = tokenizer.encode(arguments.prompt)
    began = time.perf_counter()
    generated = device_model.generate(
        prompt, arguments.tokens, temperature=arguments.temperature, seed=arguments.seed
    )
    seconds = time.perf_counter() - began
    print(tokenizer.decode(prompt + generated))
    print(f"{arguments.backend}: {len(generated)} tokens in {seconds:.1f}s")
    return generated


if __name__ == "__main__":
    main()
