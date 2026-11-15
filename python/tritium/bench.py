import argparse
from pathlib import Path

from tritium.checkpoint import load_run


def main(arguments=None):
    parser = argparse.ArgumentParser(prog="python -m tritium.bench")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--clock-mhz", type=float)
    arguments = parser.parse_args(arguments)
    model, tokenizer, _ = load_run(arguments.run)
    device_model = model.to("tritium", backend="rtl")
    prompt = tokenizer.encode("Once upon a time")[: arguments.tokens]
    tokens = prompt + [0] * (arguments.tokens - len(prompt))
    for token in tokens:
        device_model.decode(token)
    statistics = device_model.statistics()
    count = statistics["tokens"]
    cycles = statistics["cycles"] / count
    weight_bytes = device_model.ternary_weight_bytes + device_model.head_weight_bytes
    lines = [
        "source: RTL simulation (Verilator), core only, abstract memory",
        f"tokens: {count}",
        f"cycles per token: {cycles:,.0f}",
        f"warp instructions per token: {statistics['warp_instructions'] / count:,.0f}",
        f"kernel launches per token: {statistics['launches'] / count:.0f}",
        f"weight bytes read per token: {weight_bytes:,}",
        f"ternary weight bytes: {device_model.ternary_weight_bytes:,}",
    ]
    if arguments.clock_mhz:
        tokens_per_second = arguments.clock_mhz * 1e6 / cycles
        lines.append(f"tokens per second at {arguments.clock_mhz:g} MHz: {tokens_per_second:.2f}")
    print("\n".join(lines))
    return statistics


if __name__ == "__main__":
    main()
