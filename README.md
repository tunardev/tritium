# tritium

A small GPU, its CUDA-like toolchain and a PyTorch-style framework, built together for one workload: ternary (1.58-bit) language model inference.

- `hardware/` SystemVerilog GPU: SIMT compute units of 16 lanes with a multiplier-free ternary dot unit, a Verilator harness and a Yosys and nextpnr flow for the ECP5-85F.
- `crates/` Rust: instruction set and assembler, ISA simulator, Verilator model, runtime, kernel language compiler, framework core and the integer model runtime.
- `python/` the `tritium` package: tensors with autograd, `nn`, `optim`, ternary `BitLinear` training and `model.to("tritium")`.

## Usage

Needs Rust, Verilator 5 and uv.

```sh
cargo test
cd python && uv sync && uv run pytest
```

Train on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), then generate on the RTL GPU:

```sh
uv run python -m tritium.train --text train.txt --validation valid.txt --output runs/stories
uv run python -m tritium.generate --run runs/stories --backend rtl --prompt "Once upon a time"
uv run python -m tritium.bench --run runs/stories --weights ternary
```

```python
from tritium.checkpoint import load_run

model, tokenizer, _ = load_run("runs/stories")
gpu = model.to("tritium", backend="rtl")
tokens = gpu.generate(tokenizer.encode("Once upon a time"), 40, temperature=0.8)
print(tokenizer.decode(tokens))
```

Synthesis: `cd hardware && make synthesis timing VARIANT=ternary` (or `int8`).

## Design

- Weights pack five trits per byte. `tdot` adds or subtracts 20 int8 activations per weight word, so every projection is free of multipliers. Only `mul16` multiplies, for attention and the int8 output head.
- Activations are int8 per token with a power-of-two absmax shift. The residual stream is fixed point in int32, and softmax uses table-driven `exp2` and `rsqrt`.
- Model kernels are written in the kernel language and compiled by `tritium-compiler`. Every token on the GPU matches the integer reference bit for bit, on the simulator and on the RTL.
- The int8 variant runs the same model with int8 weights and a `dot8` multiply-accumulate unit, to compare hardware cost.
