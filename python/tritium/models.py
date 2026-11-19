from dataclasses import asdict, dataclass

from tritium import functional as F
from tritium import nn
from tritium._core import DeviceModel
from tritium.nn import _seed

SQUARED_ONE = nn.RESIDUAL_ONE * nn.RESIDUAL_ONE


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = 2048
    width: int = 128
    layers: int = 4
    heads: int = 4
    hidden: int = 512
    context: int = 128

    def to_dict(self):
        return asdict(self)


class Block(nn.Module):
    def __init__(self, config, name):
        super().__init__()
        width, hidden = config.width, config.hidden
        self.query = nn.BitLinear(width, width, seed=_seed(name + ".query"))
        self.key = nn.BitLinear(width, width, seed=_seed(name + ".key"))
        self.value = nn.BitLinear(width, width, seed=_seed(name + ".value"))
        self.output = nn.BitLinear(width, width, seed=_seed(name + ".output"))
        self.up = nn.BitLinear(width, hidden, seed=_seed(name + ".up"))
        self.down = nn.BitLinear(hidden, width, input_scale=SQUARED_ONE, seed=_seed(name + ".down"))

    def forward(self, x, batch, heads):
        attention = F.causal_attention(self.query(x), self.key(x), self.value(x), batch, heads)
        x = x + self.output(attention)
        hidden = F.relu_squared(self.up(x) * nn.RESIDUAL_ONE) * (1.0 / SQUARED_ONE)
        return x + self.down(hidden)


class TernaryTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width, seed=_seed("token"))
        self.position_embedding = nn.Embedding(config.context, config.width, seed=_seed("position"))
        blocks = (Block(config, f"blocks.{index}") for index in range(config.layers))
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Int8Linear(config.width, config.vocab_size, seed=_seed("head"))

    def forward(self, tokens, batch=1):
        sequence = len(tokens) // batch
        if sequence * batch != len(tokens) or sequence > self.config.context:
            raise ValueError(
                f"{len(tokens)} tokens do not form {batch} sequences of at most "
                f"{self.config.context}"
            )
        positions = [position for _ in range(batch) for position in range(sequence)]
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x, batch, self.config.heads)
        return self.head(x)

    def to(self, device, backend="simulator", weights="ternary"):
        if device == "cpu":
            return self
        if device != "tritium":
            raise ValueError(f"unknown device {device!r}; expected 'cpu' or 'tritium'")
        state = list(self.state_dict().items())
        return DeviceModel(self.config.to_dict(), state, backend, weights)
