import math
import zlib

from tritium import functional as F
from tritium._core import Tensor, normal

RESIDUAL_ONE = 256.0


def _seed(name):
    return zlib.crc32(name.encode())


class Module:
    def __init__(self):
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})

    def __setattr__(self, name, value):
        if isinstance(value, Tensor) and value.requires_grad:
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def __call__(self, *arguments, **keywords):
        return self.forward(*arguments, **keywords)

    def forward(self, *arguments, **keywords):
        raise NotImplementedError

    def named_parameters(self, prefix=""):
        for name, parameter in self._parameters.items():
            yield prefix + name, parameter
        for name, module in self._modules.items():
            yield from module.named_parameters(f"{prefix}{name}.")

    def parameters(self):
        return [parameter for _, parameter in self.named_parameters()]

    def state_dict(self):
        return {name: parameter.detach() for name, parameter in self.named_parameters()}

    def load_state_dict(self, state_dict):
        parameters = dict(self.named_parameters())
        missing = sorted(parameters.keys() - state_dict.keys())
        unexpected = sorted(state_dict.keys() - parameters.keys())
        if missing or unexpected:
            raise KeyError(f"missing {missing}, unexpected {unexpected}")
        for name, parameter in parameters.items():
            parameter.copy_(state_dict[name])


class ModuleList(Module):
    def __init__(self, modules):
        super().__init__()
        self._items = list(modules)
        for index, module in enumerate(self._items):
            self._modules[str(index)] = module

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]


class Linear(Module):
    def __init__(self, in_features, out_features, seed=0):
        super().__init__()
        std = 1.0 / math.sqrt(in_features)
        self.weight = normal([out_features, in_features], std, seed, True)

    def forward(self, input):
        return F.linear(input, self.weight)


class BitLinear(Module):
    def __init__(self, in_features, out_features, input_scale=RESIDUAL_ONE, seed=0):
        super().__init__()
        self.input_scale = input_scale
        std = 1.0 / math.sqrt(in_features)
        self.weight = normal([out_features, in_features], std, seed, True)

    def forward(self, input):
        output = F.bit_linear(input * self.input_scale, self.weight)
        return output * (1.0 / self.input_scale)


class Int8Linear(Module):
    def __init__(self, in_features, out_features, input_scale=RESIDUAL_ONE, seed=0):
        super().__init__()
        self.input_scale = input_scale
        std = 1.0 / math.sqrt(in_features)
        self.weight = normal([out_features, in_features], std, seed, True)

    def forward(self, input):
        output = F.int8_linear(input * self.input_scale, self.weight)
        return output * (1.0 / self.input_scale)


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, seed=0):
        super().__init__()
        self.weight = normal([num_embeddings, embedding_dim], 0.1, seed, True)

    def forward(self, indices):
        return F.embedding(self.weight, list(indices))


__all__ = [
    "BitLinear",
    "Embedding",
    "Int8Linear",
    "Linear",
    "Module",
    "ModuleList",
    "RESIDUAL_ONE",
]
