from tritium import functional, nn, optim
from tritium._core import Tensor, TokenDataset, Tokenizer, is_grad_enabled, set_grad_enabled
from tritium._core import load_tensors as _load_tensors
from tritium._core import normal as _normal
from tritium._core import save_tensors as _save_tensors

__all__ = [
    "Tensor",
    "TokenDataset",
    "Tokenizer",
    "functional",
    "is_grad_enabled",
    "load",
    "nn",
    "no_grad",
    "optim",
    "randn",
    "save",
    "tensor",
    "zeros",
]


def _flatten(values, shape):
    if isinstance(values, (list, tuple)):
        shape.append(len(values))
        flat = []
        for index, value in enumerate(values):
            nested = []
            flat.extend(_flatten(value, nested))
            if index == 0:
                inner = nested
            elif nested != inner:
                raise ValueError("nested lists must have equal lengths")
        shape.extend(inner if values else [])
        return flat
    return [float(values)]


def tensor(values, requires_grad=False):
    shape = []
    data = _flatten(values, shape)
    return Tensor(data, shape, requires_grad)


def zeros(*shape, requires_grad=False):
    count = 1
    for dimension in shape:
        count *= dimension
    return Tensor([0.0] * count, list(shape), requires_grad)


def randn(*shape, std=1.0, seed=0, requires_grad=False):
    return _normal(list(shape), std, seed, requires_grad)


class no_grad:
    def __enter__(self):
        self._previous = set_grad_enabled(False)
        return self

    def __exit__(self, *exception):
        set_grad_enabled(self._previous)
        return False


def save(state_dict, path):
    _save_tensors(str(path), list(state_dict.items()))


def load(path):
    return dict(_load_tensors(str(path)))
