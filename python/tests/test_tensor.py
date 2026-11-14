import pytest

import tritium


def test_tensor_infers_nested_shape():
    value = tritium.tensor([[1, 2, 3], [4, 5, 6]])
    assert value.shape == [2, 3]
    assert value.tolist() == [1, 2, 3, 4, 5, 6]


def test_ragged_lists_are_rejected():
    with pytest.raises(ValueError):
        tritium.tensor([[1, 2], [3]])


def test_backward_fills_leaf_gradients():
    left = tritium.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    right = tritium.tensor([[0.5], [-1.0]], requires_grad=True)
    loss = ((left @ right) * 2.0 + 1.0).sum()
    loss.backward()
    assert left.grad.tolist() == [1.0, -2.0, 1.0, -2.0]
    assert right.grad.tolist() == [8.0, 12.0]


def test_no_grad_skips_the_graph():
    weight = tritium.randn(3, 3, requires_grad=True)
    with tritium.no_grad():
        output = (weight * 2.0).sum()
    assert not output.requires_grad
    assert tritium.is_grad_enabled()


def test_shape_mismatch_raises_value_error():
    with pytest.raises(ValueError):
        tritium.zeros(2, 3) + tritium.zeros(3, 2)


def test_save_and_load_round_trip(tmp_path):
    state = {"weight": tritium.tensor([[1.5, -2.0]]), "bias": tritium.tensor([3.0])}
    tritium.save(state, tmp_path / "state.bin")
    loaded = tritium.load(tmp_path / "state.bin")
    assert loaded["weight"].shape == [1, 2]
    assert loaded["bias"].tolist() == [3.0]
