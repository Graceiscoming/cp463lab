import numpy as np
import pytest

from nnlab.activations import ACTIVATIONS, d_relu, d_sigmoid, d_tanh, numerical_derivative, relu, sigmoid, softmax, tanh


@pytest.mark.parametrize("name", ["sigmoid", "tanh", "relu"])
def test_derivative_matches_numerical(name):
    g, dg = ACTIVATIONS[name]
    z = np.linspace(-3, 3, 25) + 0.013      # เลี่ยง z = 0 พอดี (ReLU นิยามไม่ได้)
    numeric = numerical_derivative(g, z, dx=1e-5, method="central")
    assert np.allclose(dg(z), numeric, atol=1e-4)


def test_sigmoid_properties():
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert sigmoid(np.array([-1000.0, 1000.0])).tolist() == pytest.approx([0.0, 1.0], abs=1e-12)
    with np.errstate(over="raise"):          # ต้องไม่ overflow เลย
        sigmoid(np.array([-1000.0, 1000.0]))


def test_sigmoid_of_deck_demo():
    """errata: σ([0.8, 0.8, 0.7]) ไม่ใช่ [0.49, 0.45, 0.45] ตามที่สไลด์ p.52 พิมพ์ไว้"""
    assert np.allclose(sigmoid(np.array([0.8, 0.8, 0.7])), [0.6900, 0.6900, 0.6682], atol=1e-4)


def test_tanh_and_relu_ranges():
    z = np.linspace(-5, 5, 11)
    assert np.all(np.abs(tanh(z)) < 1)
    assert np.all(relu(z) >= 0) and relu(np.array([-2.0, 3.0])).tolist() == [0.0, 3.0]
    assert d_relu(np.array([-1.0, 2.0])).tolist() == [0.0, 1.0]


def test_closed_forms():
    z = np.array([-1.0, 0.0, 2.0])
    assert np.allclose(d_sigmoid(z), sigmoid(z) * (1 - sigmoid(z)))
    assert np.allclose(d_tanh(z), 1 - np.tanh(z) ** 2)


def test_softmax_sums_to_one_and_is_stable():
    z = np.array([[1000.0, 1000.0, 999.0], [0.0, 0.0, 0.0]]).T   # (C=3, m=2)
    p = softmax(z, axis=0)
    assert np.allclose(p.sum(axis=0), 1.0)
    assert np.isfinite(p).all()
    assert np.allclose(softmax(np.array([1.0, 2.0]), axis=0), [0.26894142, 0.73105858])


def test_numerical_derivative_forward_matches_slide_p9():
    slope = numerical_derivative(lambda x: x**2, np.array(2.0), dx=0.001, method="forward")
    assert slope == pytest.approx(4.001, abs=1e-6)     # สไลด์: 0.004/0.001 = 4
