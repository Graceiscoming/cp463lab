import numpy as np
import pytest

from nnlab.data import load_blobs
from nnlab.perceptron import Perceptron, backward, forward, gradient_check, naive_epoch


def test_forward_reproduces_deck_demo(deck_XY):
    """สไลด์ p.42-49: θ₀=0.3, θ=(0.1, 0.2) → θᵀX = [0.5, 0.5, 0.4], z = [0.8, 0.8, 0.7]"""
    X, _ = deck_XY
    w = np.array([[0.1], [0.2]])
    A, Z = forward(w, 0.3, X)
    assert np.allclose(w.T @ X, [[0.5, 0.5, 0.4]])
    assert np.allclose(Z, [[0.8, 0.8, 0.7]])
    assert np.allclose(A, [[0.6900, 0.6900, 0.6682]], atol=1e-4)    # errata ของ p.52
    assert Z.shape == A.shape == (1, 3)


def test_backward_dz_is_a_minus_y(deck_XY):
    X, Y = deck_XY
    A, _ = forward(np.array([[0.1], [0.2]]), 0.3, X)
    dw, db, dz = backward(X, A, Y)
    assert np.allclose(dz, A - Y)
    assert dw.shape == (2, 1) and isinstance(db, float)
    assert np.allclose(dw, X @ dz.T / 3)


def test_gradient_check_small(deck_XY):
    X, Y = deck_XY
    assert gradient_check(np.array([[0.1], [0.2]]), 0.3, X, Y) < 1e-7


def test_naive_loop_equals_vectorized(deck_XY):
    X, Y = deck_XY
    w, b, lr = np.array([[0.1], [0.2]]), 0.3, 0.5
    w_naive, b_naive, J_naive = naive_epoch(w, b, X, Y, lr)
    A, _ = forward(w, b, X)
    dw, db, _ = backward(X, A, Y)
    assert np.allclose(w_naive, w - lr * dw) and b_naive == pytest.approx(b - lr * db)
    from nnlab.losses import binary_cross_entropy

    assert J_naive == pytest.approx(binary_cross_entropy(A, Y))


def test_perceptron_learns_blobs():
    X, y = load_blobs(n=200, cluster_std=1.0)
    model = Perceptron(lr=0.1, epochs=300).fit(X, y)
    assert model.w_.shape == (2, 1)
    assert model.evaluate(X, y)["accuracy"] >= 0.98
    assert model.history_["cost"][-1] < model.history_["cost"][0]
    assert model.predict_proba(X).shape == (200,)


def test_perceptron_l2_shrinks_weights():
    X, y = load_blobs(n=200, cluster_std=1.0)
    plain = Perceptron(lr=0.1, epochs=300).fit(X, y)
    reg = Perceptron(lr=0.1, epochs=300, l2=5.0).fit(X, y)
    assert np.linalg.norm(reg.w_) < np.linalg.norm(plain.w_)


def test_save_load_round_trip(tmp_path):
    X, y = load_blobs(n=100)
    model = Perceptron(lr=0.1, epochs=50).fit(X, y)
    path = model.save(tmp_path / "perceptron")
    loaded = Perceptron.load(path)
    assert np.allclose(loaded.predict_proba(X), model.predict_proba(X))
    assert loaded.history_["cost"] == pytest.approx(model.history_["cost"])
