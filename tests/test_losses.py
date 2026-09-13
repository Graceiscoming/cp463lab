import numpy as np
import pytest
from sklearn.metrics import log_loss

from nnlab.losses import bce_loss, binary_cross_entropy, cross_entropy, l2_penalty


def test_bce_matches_sklearn(rng):
    y = rng.integers(0, 2, 50)
    a = rng.random(50)
    assert binary_cross_entropy(a, y) == pytest.approx(log_loss(y, a), rel=1e-9)


def test_bce_accepts_deck_shapes():
    A = np.array([[0.9, 0.2, 0.6]])
    Y = np.array([[1, 0, 1]])
    expected = -(np.log(0.9) + np.log(0.8) + np.log(0.6)) / 3
    assert binary_cross_entropy(A, Y) == pytest.approx(expected)


def test_bce_handles_extremes_without_nan():
    val = binary_cross_entropy(np.array([0.0, 1.0]), np.array([1, 0]))
    assert np.isfinite(val) and val > 20          # log(1e-12) ≈ -27.6


def test_bce_per_sample_when_confident_is_near_zero():
    assert bce_loss(np.array([0.999]), np.array([1]))[0] < 0.01
    assert bce_loss(np.array([0.001]), np.array([0]))[0] < 0.01


def test_cross_entropy_axis0():
    P = np.array([[0.7, 0.2], [0.2, 0.7], [0.1, 0.1]])     # (C=3, m=2)
    Y = np.array([[1, 0], [0, 1], [0, 0]])
    assert cross_entropy(P, Y, axis=0) == pytest.approx(-np.log(0.7), rel=1e-9)


def test_l2_penalty_ignores_bias():
    params = {"W1": np.ones((2, 2)), "b1": np.ones((2, 1)) * 100, "W2": np.ones((1, 2))}
    # ‖W1‖² = 4, ‖W2‖² = 2 → λ/(2m) × 6 เมื่อ λ=1, m=3 → 1.0
    assert l2_penalty(params, lam=1.0, m=3) == pytest.approx(1.0)
    assert l2_penalty([params["W1"], params["W2"]], lam=1.0, m=3) == pytest.approx(1.0)
    assert l2_penalty(params, lam=0.0, m=3) == 0.0
