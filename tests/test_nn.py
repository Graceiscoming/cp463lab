import numpy as np
import pytest

from nnlab.data import load_digits_images, load_moons
from nnlab.nn import NeuralNetwork, backward, forward, gradient_check, init_params


def test_param_shapes_follow_slide_p77_78():
    params = init_params([3, 5, 4, 3, 1])
    assert params["W1"].shape == (5, 3) and params["b1"].shape == (5, 1)
    assert params["W3"].shape == (3, 4) and params["b3"].shape == (3, 1)
    assert params["W4"].shape == (1, 3)
    assert set(params) == {"W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4"}
    assert not np.allclose(params["W1"], 0)


def test_forward_shapes_and_cache(rng):
    X = rng.normal(size=(3, 7))                        # n_x=3, m=7
    params = init_params([3, 5, 1])
    A, caches = forward(X, params, ["relu", "sigmoid"])
    assert A.shape == (1, 7) and len(caches) == 2
    assert caches[0]["Z"].shape == (5, 7) and caches[1]["A_prev"].shape == (5, 7)


@pytest.mark.parametrize("acts", [["relu", "sigmoid"], ["tanh", "sigmoid"], ["tanh", "relu", "sigmoid"]])
def test_gradient_check_binary(acts, rng):
    sizes = [3] + [4] * (len(acts) - 1) + [1]
    X = rng.normal(size=(3, 6))
    Y = rng.integers(0, 2, (1, 6)).astype(float)
    params = init_params(sizes, seed=1)
    assert gradient_check(X, Y, params, acts) < 1e-6


def test_gradient_check_with_l2_and_softmax(rng):
    X = rng.normal(size=(4, 6))
    Y = np.eye(3)[rng.integers(0, 3, 6)].T          # (C=3, m=6) one-hot
    params = init_params([4, 5, 3], seed=2)
    assert gradient_check(X, Y, params, ["relu", "softmax"], l2=0.5) < 1e-6


def test_db_keeps_dims(rng):
    X = rng.normal(size=(2, 5))
    Y = rng.integers(0, 2, (1, 5)).astype(float)
    params = init_params([2, 3, 1])
    _, caches = forward(X, params, ["relu", "sigmoid"])
    grads = backward(Y, caches, params, ["relu", "sigmoid"])
    assert grads["db1"].shape == (3, 1) and grads["dW1"].shape == (3, 2)


def test_network_overfits_tiny_dataset():
    X, y = load_moons(n=20, noise=0.1)
    model = NeuralNetwork([2, 16, 1], lr=0.3, epochs=1500).fit(X, y)
    assert model.evaluate(X, y)["accuracy"] == 1.0


def test_network_solves_moons_with_adam():
    X, y = load_moons(n=400, noise=0.15)
    model = NeuralNetwork([2, 16, 8, 1], lr=0.01, epochs=300, optimizer="adam", batch_size=32).fit(X, y)
    assert model.evaluate(X, y)["accuracy"] >= 0.95


def test_dropout_only_in_training(rng):
    X = rng.normal(size=(3, 200))
    params = init_params([3, 50, 1])
    A_train, caches = forward(X, params, ["relu", "sigmoid"], keep_prob=0.5, training=True, rng=rng)
    A_eval, _ = forward(X, params, ["relu", "sigmoid"], keep_prob=0.5, training=False)
    A_plain, _ = forward(X, params, ["relu", "sigmoid"])
    assert np.allclose(A_eval, A_plain)                       # predict ไม่ใช้ dropout
    D = caches[0]["D"]
    assert D is not None and 0.35 < D.mean() < 0.65           # ประมาณครึ่งหนึ่งถูกปิด
    # inverted dropout รักษาค่าคาดหวังของ activation
    hidden_plain = np.maximum(0, params["W1"] @ X + params["b1"])
    assert abs(caches[0]["A"].mean() - hidden_plain.mean()) < 0.15 * hidden_plain.mean() + 1e-6


def test_multiclass_digits():
    images, y = load_digits_images()
    X = images.reshape(len(images), -1) / 16.0
    model = NeuralNetwork([64, 32, 10], lr=0.01, epochs=60, optimizer="adam", batch_size=64).fit(X, y)
    assert model.predict_proba(X).shape == (len(X), 10)
    assert model.evaluate(X, y)["accuracy"] >= 0.95


def test_early_stopping_and_save_load(tmp_path):
    """train set เล็ก + network ใหญ่ → overfit → val cost เริ่มแย่ลง → ต้องหยุดก่อนครบ epochs"""
    X, y = load_moons(n=300, noise=0.35)
    model = NeuralNetwork([2, 32, 32, 1], lr=0.01, epochs=3000, optimizer="adam", early_stopping_patience=25)
    model.fit(X[:30], y[:30], X[30:], y[30:])
    assert len(model.history_["cost"]) < 3000
    assert hasattr(model, "stopped_epoch_")
    assert min(model.history_["val_cost"]) <= model.history_["val_cost"][-1] + 1e-9
    path = model.save(tmp_path / "nn")
    loaded = NeuralNetwork.load(path)
    assert np.allclose(loaded.predict_proba(X), model.predict_proba(X))
