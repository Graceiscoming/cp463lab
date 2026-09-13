import numpy as np
import pytest
import torch

from nnlab.optimizers import SGD, Adam, Momentum, RMSProp, lr_decay, make_optimizer


def _problem(rng):
    params = {"W1": rng.normal(size=(3, 2)), "b1": rng.normal(size=(3, 1))}
    grads = {"dW1": rng.normal(size=(3, 2)), "db1": rng.normal(size=(3, 1))}
    return params, grads


def test_sgd_is_plain_gradient_descent(rng):
    params, grads = _problem(rng)
    before = {k: v.copy() for k, v in params.items()}
    SGD(lr=0.1).step(params, grads)
    assert np.allclose(params["W1"], before["W1"] - 0.1 * grads["dW1"])


def test_momentum_beta_zero_equals_sgd(rng):
    p1, grads = _problem(rng)
    p2 = {k: v.copy() for k, v in p1.items()}
    SGD(lr=0.1).step(p1, grads)
    Momentum(lr=0.1, beta=0.0).step(p2, grads)
    assert np.allclose(p1["W1"], p2["W1"])


def test_adam_first_step_is_signed_lr(rng):
    """หลัง bias correction step แรกของ Adam ≈ −lr·sign(grad) ไม่ว่า gradient ใหญ่แค่ไหน"""
    params, grads = _problem(rng)
    before = params["W1"].copy()
    Adam(lr=0.01).step(params, grads)
    assert np.allclose(params["W1"] - before, -0.01 * np.sign(grads["dW1"]), atol=1e-6)


@pytest.mark.parametrize("steps", [1, 5])
def test_adam_matches_torch(steps, rng):
    W0 = rng.normal(size=(4, 3))
    grads_seq = [rng.normal(size=(4, 3)) for _ in range(steps)]
    ours = {"W1": W0.copy()}
    opt = Adam(lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8)
    for g in grads_seq:
        opt.step(ours, {"dW1": g})
    w = torch.tensor(W0.copy(), requires_grad=True)
    topt = torch.optim.Adam([w], lr=0.01, betas=(0.9, 0.999), eps=1e-8)
    for g in grads_seq:
        w.grad = torch.tensor(g)
        topt.step()
    assert np.allclose(ours["W1"], w.detach().numpy(), atol=1e-6)


def test_rmsprop_matches_torch(rng):
    W0 = rng.normal(size=(4, 3))
    grads_seq = [rng.normal(size=(4, 3)) for _ in range(4)]
    ours = {"W1": W0.copy()}
    opt = RMSProp(lr=0.01, beta=0.99, eps=1e-8)
    for g in grads_seq:
        opt.step(ours, {"dW1": g})
    w = torch.tensor(W0.copy(), requires_grad=True)
    topt = torch.optim.RMSprop([w], lr=0.01, alpha=0.99, eps=1e-8)
    for g in grads_seq:
        w.grad = torch.tensor(g)
        topt.step()
    assert np.allclose(ours["W1"], w.detach().numpy(), atol=1e-6)


def test_optimizers_converge_on_quadratic():
    """min f(w) = ‖w − 3‖² จากจุดเริ่ม 0 — ทุก optimizer ต้องเข้าใกล้ 3"""
    for name, lr in [("sgd", 0.1), ("momentum", 0.1), ("rmsprop", 0.05), ("adam", 0.05)]:
        opt = make_optimizer(name, lr)
        params = {"W1": np.zeros((2, 1))}
        for _ in range(400):
            opt.step(params, {"dW1": 2 * (params["W1"] - 3.0)})
        assert np.allclose(params["W1"], 3.0, atol=0.05), name


def test_lr_decay_and_factory():
    assert lr_decay(0.1, 0) == 0.1 and lr_decay(0.1, 100, 0.01) == pytest.approx(0.05)
    with pytest.raises(ValueError):
        make_optimizer("newton", 0.1)
