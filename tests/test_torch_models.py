import numpy as np
import pytest
import torch

from nnlab.data import load_moons
from nnlab.torch_models import LeNet5, ResidualBlock, TorchMLP, TorchPerceptron, TorchTrainer, count_parameters


def test_lenet5_shapes_match_slide_p199_200():
    """28×28 (pad=2) และ 32×32 (pad=0) ต้องได้ 28×28×6 → 14×14×6 → 10×10×16 → 5×5×16 → 400 → 120 → 84 → 10"""
    for in_hw, pad in [((1, 28, 28), 2), ((1, 32, 32), 0)]:
        rows = LeNet5(pad=pad).shapes(in_hw)
        shapes = [s for _, s in rows]
        assert (6, 28, 28) in shapes and (6, 14, 14) in shapes
        assert (16, 10, 10) in shapes and (16, 5, 5) in shapes
        assert (400,) in shapes and shapes[-3:] == [(120,), (84,), (10,)]


def test_lenet5_parameter_count_is_about_60k():
    n = count_parameters(LeNet5())
    assert 55_000 < n < 65_000            # สไลด์ p.202: ~60k parameters


def test_linear_weight_is_deck_W_shape():
    net = TorchMLP([3, 5, 1])
    assert tuple(net.net[0].weight.shape) == (5, 3)       # (n[l], n[l-1]) เหมือน W[1] ของสไลด์
    assert tuple(net.net[0].bias.shape) == (5,)
    assert net(torch.zeros(7, 3)).shape == (7,)            # binary → (m,)
    assert TorchMLP([3, 5, 4])(torch.zeros(7, 3)).shape == (7, 4)


def test_residual_block_identity_when_zero_weights():
    block = ResidualBlock(4)
    for p in block.parameters():
        torch.nn.init.zeros_(p)
    a = torch.relu(torch.randn(2, 4))
    assert torch.allclose(block(a), a)                     # z = 0 → a[l+2] = g(0 + a[l]) = a[l]


def test_trainer_learns_moons_and_round_trips(tmp_path):
    X, y = load_moons(n=400, noise=0.15)
    trainer = TorchTrainer(TorchMLP([2, 16, 8, 1]), loss="bce", optimizer="adam", lr=0.01, epochs=60, batch_size=32, device="cpu")
    trainer.fit(X, y, X, y)
    assert trainer.evaluate(X, y)["accuracy"] >= 0.95
    assert len(trainer.history_["cost"]) == 60 and len(trainer.history_["val_acc"]) == 60
    path = trainer.save(tmp_path / "mlp")
    fresh = TorchTrainer(TorchMLP([2, 16, 8, 1]), loss="bce", device="cpu").load(path)
    assert np.allclose(fresh.predict_proba(X), trainer.predict_proba(X), atol=1e-6)


def test_perceptron_logits_shape():
    net = TorchPerceptron(30)
    assert net(torch.zeros(5, 30)).shape == (5,)


# ---------------------------------------------------------------- early stopping (เพิ่ม 2026-09-08)
from nnlab.torch_models import EarlyStopping  # noqa: E402


def test_early_stopping_scripted_losses(tmp_path):
    model = TorchMLP([2, 3, 1])
    es = EarlyStopping(patience=2, checkpoint_path=tmp_path / "best.pt")
    losses = [1.0, 0.9, 0.95, 0.96, 0.97]
    stopped_at = None
    best_weights = None
    for epoch, loss in enumerate(losses):
        with torch.no_grad():
            model.net[0].weight.fill_(float(epoch))            # น้ำหนักเปลี่ยนทุก epoch
        if epoch == 1:
            best_weights = model.net[0].weight.clone()
        if es.step(loss, model):
            stopped_at = epoch
            break
    assert stopped_at == 3 and es.best_epoch == 1 and es.stopped_epoch == 3    # stopped = best + patience
    assert torch.equal(model.net[0].weight, best_weights)                    # restore แล้ว
    ckpt = torch.load(tmp_path / "best.pt", weights_only=False)
    assert ckpt["epoch"] == 1 and ckpt["val_loss"] == pytest.approx(0.9)


def test_trainer_early_stopping_on_moons():
    X, y = load_moons(n=300, noise=0.35)
    trainer = TorchTrainer(TorchMLP([2, 32, 32, 1]), loss="bce", optimizer="adam", lr=0.01, epochs=400, batch_size=None,
                           early_stopping_patience=25, device="cpu")
    trainer.fit(X[:30], y[:30], X[30:], y[30:])
    assert trainer.stopped_epoch_ is not None and trainer.stopped_epoch_ < 399
    val = np.array(trainer.history_["val_cost"])
    assert trainer.stopped_epoch_ == int(np.argmin(val)) + 25 == trainer.best_epoch_ + 25
    assert len(val) == trainer.stopped_epoch_ + 1


def test_trainer_prebuilt_optimizer_and_frozen_params():
    X, y = load_moons(n=100, noise=0.2)
    net = TorchMLP([2, 8, 1])
    for p in net.net[0].parameters():
        p.requires_grad = False                                   # freeze ชั้นแรก
    frozen_before = net.net[0].weight.clone()
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=0.01)
    trainer = TorchTrainer(net, loss="bce", optimizer=opt, epochs=5, batch_size=16, device="cpu").fit(X, y)
    assert torch.equal(net.net[0].weight, frozen_before)
    assert "Adam" in repr(trainer)
    groups = [{"params": list(net.net[2].parameters()), "lr": 0.1}]
    TorchTrainer(TorchMLP([2, 8, 1]), loss="bce", param_groups=groups, epochs=1, batch_size=16, device="cpu").fit(X, y)
