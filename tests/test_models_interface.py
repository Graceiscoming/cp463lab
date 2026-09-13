"""ทุกโมเดลต้องใช้แทนกันได้ผ่าน interface เดียว และเรียน breast_cancer ได้ดีพอ"""
import numpy as np
import pytest

from nnlab import Classifier
from nnlab.config import MODELS, TrainConfig
from nnlab.data import get_dataset
from nnlab.experiment import build_model, load_run, run


@pytest.fixture(scope="module")
def breast():
    return get_dataset("breast_cancer")


@pytest.mark.parametrize("name", MODELS)
def test_every_model_fits_breast_cancer(name, breast, tmp_path):
    X_tr, X_te, y_tr, y_te, n_classes = breast
    cfg = TrainConfig(model=name, dataset="breast_cancer", lr=0.1 if "torch" not in name else 0.01,
                      epochs=200 if "torch" not in name else 60, batch_size=32 if "torch" in name else None,
                      hidden=(16,), optimizer="sgd" if name == "numpy-perceptron" else "adam")
    model = build_model(cfg, X_tr.shape[1], n_classes)
    assert isinstance(model, Classifier)
    model.fit(X_tr, y_tr)
    metrics = model.evaluate(X_te, y_te)
    assert metrics["accuracy"] > 0.9, (name, metrics)
    assert model.predict_proba(X_te).shape == (len(y_te),)
    assert set(np.unique(model.predict(X_te))) <= {0, 1}
    path = model.save(tmp_path / name)
    assert path.exists()


def test_run_and_load_round_trip(tmp_path):
    cfg = TrainConfig(model="numpy-perceptron", dataset="blobs", epochs=100, out_dir=str(tmp_path), name="t1")
    out = run(cfg, save_plots=True)
    for f in ["config.json", "metrics.json", "history.csv", "model.npz", "cost.png", "confusion.png", "confusion.csv"]:
        assert (out / f).exists(), f
    cfg2, model, (X_tr, X_te, y_tr, y_te), metrics = load_run(out)
    assert cfg2.model == "numpy-perceptron" and metrics["accuracy"] >= 0.95
    assert model.evaluate(X_te, y_te)["accuracy"] == pytest.approx(metrics["accuracy"])


def test_multiclass_digits_with_numpy_mlp(tmp_path):
    cfg = TrainConfig(model="numpy-mlp", dataset="digits", lr=0.01, epochs=40, batch_size=64, hidden=(32,), optimizer="adam",
                      out_dir=str(tmp_path), name="digits")
    out = run(cfg, save_plots=False)
    cfg2, model, (X_tr, X_te, y_tr, y_te), metrics = load_run(out)
    assert metrics["accuracy"] > 0.9 and "macro_f1" in metrics
    assert model.predict_proba(X_te).shape == (len(y_te), 10)
