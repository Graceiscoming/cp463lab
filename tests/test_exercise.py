import numpy as np

from nnlab.exercise import check, check_close, check_shape, reset, summary


def _todo():
    raise NotImplementedError("ยังไม่ได้ทำ")


def test_states_never_raise(capsys):
    reset()
    assert check("pass", lambda: True) is True
    assert check("fail", lambda: False, hint="h") is False
    assert check("todo", _todo) is False
    assert check("ellipsis", lambda: ...) is False
    assert check("error", lambda: 1 / 0, hint="อย่าหารศูนย์") is False
    out = capsys.readouterr().out
    assert "✓ pass: PASS" in out and "✗ fail: FAIL" in out and "ยังไม่ได้ทำ" in out and "ZeroDivisionError" in out and "คำใบ้" in out
    counts = summary()
    assert counts == {"pass": 1, "fail": 1, "todo": 2, "error": 1}


def test_check_close_and_shape(capsys):
    reset()
    assert check_close("close", lambda: np.array([1.0, 2.0]), [1.0, 2.0000001])
    assert not check_close("wrong", lambda: [1.0, 2.0], [1.0, 3.0])
    assert not check_close("shape", lambda: [1.0], [1.0, 2.0])
    assert check_shape("shape ok", lambda: np.zeros((2, 3)), (2, 3))
    assert not check_shape("shape bad", lambda: np.zeros((3, 2)), (2, 3))
    assert check("array all", lambda: np.array([True, True]))
    out = capsys.readouterr().out
    assert "ได้ [1. 2.] คาด [1. 3.]" in out and "ได้ shape (3, 2) คาด (2, 3)" in out
