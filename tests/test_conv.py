import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nnlab.conv import (
    DECK_IMAGE_6x6, POOL_EXAMPLE_4x4, VERTICAL_EDGE, avg_pool2d, conv2d_single, conv2d_volume, conv_output_size,
    describe_cnn, max_pool2d, zero_pad,
)

EXPECTED_4x4 = np.array([[-5, -4, 0, 8], [-10, -2, 2, 3], [0, -2, -4, -7], [-3, -2, -3, -16]], dtype=float)
EXPECTED_PAD1 = np.array(
    [[-5, -5, -6, -1, 6, 10], [-12, -5, -4, 0, 8, 11], [-13, -10, -2, 2, 3, 11],
     [-10, 0, -2, -4, -7, 10], [-7, -3, -2, -3, -16, 12], [-6, 0, -2, 1, -9, 5]], dtype=float,
)


def test_deck_example_p166_181():
    assert np.array_equal(conv2d_single(DECK_IMAGE_6x6, VERTICAL_EDGE), EXPECTED_4x4)


def test_deck_padding_example_p184():
    assert np.array_equal(conv2d_single(DECK_IMAGE_6x6, VERTICAL_EDGE, pad=1), EXPECTED_PAD1)
    assert zero_pad(DECK_IMAGE_6x6, 2).shape == (10, 10)


def test_output_size_formula_p185_190():
    assert conv_output_size(6, 3) == 4
    assert conv_output_size(6, 3, p=1) == 6
    assert (conv_output_size(39, 3), conv_output_size(37, 5, s=2), conv_output_size(17, 5, s=2)) == (37, 17, 7)
    with pytest.raises(ValueError):
        conv_output_size(2, 3)


@pytest.mark.parametrize("n,f,p,s", [(6, 3, 0, 1), (6, 3, 1, 1), (7, 3, 0, 2), (28, 5, 2, 1), (32, 5, 0, 1), (13, 3, 1, 2)])
def test_matches_torch_conv2d(n, f, p, s, rng):
    img = rng.normal(size=(n, n))
    filt = rng.normal(size=(f, f))
    ours = conv2d_single(img, filt, stride=s, pad=p)
    theirs = F.conv2d(torch.tensor(img)[None, None], torch.tensor(filt)[None, None], stride=s, padding=p)[0, 0].numpy()
    assert ours.shape == theirs.shape == (conv_output_size(n, f, p, s),) * 2
    assert np.allclose(ours, theirs)          # torch "convolution" = cross-correlation = นิยามในสไลด์


def test_volume_conv_matches_torch(rng):
    vol = rng.normal(size=(6, 6, 3))                    # (H, W, C) ตามสไลด์
    filters = rng.normal(size=(3, 3, 3, 2))             # (f, f, C, n_F)
    bias = rng.normal(size=2)
    ours = conv2d_volume(vol, filters, bias, stride=1, pad=1)
    assert ours.shape == (6, 6, 2)                      # สไลด์ p.189: 6×6×3 * 3×3×3 (2 filters, p=1) → 6×6×2
    x = torch.tensor(vol).permute(2, 0, 1)[None]        # → (N=1, C, H, W)
    w = torch.tensor(filters).permute(3, 2, 0, 1)       # → (n_F, C, f, f)
    theirs = F.conv2d(x, w, torch.tensor(bias), padding=1)[0].permute(1, 2, 0).numpy()
    assert np.allclose(ours, theirs)


def test_max_pool_examples_p193_197():
    assert max_pool2d(POOL_EXAMPLE_4x4, f=2, s=2).tolist() == [[9, 2], [6, 3]]
    vol = np.stack([POOL_EXAMPLE_4x4, POOL_EXAMPLE_4x4 * 2], axis=-1)     # 4×4×2
    out = max_pool2d(vol, 2, 2)
    assert out.shape == (2, 2, 2) and out[..., 1].tolist() == [[18, 4], [12, 6]]
    assert max_pool2d(np.ones((5, 5, 2)), f=3, s=1).shape == (3, 3, 2)    # สไลด์ p.197: 5×5×2 → 3×3×2
    assert avg_pool2d(POOL_EXAMPLE_4x4)[0, 0] == pytest.approx((1 + 3 + 2 + 9) / 4)


def test_describe_cnn_lenet_p199_200():
    rows = describe_cnn(
        (32, 32, 3),
        [{"type": "conv", "f": 5, "n_f": 6}, {"type": "pool"}, {"type": "conv", "f": 5, "n_f": 16}, {"type": "pool"},
         {"type": "fc", "units": 120}, {"type": "fc", "units": 84}, {"type": "fc", "units": 10}],
    )
    shapes = [r["shape"] for r in rows]
    assert shapes[:5] == [(32, 32, 3), (28, 28, 6), (14, 14, 6), (10, 10, 16), (5, 5, 16)]
    assert (400,) in shapes and shapes[-3:] == [(120,), (84,), (10,)]
    fc3 = next(r for r in rows if r["layer"] == "FC1")
    assert fc3["params"] == 400 * 120 + 120                                  # W[3] = 120×400 (สไลด์ p.199)


def test_describe_cnn_simple_p190():
    rows = describe_cnn((39, 39, 3), [{"type": "conv", "f": 3, "n_f": 10}, {"type": "conv", "f": 5, "s": 2, "n_f": 20},
                                      {"type": "conv", "f": 5, "s": 2, "n_f": 40}, {"type": "flatten"}])
    assert [r["shape"] for r in rows] == [(39, 39, 3), (37, 37, 10), (17, 17, 20), (7, 7, 40), (1960,)]
