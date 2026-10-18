import numpy as np

from baked_engine.core.imageops import atrous_denoise, dilate, encode


def test_dilate_fills_everything_and_keeps_masked():
    img = np.zeros((32, 32, 3))
    mask = np.zeros((32, 32), bool)
    mask[10:14, 10:14] = True
    img[mask] = [1.0, 0.5, 0.25]
    out = dilate(img, mask)
    assert np.allclose(out[mask], [1.0, 0.5, 0.25])
    assert np.allclose(out, [1.0, 0.5, 0.25])


def test_denoise_smooths_noise_but_not_across_charts():
    rng = np.random.default_rng(0)
    h, w = 32, 64
    yy, xx = np.mgrid[0:h, 0:w]
    # two charts side by side in the atlas, far apart in the world, different brightness
    left = xx < 32
    pos = np.stack([xx * 0.1 + np.where(left, 0, 100), yy * 0.1, np.zeros_like(xx, float)], -1).astype(float)
    nrm = np.zeros((h, w, 3))
    nrm[..., 2] = 1
    truth = np.where(left, 1.0, 0.2)[..., None] * np.ones(3)
    noisy = truth + rng.normal(0, 0.2, (h, w, 1))
    var = np.full((h, w), 0.04)
    mask = np.ones((h, w), bool)
    out = atrous_denoise(noisy, var, mask, pos, nrm, np.full((h, w), 0.1))
    err_before = np.abs(noisy - truth)[:, 4:28].mean()
    err_after = np.abs(out - truth)[:, 4:28].mean()
    assert err_after < err_before * 0.35
    # the boundary column of the bright chart is not pulled towards the dark one
    assert abs(out[:, 31].mean() - 1.0) < 0.08
    assert abs(out[:, 32].mean() - 0.2) < 0.08


def test_encode_ranges():
    x = np.array([[0.0, 0.18, 1.0, 100.0]]).T * np.ones(3)
    for view in ("STANDARD", "ACES", "AGX"):
        y = encode(x, view)
        assert (y >= 0).all() and (y <= 1).all()
        assert (np.diff(y[:, 0]) >= 0).all()
