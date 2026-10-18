"""Atlas image operations: guided a-trous denoising, gutter dilation, colour encoding."""

from __future__ import annotations

import numpy as np

from .scene import luminance

_B3 = np.array([1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16])


def _shift(a, dx, dy, fill=0):
    """out[y, x] = a[y + dy, x + dx] (with ``fill`` outside)."""
    h, w = a.shape[:2]
    out = np.full_like(a, fill)
    ys = slice(max(0, -dy), min(h, h - dy))
    xs = slice(max(0, -dx), min(w, w - dx))
    ys2 = slice(max(0, dy), min(h, h + dy))
    xs2 = slice(max(0, dx), min(w, w + dx))
    out[ys, xs] = a[ys2, xs2]
    return out


def atrous_denoise(color, variance, mask, position, normal, texel_size, iterations=5, sigma_l=4.0, sigma_n=64.0, sigma_p=1.0):
    """Edge-avoiding a-trous wavelet filter (SVGF style, spatial only) in atlas space.

    color: (H, W, 3) demodulated lighting; variance: (H, W) of the luminance mean;
    mask: (H, W) bool valid texels; position/normal: (H, W, 3) world-space guides;
    texel_size: (H, W) world size of a texel. Neighbours from unrelated charts are
    rejected by the position term, so the filter never mixes atlas islands.
    """
    c = np.where(mask[..., None], color, 0).astype(np.float64)
    var = np.where(mask, variance, 0).astype(np.float64)
    m = mask.astype(np.float64)
    ts = np.maximum(texel_size, 1e-12)
    for it in range(iterations):
        step = 1 << it
        lum = luminance(c)
        # 3x3 gaussian of variance for a stabler luminance edge-stop
        vblur = np.zeros_like(var)
        wsum = np.zeros_like(var)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                k = (2 - abs(dx)) * (2 - abs(dy)) / 16.0
                vblur += _shift(var, dx, dy) * k * _shift(m, dx, dy)
                wsum += k * _shift(m, dx, dy)
        vblur = vblur / np.maximum(wsum, 1e-12)
        sl = sigma_l * np.sqrt(np.maximum(vblur, 0)) + 1e-6
        acc = np.zeros_like(c)
        accv = np.zeros_like(var)
        accw = np.zeros_like(var)
        for j, ky in enumerate(_B3):
            dy = (j - 2) * step
            for i, kx in enumerate(_B3):
                dx = (i - 2) * step
                k = kx * ky
                cn = _shift(c, dx, dy)
                vn = _shift(var, dx, dy)
                mn = _shift(m, dx, dy)
                pn = _shift(position, dx, dy)
                nn = _shift(normal, dx, dy)
                dist = np.linalg.norm(pn - position, axis=2) / (ts * step * 2.5 * sigma_p)
                wp = np.exp(-dist * dist)
                wn = np.maximum(np.einsum("ijk,ijk->ij", nn, normal), 0) ** sigma_n
                wl = np.exp(-np.abs(luminance(cn) - lum) / sl)
                w = k * wp * wn * wl * mn
                acc += cn * w[..., None]
                accv += vn * w * w
                accw += w
        good = accw > 1e-12
        c = np.where(good[..., None], acc / np.maximum(accw, 1e-12)[..., None], c)
        var = np.where(good, accv / np.maximum(accw * accw, 1e-24), var)
        c = np.where(mask[..., None], c, 0)
    return c


def dilate(image, mask, iterations=None):
    """Fill pixels outside ``mask`` from their neighbours.

    A few rings of exact neighbour averaging protect bilinear/mip sampling at chart
    borders, then a pull-push pyramid fills the rest of the image smoothly.
    """
    img = np.where(mask[..., None], image, 0).astype(np.float64)
    m = mask.astype(np.float64)
    rings = 4 if iterations is None else iterations
    for _ in range(rings):
        acc = np.zeros_like(img)
        cnt = np.zeros_like(m)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                acc += _shift(img, dx, dy) * _shift(m, dx, dy)[..., None]
                cnt += _shift(m, dx, dy)
        new = (m == 0) & (cnt > 0)
        img[new] = acc[new] / cnt[new][:, None]
        m = np.where(new, 1.0, m)
    if (m == 0).any():
        img = _pull_push(img, m)
    return img


def _pull_push(img, m):
    h, w = m.shape
    if h <= 1 and w <= 1 or m.sum() == 0:
        fill = img[m > 0].mean(axis=0) if (m > 0).any() else np.zeros(img.shape[2])
        return np.where(m[..., None] > 0, img, fill)
    hp, wp = h + (h & 1), w + (w & 1)
    ip = np.zeros((hp, wp, img.shape[2]))
    mp = np.zeros((hp, wp))
    ip[:h, :w] = img * m[..., None]
    mp[:h, :w] = m
    ms = mp.reshape(hp // 2, 2, wp // 2, 2).sum(axis=(1, 3))
    isum = ip.reshape(hp // 2, 2, wp // 2, 2, -1).sum(axis=(1, 3))
    coarse = np.where(ms[..., None] > 0, isum / np.maximum(ms, 1e-12)[..., None], 0)
    cm = (ms > 0).astype(np.float64)
    coarse = _pull_push(coarse, cm)
    up = np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1)[:h, :w]
    return np.where(m[..., None] > 0, img, up)


# ---------------------------------------------------------------------------
# colour encoding


def srgb_oetf(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def aces_fitted(x):
    """Narkowicz ACES approximation, applied per channel."""
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    x = np.maximum(x, 0)
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)


_AGX_IN = np.array(
    [
        [0.842479062253094, 0.0423282422610123, 0.0423756549057051],
        [0.0784335999999992, 0.878468636469772, 0.0784336],
        [0.0792237451477643, 0.0791661274605434, 0.879142973793104],
    ]
)
_AGX_OUT = np.array(
    [
        [1.19687900512017, -0.0528968517574562, -0.0529716355144438],
        [-0.0980208811401368, 1.15190312990417, -0.0980434501171241],
        [-0.0990297440797205, -0.0989611768448433, 1.15107367264116],
    ]
)


def agx_like(x):
    """AgX base view (Wrensch's fit): inset, log2 encode, sigmoid, outset. Returns display-linear."""
    v = np.maximum(np.asarray(x, dtype=np.float64), 0) @ _AGX_IN
    lo, hi = -12.47393, 4.026069
    v = (np.clip(np.log2(np.maximum(v, 1e-10)), lo, hi) - lo) / (hi - lo)
    v2 = v * v
    v4 = v2 * v2
    v = 15.5 * v4 * v2 - 40.14 * v4 * v + 31.96 * v4 - 6.868 * v2 * v + 0.4298 * v2 + 0.1191 * v - 0.00232
    v = v @ _AGX_OUT
    return np.clip(np.power(np.maximum(v, 0), 2.2), 0, 1)


def encode(img, view="STANDARD", exposure=0.0):
    """Scene-linear RGB -> display sRGB in [0, 1]."""
    x = np.asarray(img, dtype=np.float64)[..., :3] * (2.0 ** exposure)
    if view == "ACES":
        x = aces_fitted(x)
    elif view == "AGX":
        x = agx_like(x)
    return srgb_oetf(x)
