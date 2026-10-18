"""Sample generation: randomly shifted R2 low-discrepancy points for the first dimensions."""

from __future__ import annotations

import numpy as np

_G = 1.32471795724474602596  # plastic number
_A1 = 1.0 / _G
_A2 = 1.0 / (_G * _G)


class Sampler:
    """Per-point stratified sampler.

    ``pair(i, dim)`` returns a (n, 2) array for sample index ``i`` of dimension pair ``dim``.
    The first ``qmc_pairs`` dimension pairs use an R2 sequence with a per-point Cranley-Patterson
    shift and a per-dimension index permutation; further dimensions are independent uniforms.
    """

    def __init__(self, n, spp, seed=0, qmc_pairs=4):
        self.n = n
        self.spp = max(int(spp), 1)
        self.rng = np.random.default_rng(seed)
        self.qmc_pairs = qmc_pairs
        self.shift = self.rng.random((qmc_pairs, n, 2))
        # index permutations: i -> (a * i + b) mod spp with a coprime to spp
        self.perm = []
        for _ in range(qmc_pairs):
            while True:
                a = int(self.rng.integers(1, max(self.spp, 2)))
                if np.gcd(a, self.spp) == 1:
                    break
            self.perm.append((a, int(self.rng.integers(0, self.spp))))

    def pair(self, i, dim, idx=None):
        n = self.n if idx is None else len(idx)
        if dim < self.qmc_pairs:
            a, b = self.perm[dim]
            j = (a * i + b) % self.spp + 1
            base = np.array([(j * _A1) % 1.0, (j * _A2) % 1.0])
            sh = self.shift[dim] if idx is None else self.shift[dim][idx]
            return (base[None, :] + sh) % 1.0
        return self.rng.random((n, 2))

    def uniform(self, n):
        return self.rng.random(n)
