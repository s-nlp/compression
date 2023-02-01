import torch as t
from torch import nn
from torch.nn.functional import pad

from .ttm2 import factorize, TTM



class TTMLinear(nn.Module):
    def __init__(self, d_in: int, d_out: int, rank: int):
        super().__init__()

        self.d_in = d_in
        self.d_in_ttm = best_approx(d_in)
        self.d_out = d_out
        self.d_out_ttm = best_approx(d_out)

        self.ttm = TTM(self.d_in_ttm, self.d_out_ttm, rank)

    def forward(self, x: t.tensor):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        x = pad(x, (0, 0, 0, self.d_in_ttm - self.d_in))
        x = self.ttm(x)
        x = x.reshape(shape[0], -1, self.d_out)
        return x[:, :self.d_out]

    def row_norms(self):
        return self.ttm.full_tensor()[:self.d_in, :self.d_out].norm(p=2, dim=0)


def best_approx(n: int, max_factor: int = 3):
    n_factors = log2(n, False)
    while True:
        factors = factorize(n)
        if len(factors) <= n_factors and all([f <= max_factor for f in factors]):
            return n
        n += 1


def log2(n: int, assert_eq: bool):
    res = 0
    while 2 ** res < n:
        res += 1

    if assert_eq:
        assert 2 ** res == n

    return res
