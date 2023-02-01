from functools import lru_cache
from typing import List, Tuple

import torch as t
from torch import nn
from torch.nn.functional import pad

from .full_matrix_backward import full_matrix_backward
from .ttm import TTM, forward_backward_module, einsum_forward


class TTMLinearBase(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        if bias:
            self.bias = nn.Parameter(t.zeros(dim_out))
        else:
            self.bias = None

    def forward(self, x: t.tensor):
        x_shape = x.shape
        x = x.view(-1, x.shape[-1])
        x = pad(x, (0, self.ttm.dim_in - self.dim_in, 0, 0))
        x = self.ttm(x)
        x = x[:, :self.dim_out]
        if self.bias is not None:
            x = x + self.bias
        x = x.view(*x_shape[:-1], -1)
        return x


class FactorizationTTMLinear(TTMLinearBase):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        *,
        rank: int,
        max_core_dim_product: int,
        reverse_factorization: bool = False,
        strictly_less_or_equal: bool = False,
        large_first: bool = False,
        large_last: bool = False,
        first_last_mix: bool = False,
    ):
        super().__init__(dim_in, dim_out)

        self.dims = prepare_core_shapes(
            dim_in, dim_out, max_core_dim_product, rank,
            reverse_factorization, strictly_less_or_equal, large_first, large_last, first_last_mix
        )

        self.ttm = TTM(self.dims, rank, forward_backward_module(einsum_forward, full_matrix_backward(einsum_forward)))


@lru_cache(maxsize=None)
def prepare_core_shapes(
    dim_in: int, dim_out: int, max_core_dim_product: int, rank: int,
    reverse_factorization: bool,
    strictly_less_or_equal: bool,
    large_first: bool,
    large_last: bool,
    first_last_mix: bool,
) -> List[Tuple[int, int]]:
    original_linear_size = dim_in * dim_out

    print('-------------------------------------')
    print(f'TTM-Linear required dimensions: dim_in={dim_in}, dim_out={dim_out}, rank={rank}, max_dim={max_core_dim_product}')
    # increase dimensions in such a way, that factorization have a lot of small factors
    dim_in = best_approx(dim_in)
    dim_out = best_approx(dim_out)
    print(f'    after best_approx: dim_in={dim_in}, dim_out={dim_out}')

    # factorization
    dims_in = tuple(factorize(dim_in))  # [::-1]
    dims_out = tuple(factorize(dim_out))  # [::-1]
    print(f'    dim_in factorization:  {dims_in}')
    print(f'    dim_out factorization: {dims_in}')
    if reverse_factorization:
        dims_in = dims_in[::-1]
        dims_out = dims_out[::-1]

    # padding
    dims_in += (1,) * (len(dims_out) - len(dims_in))
    dims_out += (1,) * (len(dims_in) - len(dims_out))
    # union
    dims = list(zip(dims_in, dims_out))
    # first-last mix optimization
    # 1, 2, 3, 4, ... -> 1, 3, 5, ..., 6, 4, 2
    if first_last_mix:
        dims = dims[::2] + dims[1::2][::-1]
    # shrinking
    print(f'    dims before shrink:  {dims}')
    dims = shrink(dims, max_core_dim_product, rank, strictly_less_or_equal, large_first, large_last)
    print(f'    final TTM dims:  {dims}')

    ranks = [1] + [rank] * (len(dims) - 1) + [1]
    total_ttm_params = sum(r1 * d1 * d2 * r2 for r1, (d1, d2), r2 in zip(ranks[:-1], dims, ranks[1:]))
    compression = total_ttm_params / original_linear_size
    print(f'    Original linear params: {original_linear_size}, ttm params: {total_ttm_params} (x{compression:.3f})')
    print('-------------------------------------')

    return dims


def shrink(
    dims: List[Tuple[int, int]], max_core_dim_product: int, rank: int, strictly_less_or_equal: bool, large_first: bool, large_last: bool
) -> List[Tuple[int, int]]:
    result_dims = []

    if large_first:
        shape, dims = consume(dims, max_core_dim_product * rank, strictly_less_or_equal)
        result_dims.append(shape)
    if large_last and dims:
        last_dim, dims = consume(dims[::-1], max_core_dim_product * rank, strictly_less_or_equal)
        dims = dims[::-1]
    else:
        large_last = False
    while dims:
        shape, dims = consume(dims, max_core_dim_product, strictly_less_or_equal)
        result_dims.append(shape)
    if large_last:
        result_dims.append(last_dim)

    return result_dims


def consume(
    dims: List[Tuple[int, int]], max_core_dim_product: int, strictly_less_or_equal: bool
) -> Tuple[Tuple[int, int], List[Tuple[int, int]]]:
    assert len(dims) > 0

    result_d1, result_d2 = dims[0]
    for i, (d1, d2) in enumerate(dims[1:], start=1):
        if result_d1 * result_d2 * d1 * d2 > max_core_dim_product and strictly_less_or_equal:
            return (result_d1, result_d2), dims[i:]
        result_d1 *= d1
        result_d2 *= d2
        if result_d1 * result_d2 >= max_core_dim_product:
            return (result_d1, result_d2), dims[i + 1:]

    return (result_d1, result_d2), dims[len(dims):]


def best_approx(n: int, max_factor: int = 3) -> int:
    assert n > 0

    if n == 1:
        return 1

    n_factors = log2(n, False)
    for i in range(n, 2 * n):
        factors = factorize(i)
        if len(factors) <= n_factors and all([f <= max_factor for f in factors]):
            return i


def factorize(n: int) -> List[int]:
    assert n > 0

    if n == 1:
        return [1]
    factors = []
    i = 2
    while i * i <= n:
        while n % i == 0:
            factors.append(i)
            n //= i
        i += 1
    if n != 1:
        factors.append(n)
    return factors


def log2(n: int, assert_eq: bool) -> int:
    assert n > 0

    res = 0
    while 2 ** res < n:
        res += 1

    if assert_eq:
        assert 2 ** res == n

    return res
