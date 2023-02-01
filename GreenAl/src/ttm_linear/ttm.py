import operator
from functools import lru_cache, reduce
from typing import Union, List, Tuple, Sequence

import numpy as np
import torch as t
import torch.jit
from torch.utils.checkpoint import checkpoint
from opt_einsum.contract import ContractExpression
from torch import nn
import opt_einsum

from .utils import cached_einsum, cached_einsum_expr
from .full_matrix_backward import full_matrix_backward
from .full_einsum_backward import full_einsum_backward
from .super_full_einsum_backward import super_full_einsum_backward


def einsum_forward(cores, x):
    return build_forward_expr(tuple(core.shape for core in cores), x.shape)(*cores, x)


def by_hands_forward(cores, x):
    n_cores = len(cores)

    dims = ''.join(opt_einsum.get_symbol(i) for i in range(n_cores))
    r1 = opt_einsum.get_symbol(n_cores)
    r2 = opt_einsum.get_symbol(n_cores + 1)
    new_dim = opt_einsum.get_symbol(n_cores + 2)
    batch = opt_einsum.get_symbol(n_cores + 3)

    x = x[..., None]
    for i, core in enumerate(cores):
        x = cached_einsum(f'{batch}{dims}{r1},{r1}{dims[i]}{new_dim}{r2}->{batch}{dims[:i]}{new_dim}{dims[i + 1:]}{r2}', x, core)
    return t.squeeze(x, -1)


def with_self_checkpoint(forward_fn):
    def wrapper(cores, x):
        return checkpoint(forward_fn, cores, x)
    return wrapper


def forward_backward_module(forward_fn, backward_fn):
    class Wrapper(t.autograd.Function):
        @staticmethod
        def forward(ctx, x, *cores):
            ctx.save_for_backward(*cores, x)
            return forward_fn(cores, x)

        @staticmethod
        def backward(ctx, dy):
            *cores, x = ctx.saved_tensors
            return backward_fn(x, dy, cores)

    def wrapper(cores, x):
        return Wrapper.apply(x, *cores)

    return wrapper


class TTM(nn.Module):
    def __init__(
        self, dims: List[Tuple[int, int]], ranks: Union[int, List[int]], forward_fn=einsum_forward
    ):
        super().__init__()

        if isinstance(ranks, int):
            ranks = [ranks] * (len(dims) - 1)

        self.dims = dims
        self.dims_in = tuple(d for d, _ in self.dims)
        self.dims_out = tuple(d for _, d in self.dims)
        self.dim_in = reduce(operator.mul, [dim[0] for dim in self.dims])
        self.dim_out = reduce(operator.mul, [dim[1] for dim in self.dims])

        self.tt = PlainTTMContainer(self.dims, ranks)
        # self.tt = SpectralTTMContainer(self.dims, rank)
        # print(f'Total number of parameters in tt: {sum(param.numel() for param in self.tt.parameters())}')

        self.forward_fn = forward_fn

    def forward(self, x):
        x_reshaped = x.reshape(x.shape[:1] + self.dims_in)
        y_reshaped = self.forward_fn(self.tt.cores, x_reshaped)
        return y_reshaped.reshape(x_reshaped.shape[0], self.dim_out)

    @property
    def full_tensor(self) -> t.Tensor:
        cores = self.tt.cores
        return build_full_tensor_expr(
            tuple(core.shape for core in cores)
        )(*cores).reshape(self.dim_in, self.dim_out)


@lru_cache(maxsize=None)
def build_full_tensor_expr(core_shapes: Tuple[Tuple[int]]) -> ContractExpression:
    n_cores = len(core_shapes)
    ds = [(opt_einsum.get_symbol(i * 2 + 0), opt_einsum.get_symbol(i * 2 + 1)) for i in range(n_cores)]
    rs = [opt_einsum.get_symbol(n_cores * 2 + i) for i in range(n_cores + 1)]

    left = ','.join(f'{r1}{d1}{d2}{r2}' for r1, (d1, d2), r2 in zip(rs[:-1], ds, rs[1:]))
    right = ''.join(d for ds in zip(*ds) for d in ds)

    return cached_einsum_expr(f'{left}->{right}', *core_shapes)


class PlainTTMContainer(nn.Module):
    REQUIRED_ELEMENT_STD = 0.02  # see initialization of transformers.torch_utils.Conv1D weight matrix

    def __init__(self, dims: List[Tuple[int, int]], rank_or_ranks: Union[int, List[int]]):
        super().__init__()

        ranks = rank_or_ranks if isinstance(rank_or_ranks, list) else [rank_or_ranks] * (len(dims) - 1)
        assert len(dims) == len(ranks) + 1

        for i, (d1, d2) in enumerate(dims):
            r1 = ranks[i - 1] if i else 1
            r2 = ranks[i] if i < len(ranks) else 1
            if d1 * d2 * r1 < r2:
                ranks[i] = d1 * d2 * r1

        self.cores = nn.ParameterList(
            [
                # nn.Parameter(t.randn(r1, dim1, dim2, r2) / np.sqrt(dim1 * dim2 * r1), requires_grad=True)
                nn.Parameter(t.randn(r1, dim1, dim2, r2) / np.sqrt(r1), requires_grad=True)
                for (dim1, dim2), r1, r2 in zip(dims, [1] + ranks, ranks + [1])
            ]
        )
        self.cores[-1].data *= self.REQUIRED_ELEMENT_STD
        self.orthogonalize()

    @property
    def n_cores(self):
        return len(self.cores)

    def orthogonalize(self):
        with t.no_grad():
            state = t.eye(1).to(self.cores[0])
            for core_param in self.cores[:-1]:
                core = cached_einsum('ij,jklm->iklm', state, core_param)
                shape = core.shape
                core = core.reshape(-1, shape[-1])
                q, r = t.linalg.qr(core)
                core_param.data.copy_(q.reshape(shape))
                state = r
            self.cores[-1].data.copy_(cached_einsum('ij,jklm->iklm', state, self.cores[-1]))


@lru_cache(maxsize=None)
def build_forward_expr(core_shapes: Tuple[Tuple[int]], x_shape: Tuple[int]) -> ContractExpression:
    n_cores = len(core_shapes)

    ein_str_parts = []

    for i in range(n_cores):
        r1 = opt_einsum.get_symbol(n_cores * 2 + i)
        dim_in = opt_einsum.get_symbol(i * 2)
        dim_out = opt_einsum.get_symbol(i * 2 + 1)
        r2 = opt_einsum.get_symbol(n_cores * 2 + i + 1)

        ein_str_parts.append(f'{r1}{dim_in}{dim_out}{r2}')

    batch_dim = n_cores * 3 + 1

    in_dims = [batch_dim] + (2 * np.arange(n_cores)).tolist()
    ein_str_parts.append(''.join(opt_einsum.get_symbol(dim) for dim in in_dims))

    out_dims = [batch_dim] + (2 * np.arange(n_cores) + 1).tolist()
    ein_str_parts.append(''.join(opt_einsum.get_symbol(dim) for dim in out_dims))

    ein_str = f'{",".join(ein_str_parts[:-1])}->{ein_str_parts[-1]}'
    return cached_einsum_expr(ein_str, *core_shapes, x_shape)