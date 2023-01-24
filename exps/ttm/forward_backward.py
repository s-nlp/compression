import torch
import opt_einsum
import torch.nn as nn
import tntorch as tn

from functools import lru_cache, reduce
from operator import mul

## Forward

def forward(cores, x):
    input_dims = [core.shape[1] for core in cores]
    n_rows = reduce(mul, input_dims)
    batch_size = x.reshape(-1, n_rows).shape[0]
    x = x.reshape(batch_size, -1).T
    result = x.reshape(input_dims[0], -1)
    result = opt_einsum.contract('id,lior->ldor', result, cores[0])

    for d in range(1, len(cores)):
        result = result.reshape(input_dims[d], -1, cores[d].shape[0])
        result = opt_einsum.contract('idr,riob->dob', result, cores[d])

    return result.reshape(batch_size, -1)


## Backward

@lru_cache(maxsize=None)
def cached_einsum_expr(expr: str, *shapes):
    return opt_einsum.contract_expression(expr, *shapes, optimize='optimal')


def full_matrix_backward(forward_for_backward_fn):
    def wrapper(x, dy, cores):
        dx = forward_for_backward_fn([core.permute(0, 2, 1, 3) for core in cores], dy)
        dx = dx.reshape(x.shape)

        input_dims = tuple(core.shape[1] for core in cores)
        output_dims = tuple(core.shape[2] for core in cores)
        x = x.reshape((-1,) + input_dims)
        dy = dy.reshape((-1,) + output_dims)

        dW_expr, d_core_exprs = get_einsums(x.shape, dy.shape, tuple(core.shape for core in cores))
        dW = dW_expr(x, dy)
        dW = dW[None, ..., None]

        with opt_einsum.shared_intermediates():
            d_cores = [d_core_expr(dW, *cores[:i], *cores[i + 1:]) for i, d_core_expr in enumerate(d_core_exprs)]

        return (dx, *d_cores)

    return wrapper


@lru_cache(maxsize=None)
def get_einsums(x_shape, dy_shape, core_shapes):
    n_cores = len(core_shapes)
    in_dims = ''.join(opt_einsum.get_symbol(i) for i in range(n_cores))
    out_dims = ''.join(opt_einsum.get_symbol(n_cores + i) for i in range(n_cores))
    batch = opt_einsum.get_symbol(2 * n_cores)
    rs = [opt_einsum.get_symbol(2 * n_cores + 1 + i) for i in range(n_cores + 1)]
    core_einsums = [f'{r1}{d1}{d2}{r2}' for r1, d1, d2, r2 in zip(rs[:-1], in_dims, out_dims, rs[1:])]
    W_einsum = f'{rs[0]}{in_dims}{out_dims}{rs[-1]}'

    dW_expr = cached_einsum_expr(f'{batch}{in_dims},{batch}{out_dims}->{in_dims}{out_dims}', x_shape, dy_shape)
    d_core_exprs = []
    for i in range(n_cores):
        d_core_exprs.append(cached_einsum_expr(
            f'{",".join([W_einsum] + core_einsums[:i] + core_einsums[i + 1:])}->{core_einsums[i]}',
            (1,) + x_shape[1:] + dy_shape[1:] + (1,), *core_shapes[:i], *core_shapes[i + 1:]
        ))

    return dW_expr, d_core_exprs

## Whole module

def forward_backward_module(forward_fn, backward_fn):
    class Wrapper(torch.autograd.Function):
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

## Opt_einsum

def opt_tt_multiply(tt_matrix: tn.TTMatrix, tensor: torch.Tensor):
    """
    Changed to use opt_einsum.
    Multiply TTMatrix by any tensor of more than 1-way.
    For vectors, reshape them to matrix of shape 1 x I
    returns: torch.Tensor of shape b x num_cols(tt_matrix)
    """

    assert len(tensor.shape) > 1

    rows = torch.prod(tt_matrix.input_dims)
    b = tensor.reshape(-1, rows).shape[0]
    tensor = tensor.reshape(b, -1).T
    result = tensor.reshape(tt_matrix.input_dims[0], -1)
    result = opt_einsum.contract('id,lior->ldor', result, tt_matrix.cores[0], optimize='optimal')

    for d in range(1, tt_matrix.d):
        result = result.reshape(tt_matrix.input_dims[d], -1, tt_matrix.cores[d].shape[0])
        result = opt_einsum.contract('idr,riob->dob', result, tt_matrix.cores[d], optimize='optimal')

    return result.reshape(b, -1)
