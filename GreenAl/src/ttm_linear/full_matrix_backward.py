from functools import lru_cache
import opt_einsum

from .utils import cached_einsum_expr


def full_matrix_backward(forward_for_backward_fn):
    def wrapper(x, dy, cores):
        dx = forward_for_backward_fn([core.permute(0, 2, 1, 3) for core in cores], dy)

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
