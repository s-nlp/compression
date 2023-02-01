from functools import lru_cache
import opt_einsum

from .utils import cached_einsum_expr


def full_einsum_backward(forward_for_backward_fn):
    def wrapper(x, dy, cores):
        dx = forward_for_backward_fn([core.permute(0, 2, 1, 3) for core in cores], dy)
        x = x[..., None]
        dy = dy[..., None]

        d_core_exprs = get_einsums(x.shape, dy.shape, tuple(core.shape for core in cores))
        with opt_einsum.shared_intermediates():
            d_cores = [d_core_expr(x, dy, *cores[:i], *cores[i + 1:]) for i, d_core_expr in enumerate(d_core_exprs)]

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
    x_einsum = f'{batch}{in_dims}{rs[0]}'
    dy_einsum = f'{batch}{out_dims}{rs[-1]}'

    d_core_exprs = []
    for i in range(n_cores):
        d_core_exprs.append(cached_einsum_expr(
            f'{",".join([x_einsum, dy_einsum] + core_einsums[:i] + core_einsums[i + 1:])}->{core_einsums[i]}',
            x_shape, dy_shape, *core_shapes[:i], *core_shapes[i + 1:]
        ))

    return d_core_exprs
