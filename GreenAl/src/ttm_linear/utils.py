from functools import lru_cache

from opt_einsum import contract_expression


@lru_cache(maxsize=None)
def cached_einsum_expr(expr: str, *shapes):
    return contract_expression(expr, *shapes, optimize='optimal')


def cached_einsum(expr: str, *args):
    return cached_einsum_expr(expr, *[arg.shape for arg in args])(*args)
