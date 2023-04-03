from functools import partial
from typing import Sequence

import torch as T


def svd_truncated(mat: T.Tensor, rank: int):
    lvecs, svals, rvecs = T.linalg.svd(mat)
    return lvecs[:, :rank], svals[:rank], rvecs[:rank, :].T


def ttd(ten: T.Tensor, rank: Sequence[int], noiters: int = 1000,
        method: str = 'tsvd') -> Sequence[T.Tensor]:
    """Function ttd implements tensor-train decomposition.
    """
    if ten.ndim + 1 != len(rank):
        raise ValueError
    if rank[0] != 1 or rank[-1] != 1:
        raise ValueError

    if method == 'svd':
        factorize = svd_truncated
    elif method == 'tsvd':
        factorize = partial(T.svd_lowrank, niter=noiters)
    else:
        raise ValueError(f'Unknown method: {method}.')

    cores = []
    shape = ten.shape

    # Iterate over shape of cores and split off core from tensor.
    for core_shape in zip(rank, shape, rank[1:]):
        # breakpoint()
        # Matricization of tensor over the first two axes.
        mat = ten.reshape(core_shape[0] * core_shape[1], -1)
        # Singlular Value Decomposition (SVD).
        lvecs, svals, rvecs = factorize(mat, core_shape[2])
        # Reshape core and rest of tensor.
        core = lvecs * svals[None, :]
        core = core.reshape(core_shape)
        cores.append(core)
        # Use right vectors as a tensor itself.
        ten = rvecs.T

    return cores
