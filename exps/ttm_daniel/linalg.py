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


def weighted_ttd(ten: T.Tensor, weights: T.Tensor, rank: Sequence[int], noiters: int = 1000,
        method: str = 'tsvd') -> Sequence[T.Tensor]:
    """Function weighted_ttd implements tensor-train decomposition with element weighting.
    """
    assert ten.shape == weights.shape, f"Expected weights and ten to have same shape, got {weights.shape} and {ten.shape}"

    weights = weights.to(ten.device)

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
        weights_mat = weights.reshape(core_shape[0] * core_shape[1], -1)
        diag = weights_mat.sum(-1)

        assert T.isfinite(diag).all()

        # Singlular Value Decomposition (SVD).
        lvecs, svals, rvecs = factorize(diag[:, None] * mat, core_shape[2])
        assert T.isfinite(lvecs).all()
        assert T.isfinite(svals).all()
        assert T.isfinite(rvecs).all()
        
        core = (1 / diag[:, None]) * lvecs * svals
        assert T.isfinite(core).all()
        # Reshape core and rest of tensor.
        core = core.reshape(core_shape)
        cores.append(core)
        # Use right vectors as a tensor itself.
        ten = rvecs.T
        
        weights = lvecs.T @ weights_mat
        weights = weights.T

    return cores


def other_weighted_ttd(ten: T.Tensor, weights: T.Tensor, rank: Sequence[int], noiters: int = 1000,
        method: str = 'tsvd') -> Sequence[T.Tensor]:
    """Function weighted_ttd implements tensor-train decomposition with element weighting.
    """
    assert ten.shape == weights.shape, f"Expected weights and ten to have same shape, got {weights.shape} and {ten.shape}"

    weights = weights.to(ten.device)

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
        weights_mat = weights.reshape(core_shape[0] * core_shape[1], -1)
        diag = weights_mat.sum(-1)

        assert T.isfinite(diag).all()

        # Singlular Value Decomposition (SVD).
        lvecs, svals, rvecs = factorize(diag[:, None] * mat, core_shape[2])
        assert T.isfinite(lvecs).all()
        assert T.isfinite(svals).all()
        assert T.isfinite(rvecs).all()
        
        core = (1 / diag[:, None]) * lvecs * svals
        assert T.isfinite(core).all()
        # Reshape core and rest of tensor.
        core = core.reshape(core_shape)
        cores.append(core)
        # Use right vectors as a tensor itself.
        ten = rvecs.T
        
        left_side, rank = lvecs.shape
        weights = weights_mat[:rank].T

    return cores
