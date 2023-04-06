from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch as T
from opt_einsum import contract_expression
from opt_einsum.contract import ContractExpression

from .linalg import ttd
from .functional import compressed_linear_svd

__all__ = ('CompressedLinear', 
           'SVDCompressedLinear', 'FWSVDCompressedLinear'
           'TTCompressedLinear', 'FWTTCompressedLinear')


def chop(values, eps):
    """Function chop truncate spectrum (singular values) with given error or
    rank. The origin is taked from oseledets/ttpy.
    """
    if eps <= 0.0:
        return len(values)

    values_reversed = values.flip(0)
    cumulative = T.cumsum(abs(values_reversed)**2).flip(0)
    tail = sum(cumulative < eps**2)   # Number of elements in tail.
    return len(values) - tail


class CompressedLinear(T.nn.Module):
    """Class CompressedLinear is a parent module for any linear models with
    compressed weight representations.
    """

    @classmethod
    def from_linear(cls, linear: T.nn.Linear):
        raise NotImplementedError('This class method is abstract and '
                                  'it must be implemented in derived classes.')


class SVDCompressedLinear(CompressedLinear):
    """Class SVDCompressedLinear is a layer which represents a weight matrix of
    lineaer layer in factorized view.

    >>> linear_layer = T.nn.Linear(10, 20)
    >>> svd_layer = SVDCompressedLinear.from_linear(linear_layer, rank=5)
    """

    def __init__(self, factors: Tuple[T.Tensor, T.Tensor, T.Tensor],
                 bias: Optional[T.Tensor] = None):
        super().__init__()

        # We do not want track singular values so let's mix t into left and
        # right vectors.
        scale = T.sqrt(factors[1])

        # Store factors of W^T but build factors for W.
        self.lhs = T.nn.Parameter(factors[2].T * scale[None, :])
        self.rhs = T.nn.Parameter(factors[0].T * scale[:, None])

        self.bias = None
        if bias is not None:
            self.bias = T.nn.Parameter(bias)

        self.in_features = self.lhs.shape[0]
        self.out_features = self.rhs.shape[1]

    @classmethod
    def from_linear(cls, linear: T.nn.Linear, rank: Optional[int] = None,
                    tol: float = 1e-6):
        with T.no_grad():
            data = linear.weight.data
            lhs, vals, rhs = T.linalg.svd(data)
            if rank is None:
                raise NotImplementedError
            else:
                lhs = lhs[:, :rank]
                rhs = rhs[:rank, :]
                vals = vals[:rank]

            bias = None
            if linear.bias is not None:
                bias = T.clone(linear.bias.data)

        return SVDCompressedLinear((lhs, vals, rhs), bias)

    @classmethod
    def from_random(cls, in_features: int, out_features: int, rank: int,
                    bias: bool = True):
        lvecs = T.randn((out_features, rank))
        svals = T.ones(rank)
        rvecs = T.randn((rank, in_features))
        bias_term = None
        if bias:
            bias_term = T.randn(out_features)
        return SVDCompressedLinear((lvecs, svals, rvecs), bias_term)

    def forward(self, input: T.Tensor) -> T.Tensor:
        return compressed_linear_svd(input, self.lhs, self.rhs, self.bias)

class FWSVDCompressedLinear(CompressedLinear):
    """Class SVDCompressedLinear is a layer which represents a weight matrix of
    lineaer layer in factorized view.

    >>> linear_layer = T.nn.Linear(10, 20)
    >>> svd_layer = SVDCompressedLinear.from_linear(linear_layer, rank=5)
    """

    def __init__(self, factors: Tuple[T.Tensor, T.Tensor, T.Tensor], 
                 fisher_information: T.Tensor,
                 bias: Optional[T.Tensor] = None, ):
        super().__init__()

        # We do not want track singular values so let's mix t into left and
        # right vectors.
        scale = T.sqrt(factors[1])
        self.fisher_information = fisher_information#T.sqrt(fisher_information.sum(0))

        # Store factors of W^T but build factors for W.
        self.lhs = T.nn.Parameter(factors[2].T * scale[None, :])
        self.rhs = T.nn.Parameter(factors[0].T * scale[:, None])

        self.bias = None
        if bias is not None:
            self.bias = T.nn.Parameter(bias)

        self.in_features = self.lhs.shape[0]
        self.out_features = self.rhs.shape[1]

    @classmethod
    def from_linear(cls, linear: T.nn.Linear, fisher_information: T.Tensor,
                    rank: Optional[int] = None,
                    tol: float = 1e-6):
        with T.no_grad():
            data = linear.weight.data
            #FWSVD
            fisher_information = T.sqrt(fisher_information.sum(0))
            data = fisher_information.reshape(1, -1).to(data.device) * data
            lhs, vals, rhs = T.linalg.svd(data)
            if rank is None:
                raise NotImplementedError
            else:
                lhs = lhs[:, :rank]
                rhs = rhs[:rank, :]
                vals = vals[:rank]

            bias = None
            if linear.bias is not None:
                bias = T.clone(linear.bias.data)

        return FWSVDCompressedLinear((lhs, vals, rhs), fisher_information, bias)

    @classmethod
    def from_random(cls, in_features: int, out_features: int, rank: int,
                    bias: bool = True):
        lvecs = T.randn((out_features, rank))
        svals = T.ones(rank)
        rvecs = T.randn((rank, in_features))
        bias_term = None
        if bias:
            bias_term = T.randn(out_features)
        return FWSVDCompressedLinear((lvecs, svals, rvecs), bias_term)

    def forward(self, input: T.Tensor) -> T.Tensor:
        self.fisher_information = self.fisher_information.to(input.device)
        return compressed_linear_svd(input / self.fisher_information, 
                                     self.lhs, self.rhs, self.bias)


def factorize(value: int) -> Dict[int, int]:
    """Function factorize factorizes an interger on prime numbers.

    :param value: Interger number to factorize.
    :return: A mapping from factor to its multiplicity.
    """
    primes = {}

    def exhaust(value: int, prime: int) -> int:
        """Divide :value: by :prime: until it is possible.
        """
        count = 0
        while value % prime == 0:
            value //= prime
            count += 1
        if count:
            primes[prime] = count
        return value

    # There is no primes for such numbers.
    if value < 2:
        return primes

    # Find all primes 2 in the number.
    value = exhaust(value, prime=2)

    # Now we can try all numbers from 3 to sqrt with stride 2.
    prime = 3
    while prime**2 <= value:
        value = exhaust(value, prime)
        prime += 2

    # If a remained is not lesser than 3 than it means that this number a prime
    # itself.
    if value > 2:
        primes[value] = 1

    return primes


def make_contraction(shape, rank, batch_size=32,
                     seqlen=512) -> ContractExpression:
    ndim = len(rank) - 1
    row_shape, col_shape = shape

    # Generate all contraction indexes.
    row_ix, col_ix = np.arange(2 * ndim).reshape(2, ndim)
    rank_ix = 2 * ndim + np.arange(ndim + 1)
    batch_ix = 4 * ndim  # Zero-based index.

    # Order indexes of cores.
    cores_ix = np.column_stack([rank_ix[:-1], row_ix, col_ix, rank_ix[1:]])
    cores_shape = zip(rank[:-1], row_shape, col_shape, rank[1:])

    # Order indexes of input (contraction by columns: X G_1 G_2 ... G_d).
    input_ix = np.insert(row_ix, 0, batch_ix)
    input_shape = (batch_size * seqlen, ) + row_shape

    # Order indexes of output (append rank indexes as well).
    output_ix = np.insert(col_ix, 0, batch_ix)
    output_ix = np.append(output_ix, (rank_ix[0], rank_ix[-1]))

    # Prepare contraction operands.
    ops = [input_shape, input_ix]
    for core_ix, core_shape in zip(cores_ix, cores_shape):
        ops.append(core_shape)
        ops.append(core_ix)
    ops.append(output_ix)
    ops = [tuple(op) for op in ops]

    return contract_expression(*ops)


class TTCompressedLinear(CompressedLinear):
    """Class TTCompressedLinear is a layer which represents a weight matrix of
    linear layer in factorized view as tensor train matrix.

    >>> linear_layer = T.nn.Linear(6, 6)
    >>> tt_layer = TTCompressedLinear \
    ...     .from_linear(linear_layer, rank=2, shape=((2, 3), (3, 2)))
    """

    def __init__(self, cores: Sequence[T.Tensor],
                 bias: Optional[T.Tensor] = None):
        super().__init__()

        for i, core in enumerate(cores):
            if core.ndim != 4:
                raise ValueError('Expected number of dimensions of the '
                                 f'{i}-th core is 4 but given {cores.ndim}.')

        # Prepare contaction expression.
        self.rank = (1, ) + tuple(core.shape[3] for core in cores)
        self.shape = (tuple(core.shape[1] for core in cores),
                      tuple(core.shape[2] for core in cores))
        self.contact = make_contraction(self.shape, self.rank)

        # TT-matrix is applied on the left. So, this defines number of input
        # and output features.
        self.in_features = np.prod(self.shape[0])
        self.out_features = np.prod(self.shape[1])

        # Create trainable variables.
        self.cores = T.nn.ParameterList(T.nn.Parameter(core) for core in cores)
        self.bias = None
        if bias is not None:
            if bias.size() != self.out_features:
                raise ValueError(f'Expected bias size is {self.out_features} '
                                 f'but its shape is {bias.shape}.')
            self.bias = T.nn.Parameter(bias)

    def forward(self, input: T.Tensor) -> T.Tensor:
        # We need replace the feature dimension with multi-dimension to contact
        # with TT-matrix.
        input_shape = input.shape
        input = input.reshape(-1, *self.shape[0])

        # Contract input with weights and replace back multi-dimension with
        # feature dimension.
        output = self.contact(input, *self.cores)
        output = output.reshape(*input_shape[:-1], self.out_features)

        if self.bias is not None:
            output += self.bias
        return output

    @classmethod
    def from_linear(cls, linear: T.nn.Linear,
                    shape: Tuple[Tuple[int], Tuple[int]], rank: int, **kwargs):
        ndim = len(shape[0])

        # Prepare information about shape and rank of TT (not TTM).
        tt_rank = (1, ) + (rank, ) * (ndim - 1) + (1, )
        tt_shape = tuple(n * m for n, m in zip(*shape))

        # Reshape weight matrix to tensor indexes like TT-matrix.
        matrix = linear.weight.data.T
        tensor = matrix.reshape(shape[0] + shape[1])
        for i in range(ndim - 1):
            tensor = tensor.moveaxis(ndim + i, 2 * i + 1)

        # Reshape TT-matrix to a plain TT and apply decomposition.
        tensor = tensor.reshape(tt_shape)
        cores = ttd(tensor, tt_rank, **kwargs)

        # Reshape TT-cores back to TT-matrix cores (TTM-cores).
        core_shapes = zip(tt_rank, *shape, tt_rank[1:])
        cores = [core.reshape(core_shape)
                 for core, core_shape in zip(cores, core_shapes)]

        # Make copy of bias if it exists.
        bias = None
        if linear.bias is not None:
            bias = T.clone(linear.bias.data)

        return TTCompressedLinear(cores, bias)

class FWTTCompressedLinear(CompressedLinear):
    """Class TTCompressedLinear is a layer which represents a weight matrix of
    linear layer in factorized view as tensor train matrix.

    >>> linear_layer = T.nn.Linear(6, 6)
    >>> tt_layer = TTCompressedLinear \
    ...     .from_linear(linear_layer, rank=2, shape=((2, 3), (3, 2)))
    """

    def __init__(self, cores: Sequence[T.Tensor],
                 fisher_information: T.Tensor,
                 bias: Optional[T.Tensor] = None):
        super().__init__()
        
        for i, core in enumerate(cores):
            if core.ndim != 4:
                raise ValueError('Expected number of dimensions of the '
                                 f'{i}-th core is 4 but given {cores.ndim}.')

        self.fisher_information = fisher_information

        # Prepare contaction expression.
        self.rank = (1, ) + tuple(core.shape[3] for core in cores)
        self.shape = (tuple(core.shape[1] for core in cores),
                      tuple(core.shape[2] for core in cores))
        self.contact = make_contraction(self.shape, self.rank)

        # TT-matrix is applied on the left. So, this defines number of input
        # and output features.
        self.in_features = np.prod(self.shape[0])
        self.out_features = np.prod(self.shape[1])

        # Create trainable variables.
        self.cores = T.nn.ParameterList(T.nn.Parameter(core) for core in cores)
        self.bias = None
        if bias is not None:
            if bias.size() != self.out_features:
                raise ValueError(f'Expected bias size is {self.out_features} '
                                 f'but its shape is {bias.shape}.')
            self.bias = T.nn.Parameter(bias)

    def forward(self, input: T.Tensor) -> T.Tensor:
        # We need replace the feature dimension with multi-dimension to contact
        # with TT-matrix.
        input = input / self.fisher_information.to(input.device)
        
        input_shape = input.shape
        input = input.reshape(-1, *self.shape[0])

        # Contract input with weights and replace back multi-dimension with
        # feature dimension.
        output = self.contact(input, *self.cores)
        output = output.reshape(*input_shape[:-1], self.out_features)

        if self.bias is not None:
            output += self.bias
        return output

    @classmethod
    def from_linear(cls, linear: T.nn.Linear, fisher_information: T.Tensor,
                    shape: Tuple[Tuple[int], Tuple[int]], rank: int, **kwargs):
        ndim = len(shape[0])

        # Prepare information about shape and rank of TT (not TTM).
        tt_rank = (1, ) + (rank, ) * (ndim - 1) + (1, )
        tt_shape = tuple(n * m for n, m in zip(*shape))

        # Reshape weight matrix to tensor indexes like TT-matrix.
        #FWSVD
        fisher_information = T.sqrt(fisher_information.sum(0))
        matrix = fisher_information.reshape(1, -1).to(linear.weight.data.device) * linear.weight.data
        tensor = matrix.T.reshape(shape[0] + shape[1])
        
        #matrix = linear.weight.data.T
        #tensor = matrix.reshape(shape[0] + shape[1])
        for i in range(ndim - 1):
            tensor = tensor.moveaxis(ndim + i, 2 * i + 1)

        # Reshape TT-matrix to a plain TT and apply decomposition.
        tensor = tensor.reshape(tt_shape)
        cores = ttd(tensor, tt_rank, **kwargs)

        # Reshape TT-cores back to TT-matrix cores (TTM-cores).
        core_shapes = zip(tt_rank, *shape, tt_rank[1:])
        cores = [core.reshape(core_shape)
                 for core, core_shape in zip(cores, core_shapes)]

        # Make copy of bias if it exists.
        bias = None
        if linear.bias is not None:
            bias = T.clone(linear.bias.data)

        return FWTTCompressedLinear(cores, fisher_information, bias)