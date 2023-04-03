import pytest
import torch as T

from rut5compressed.nn.modules import (SVDCompressedLinear, TTCompressedLinear,
                                       factorize, make_contraction)


@pytest.fixture
def linear_layer(bias=True) -> T.nn.Linear:
    rng = T.Generator()
    rng.manual_seed(0x160d93b1)
    mat = T.normal(0, 1, (3072, 768), generator=rng)
    mat *= (20 / 3072)**0.5
    layer = T.nn.Linear(in_features=768, out_features=3072, bias=bias)
    layer.weight.data = mat
    return layer


def test_svd_linear_sanity(linear_layer: T.nn.Linear):
    rank = 768 // 2
    layer = SVDCompressedLinear.from_linear(linear_layer, rank=rank)

    # Check model output/input dims.
    assert layer.in_features == 768
    assert layer.out_features == 3072

    # Check parameters existance.
    names = {name for name, _ in layer.named_parameters()}
    assert 'lhs' in names
    assert 'rhs' in names
    assert 'bias' in names

    # Check factor shapes.
    assert layer.lhs.shape == (768, rank)
    assert layer.rhs.shape == (rank, 3072)


def test_svd_linear_forward(linear_layer: T.nn.Linear):
    rank = 768 // 2
    layer = SVDCompressedLinear.from_linear(linear_layer, rank=rank)
    # We can't compare outputs for dense matrices.
    xs = T.ones((1000, layer.in_features), requires_grad=False)
    _ = layer(xs)
    _ = linear_layer(xs)


def test_svd_linear_backward(linear_layer: T.nn.Linear):
    rank = 768 // 2
    layer = SVDCompressedLinear.from_linear(linear_layer, rank=rank)
    # Pass a batch.
    xs = T.ones((1000, layer.in_features), requires_grad=True)
    ys = layer(xs)
    ys.backward(T.ones_like(ys, requires_grad=False))
    # And check existance of gradients.
    assert xs.grad.shape == xs.shape
    assert layer.lhs.grad.shape == layer.lhs.shape
    assert layer.rhs.grad.shape == layer.rhs.shape


class TestFactorize:

    def test_trivial(self):
        assert factorize(0) == {}
        assert factorize(1) == {}

    def test_even(self):
        assert factorize(2) == {2: 1}
        assert factorize(4) == {2: 2}
        assert factorize(8) == {2: 3}

    def test_prime(self):
        for exp in (1, 2):
            for prime in (3, 5, 7, 11, 13, 19, 23):
                assert factorize(prime**exp) == {prime: exp}

    def test_arbitrary(self):
        assert factorize(6) == {2: 1, 3: 1}
        assert factorize(9) == {3: 2}
        assert factorize(12) == {2: 2, 3: 1}
        assert factorize(15) == {3: 1, 5: 1}
        assert factorize(36) == {2: 2, 3: 2}
        assert factorize(768) == {2: 8, 3: 1}


class TestTTCompressedLinear:

    def test_forward(self, linear_layer: T.nn.Linear):
        rank = 32  # Uniform TT-rank.
        shape = (
            (2**3, 2**3, 2**2 * 3),  # Row dimention.
            (2**4, 2**4, 2**2 * 3),  # Column dimention.
        )
        layer = TTCompressedLinear \
            .from_linear(linear_layer, shape=shape, rank=rank)

        # Check model output/input dims.
        assert layer.in_features == 768
        assert layer.out_features == 3072

        # Check parameters existance.
        names = {name for name, _ in layer.named_parameters()}
        assert 'cores.0' in names
        assert 'cores.1' in names
        assert 'cores.2' in names
        assert 'bias' in names

        # Check cores ranks.
        assert layer.rank == (1, rank, rank, 1)

        # Check cores shapes.
        assert layer.cores[0].shape == (1, shape[0][0], shape[1][0], rank)
        assert layer.cores[1].shape == (rank, shape[0][1], shape[1][1], rank)
        assert layer.cores[2].shape == (rank, shape[0][2], shape[1][2], 1)

        # Calculate output and check shape.
        input = T.ones((2, 3, layer.in_features))
        output = layer(input)
        assert output.shape[:-1] == input.shape[:-1]
        assert output.shape[-1] == layer.out_features


def test_make_contraction():
    rank = (1, 32, 32, 1)
    shape = ((2**3, 2**3, 2**2 * 3),  # Row dimention.
             (2**4, 2**4, 2**2 * 3))  # Column dimention.
    expr = make_contraction(shape, rank)
    cores = [T.randn(core_shape) for core_shape in zip(rank, *shape, rank[1:])]
    input = T.randn((32, ) + shape[0], requires_grad=True)
    output = expr(input, *cores, backend='torch')
    assert output.requires_grad
    assert output.shape[-2:] == (1, 1)
    output = output.squeeze()
    assert output.shape == (input.shape[0], ) + shape[1]
