from typing import Optional, Sequence

import torch as T


class SVDCompressedLinearFunc(T.autograd.Function):

    @staticmethod
    def forward(ctx, input: T.Tensor, lhs: T.Tensor,
                rhs: T.Tensor, bias: Optional[T.Tensor] = None) -> T.Tensor:
        # See PEP-0465 on matmul operator associativity.
        # https://peps.python.org/pep-0465/#precedence-and-associativity
        output = (input @ lhs) @ rhs
        if bias is not None:
            output += bias[None, :]
        ctx.bias = bias is not None
        ctx.save_for_backward(input, lhs, rhs)
        return output

    @staticmethod
    def backward(ctx, grad_output: Sequence[T.Tensor]):
        input, lhs, rhs = ctx.saved_tensors

        # Flatten input and output gradients over the leading dimensions.
        inp_size = lhs.shape[0]
        out_size = rhs.shape[1]
        input_shape = input.shape
        input = input.reshape(-1, inp_size)
        grad_output = grad_output.reshape(-1, out_size)

        input_grad = None
        if ctx.needs_input_grad[0]:
            input_grad = (grad_output @ rhs.T) @ lhs.T

        lhs_grad = None
        if ctx.needs_input_grad[1]:
            # On practice for large models embedding dimension is large than
            # batch size.
            lhs_grad = input.T @ (grad_output @ rhs.T)

        rhs_grad = None
        if ctx.needs_input_grad[2]:
            # Again, batch size is usually lesser then embedding dimension.
            rhs_grad = (input @ lhs).T @ grad_output

        bias_grad = None
        if ctx.needs_input_grad[3]:
            bias_grad = grad_output.sum(axis=0)

        # Restore shape of input gradients.
        input_grad = input_grad.reshape(input_shape)
        return input_grad, lhs_grad, rhs_grad, bias_grad


compressed_linear_svd = SVDCompressedLinearFunc.apply
