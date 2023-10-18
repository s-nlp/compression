from typing import List, Callable
import torch
import torch.nn as nn
import tntorch as tn
from math import sqrt
from .ttmatrix_fisher import TTMatrix

from .forward_backward import forward, einsum_forward

class WeightedTTLinear(nn.Module):
    def __init__(self, fisher_information: torch.Tensor, *args, **kwargs):
        super().__init__()
        self.fisher_information = torch.sqrt(fisher_information.sum(0))
        self.alpha = 0.06
        self.fisher_information += torch.ones_like(self.fisher_information) * self.alpha
        self.ttlinear = TTLinear(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        self.fisher_information = self.fisher_information.to(x.device)
        return self.ttlinear(x / self.fisher_information)

    def set_weight(self, new_weights: torch.Tensor):
        self.ttlinear.set_weight(self.fisher_information.reshape(1, -1).to(new_weights.device) * new_weights)

    def set_from_linear(self, linear):
        self.set_weight(linear.weight.data)
        self.ttlinear.bias = nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None

class TTLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ranks: List[int], input_dims: List[int],
                 output_dims: List[int], bias: bool = True, device=None, dtype=None,
                 forward_fn: Callable = einsum_forward):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ranks = list(ranks)
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.forward_fn = forward_fn

        # Initialize weights from uniform[-1 / sqrt(in_features), 1 / sqrt(in_features)]
        factory_kwargs = {"device": device, "dtype": dtype}
        init = torch.rand(in_features, out_features, **factory_kwargs)
        init = (2 * init - 1) / sqrt(in_features)
        
        init_fisher = torch.zeros(init.shape)
        for i in range(0, min(init.shape)):
            init_fisher[i, i] = 1.0

        
        self.weight = tn.TTMatrix(init, list(ranks), input_dims, output_dims)

        # torch doesn't recognize attributes of self.weight as parameters,
        # so we have to use ParameterList
        self.cores = nn.ParameterList([nn.Parameter(core) for core in self.weight.cores])
        self.weight.cores = self.cores

        if bias:
            init = torch.rand(out_features, **factory_kwargs)
            init = (2 * init - 1) / sqrt(out_features)
            self.bias = nn.Parameter(init)
        else:
            self.register_parameter('bias', None)


    def forward(self, x: torch.Tensor):
        res = self.forward_fn(self.cores, x)

        new_shape = x.shape[:-1] + (self.out_features,)
        res = res.reshape(*new_shape)

        if self.bias is not None:
            res += self.bias

        return res

    def set_weight(self, new_weights: torch.Tensor):
        # in regular linear layer weights are transposed, so we transpose back
        new_weights = new_weights.clone().detach().T

        shape = torch.Size((self.in_features, self.out_features))
        assert new_weights.shape == shape, f"Expected shape {shape}, got {new_weights.shape}"

        self.weight = tn.TTMatrix(new_weights, self.ranks, self.input_dims, self.output_dims)
        self.cores = nn.ParameterList([nn.Parameter(core) for core in self.weight.cores])
        self.weight.cores = self.cores
        
    def set_weight_with_fisher(self, new_weights: torch.Tensor, fisher_matrix: torch.Tensor):
        # in regular linear layer weights are transposed, so we transpose back
        new_weights = new_weights.clone().detach().T

        shape = torch.Size((self.in_features, self.out_features))
        assert new_weights.shape == shape, f"Expected shape {shape}, got {new_weights.shape}"

        self.weight = TTMatrix(new_weights, fisher_matrix,
                               self.ranks, self.input_dims, self.output_dims)
        self.cores = nn.ParameterList([nn.Parameter(core) for core in self.weight.cores])
        self.weight.cores = self.cores

    def set_from_linear(self, linear: nn.Linear):
        self.set_weight(linear.weight.data)
        self.bias = nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None
        
    def set_from_linear_w(self, linear: nn.Linear, fisher_matrix: torch.Tensor):
        self.set_weight_with_fisher(linear.weight.data, fisher_matrix)
        self.bias = nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None