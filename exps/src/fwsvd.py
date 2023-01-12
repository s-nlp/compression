from typing import Tuple, Dict, Optional, Callable
from collections import defaultdict

from tqdm import tqdm

import numpy as np
import torch
import torch.linalg as LA
from torch.utils.data import DataLoader
from transformers import BertModel

from .torch_utils import FWDense
from .weighted_low_rank_decompositions import weighted_svd, nesterov, anderson


def compute_row_sum_svd_decomposition(A: torch.Tensor, weights: Optional[torch.Tensor] = None, rank: Optional[int] = None):
    """Computes FWSVD from https://arxiv.org/pdf/2207.00112.pdf.

    Args:
      A (torch.Tensor): matrix of size (H, W) to decompose
      weights (Optional[torch.Tensor]): matrix of size (H, W) or (H,) - Fisher weights.
        If None (default), set to ones.
      rank (Optional[int]): approx. rank in SVD. If None (default), computes
        full-rank decomposition without compression.

    Returns:
      left_w (torch.Tensor): matrix [H, r] = I_hat_inv @ Ur @ Sr
      right_w (torch.Tensor): matrix [r, W] = Vr.T
    """
    h, w = A.shape

    if weights is None:
        weights = torch.ones(h)

    if weights.ndim > 1:
        weights = weights.sum(dim=1)

    i_hat = torch.diag(torch.sqrt(weights))
    i_hat_inv = LA.inv(i_hat)  # actually it's diagonal so we can just take 1 / i_hat

    u, s, v = LA.svd(i_hat @ A, full_matrices=True)
    s = torch.diag(s)  # more convenient form

    if rank is not None:
        u = u[:, :rank]
        s = s[:rank, :rank]
        v = v[:rank]
    else:
        s_tmp = s
        s = torch.zeros_like(A)
        s[:min(h, w), :min(h, w)] = s_tmp

    left_w = i_hat_inv @ (u @ s)
    right_w = v

    return left_w, right_w


def estimate_fisher_weights_bert(
    model: BertModel,
    dataloader: DataLoader,
    loss_fn: Optional[Callable] = None,
    compute_full: bool = True,
    device: str = 'cpu',
) -> Tuple:
    """Calculate Fisher information in each linear layer of the Bert-type model.

    Args:
      model (BertModel): BertModel instance from transformers package
      dataloader (Dataloader): instance of torch.utils.Dataloader with e.g. FineTuneDataset instance as dataset. 
        Data on which the gradients will be computed.
      loss_fn (Optional[Callable]): loss function. If None (default),
        assume that model forward pass returns loss value.
        Note: If loss_fn is not None, signature should be like loss_fn(inputs, outputs),
          where inputs is a batch of data from dataloader, outputs is the result of model(inputs)
      compute_full (bool): If True (default), stores gradients for each weight.
        If False, stores row gradients as sum over gradients of weights in each row.

    Returns:
        fisher_int, fisher_out (Tuple[torch.Tensor]): 2 Dicts of len = # of linear layers in the model
          with fisher information for intermediate and output linear layers of Bert-type model.
    """
    hidden_dim = model.config.hidden_size
    intermediate_dim = model.config.intermediate_size
    num_hidden_layers = model.config.num_hidden_layers

    n_steps_per_epoch = len(dataloader)
    model = model.to(device)
    model.train()

    if compute_full:
        fisher_int = defaultdict(lambda: torch.zeros((hidden_dim, intermediate_dim), device='cpu'))
        fisher_out = defaultdict(lambda: torch.zeros((intermediate_dim, hidden_dim), device='cpu'))
    else:
        fisher_int = defaultdict(lambda: torch.zeros(hidden_dim, device='cpu'))
        fisher_out = defaultdict(lambda: torch.zeros(intermediate_dim, device='cpu'))

    for inputs in tqdm(dataloader, total=n_steps_per_epoch):
        if isinstance(inputs, dict):
            for key, val in inputs.items():  # store all tensors to model device
                if isinstance(val, torch.Tensor):
                    inputs[key] = val.to(device)

            outputs = model.forward(**inputs)
        else:  # assume it's a tuple
            inputs = (inp.to(device) for inp in inputs)

            outputs = model.forward(inputs)

        if loss_fn is None:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        else:
            raise ValueError("Not supported!")
            loss = loss_fn(inputs, outputs)

        loss.backward()

        for i in range(num_hidden_layers):
            grad_int = model.bert.encoder.layer[i].intermediate.dense.weight.grad.detach().cpu().transpose(0, 1) ** 2
            grad_out = model.bert.encoder.layer[i].output.dense.weight.grad.detach().cpu().transpose(0, 1) ** 2

            if not compute_full:
                grad_int = grad_int.sum(axis=1)
                grad_out = grad_out.sum(axis=1)

            fisher_int[i] += grad_int
            fisher_out[i] += grad_out

    fisher_int = dict(map(lambda x: (x[0], x[1] / x[1].max()), fisher_int.items()))
    fisher_out = dict(map(lambda x: (x[0], x[1] / x[1].max()), fisher_out.items()))

    for key, item in fisher_int.items():
        np.save(f"fisher_matrix_inter_layer_{key}.npy", item.detach().cpu().numpy())

    for key, item in fisher_out.items():
        np.save(f"figher_matrix_outer_layer_{key}.npy", item.detach().cpu().numpy())

    return fisher_int, fisher_out


def replace_dense2fw_bert(model: BertModel, fisher_int: Dict, fisher_out: Dict, rank: int = None, 
                          low_rank_method: str = "row-sum-weighted-svd") -> BertModel:
    """Replace Dense layers to FWDense layers in bert-type model.
      See estimate_fisher_weights_bert output for more details.
      rank is the approx. rank in SVD decomposition.
    """
    model = model.to('cpu')
    model.eval()
    
    if low_rank_method == "row-sum-weighted-svd":
        get_decomposition = compute_row_sum_svd_decomposition
    elif low_rank_method == "weighted-svd":
        get_decomposition = weighted_svd
    elif low_rank_method == "nesterov":
        get_decomposition = nesterov
    elif low_rank_method == "anderson":
        get_decomposition = anderson
    else:
        raise ValueError(f"Method {low_rank_method} for low rank factorization is not implemented")
    
    for idx, weights in fisher_int.items():
        w_mat = model.bert.encoder.layer[idx].intermediate.dense.weight.data.transpose(0, 1)
        bias = model.bert.encoder.layer[idx].intermediate.dense.bias.data

        left_w, right_w = get_decomposition(w_mat, weights, rank)
        
        if torch.any(torch.isnan(left_w)) or torch.any(torch.isnan(right_w)):
            raise RunTimeError("Nan in weights after decomposition")

        fw_dense = FWDense(input_dim=left_w.shape[0], hidden_dim=rank, output_dim=right_w.shape[1])
        fw_dense._init_weights(left_w, right_w, bias)

        model.bert.encoder.layer[idx].intermediate.dense = fw_dense
      
    for idx, weights in fisher_out.items():
        w_mat = model.bert.encoder.layer[idx].output.dense.weight.data.transpose(0, 1)
        bias = model.bert.encoder.layer[idx].output.dense.bias.data

        left_w, right_w = get_decomposition(w_mat, weights, rank)

        if torch.any(torch.isnan(left_w)) or torch.any(torch.isnan(right_w)):
            raise RunTimeError("Nan in weights after decomposition")
        
        fw_dense = FWDense(input_dim=left_w.shape[0], hidden_dim=rank, output_dim=right_w.shape[1])
        fw_dense._init_weights(left_w, right_w, bias)

        model.bert.encoder.layer[idx].output.dense = fw_dense

    return model
