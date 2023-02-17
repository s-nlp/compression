import torch
import torch.nn as nn
import tntorch as tn
from typing import Optional, Tuple

from .TTLinear import TTLinear, WeightedTTLinear, Checkpointed
from .forward_backward import forward, full_matrix_backward, forward_backward_module, opt_tt_multiply

from ..src.fwsvd import estimate_fisher_weights_bert

def ttm_compress_bert_ffn(model, ranks, input_dims, output_dims, with_checkpoints=True,
                          dataloader=None):
    if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
        encoder = model.bert.encoder
    elif hasattr(model, "encoder"):
        encoder = model.encoder
    else:
        raise ValueError("Expected model to have attribute 'encoder' or 'bert.encoder'.")

    if dataloader is not None:
        fisher_intermediate, fisher_output = estimate_fisher_weights_bert(model, dataloader, compute_full=False, device="cuda:0")
    
    for i, layer in enumerate(encoder.layer):
        token_dim, hidden_dim = layer.intermediate.dense.weight.T.shape

        if dataloader is not None:
            tt_weight = WeightedTTLinear(fisher_intermediate[i], token_dim, hidden_dim, ranks, input_dims, output_dims)
        else:
            tt_weight = TTLinear(token_dim, hidden_dim, ranks, input_dims, output_dims,)

        tt_weight.set_from_linear(layer.intermediate.dense)

        if with_checkpoints:
            print("Checkpoint!")
            layer.intermediate.dense = Checkpointed(tt_weight)
        else:
            layer.intermediate.dense = tt_weight

        if dataloader is not None:
            tt_weight = WeightedTTLinear(fisher_output[i], token_dim, hidden_dim, ranks, input_dims, output_dims)
        else:
            # second linear layer has reversed dimensions,
            # so we swap input_dims and output_dims
            tt_weight = TTLinear(token_dim, hidden_dim, ranks, input_dims, output_dims,)

        tt_weight = TTLinear(hidden_dim, token_dim, ranks, output_dims, input_dims,)
        tt_weight.set_from_linear(layer.output.dense)

        if with_checkpoints:
            print("Checkpoint!")
            layer.output.dense = Checkpointed(tt_weight)
        else:
            layer.output.dense = tt_weight

    return model


