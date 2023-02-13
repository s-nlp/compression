import torch
import torch.nn as nn
import tntorch as tn
from torch.utils.checkpoint import checkpoint

from .TTLinear import TTLinear
from .forward_backward import forward, full_matrix_backward, forward_backward_module, opt_tt_multiply


class Checkpointed(nn.Sequential):
    def forward(self, *args):
        return checkpoint(super().forward, *args)


def ttm_compress_bert_ffn(model, ranks, input_dims, output_dims, with_checkpoints=True, weight_int=None, weight_out=None, weight_count=None):
    if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
        encoder = model.bert.encoder
    elif hasattr(model, "encoder"):
        encoder = model.encoder
    else:
        raise ValueError("Expected model to have attribute 'encoder' or 'bert.encoder'.")


    for i, layer in enumerate(encoder.layer):
        token_dim, hidden_dim = layer.intermediate.dense.weight.T.shape
        tt_weight = TTLinear(token_dim, hidden_dim, ranks, input_dims, output_dims,)
                             #forward_fn=forward_backward_module(forward, full_matrix_backward(forward)))
        if weight_int is not None:
            weight_int_ = weight_int[i]
        else: 
            weight_int_ = torch.ones_like(layer.intermediate.dense.weight.data)

        tt_weight.set_from_linear_w(layer.intermediate.dense, weight_int_)
        layer.intermediate.dense = tt_weight

        if with_checkpoints:
            print("Checkpoint!")
            layer.intermediate.dense = Checkpointed(tt_weight)
        else:
            layer.intermediate.dense = tt_weight

        # second linear layerhas reversed dimensions,
        # so we swap input_dims and output_dims
        tt_weight = TTLinear(hidden_dim, token_dim, ranks, output_dims, input_dims,)
                             #forward_fn=forward_backward_module(forward, full_matrix_backward(forward)))
        if weight_out is not None:
            weight_out_ = weight_out[i]
        else: 
            weight_out_ = torch.ones_like(layer.output.dense.weight.data)
        
        tt_weight.set_from_linear_w(layer.output.dense, weight_out_)

        if with_checkpoints:
            print("Checkpoint!")
            layer.output.dense = Checkpointed(tt_weight)
        else:
            layer.output.dense = tt_weight

    return model


