import torch
import torch.nn as nn
import tntorch as tn
from torch.utils.checkpoint import checkpoint

from .TTLinear import TTLinear
from .forward_backward import forward, full_matrix_backward, forward_backward_module, opt_tt_multiply


class Checkpointed(nn.Sequential):
    def forward(self, *args):
        return checkpoint(super().forward, *args)


def ttm_compress_bert_ffn(model, ranks, input_dims, output_dims, with_checkpoints=True):
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

        tt_weight.set_from_linear(layer.intermediate.dense)
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
        tt_weight.set_from_linear(layer.output.dense)

        if with_checkpoints:
            print("Checkpoint!")
            layer.output.dense = Checkpointed(tt_weight)
        else:
            layer.output.dense = tt_weight

    return model


