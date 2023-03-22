import torch
import torch.nn as nn
import tntorch as tn

from .modules import SVDCompressedLinear, TTCompressedLinear

def ttm_alt_compress_bert_ffn(model, ranks, 
                          input_dims, output_dims, 
                          with_checkpoints=True, 
                          weight_int=None, weight_out=None, 
                          weight_count=None,
                          invasive=False):
    if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
        encoder = model.bert.encoder
    elif hasattr(model, "encoder"):
        encoder = model.encoder
    else:
        raise ValueError("Expected model to have attribute 'encoder' or 'bert.encoder'.")

    for i, layer in enumerate(encoder.layer):
        if weight_int is not None:
            if invasive:
                pass
                #tt_weight = TTLinear(token_dim, hidden_dim, ranks, 
                #                     input_dims, output_dims,)
                #tt_weight.set_from_linear_w(layer.intermediate.dense, 
                #                            weight_int[i] / weight_count)
            else:
                pass
                #tt_weight = WeightedTTLinear(weight_int[i] / weight_count, 
                #                             token_dim, hidden_dim, ranks, 
                #                             input_dims, output_dims)
                #tt_weight.set_from_linear(layer.intermediate.dense)
        else: 
            tt_weight = TTCompressedLinear.from_linear(layer.intermediate, 
                                                       rank=ranks[0], shape=(input_dims, output_dims))
        layer.intermediate = tt_weight

        # second linear layerhas reversed dimensions,
        # so we swap input_dims and output_dims
        
        if weight_out is not None:
            if invasive:
                pass
                #tt_weight = TTLinear(hidden_dim, token_dim, ranks, 
                #                     output_dims, input_dims,)
                #tt_weight.set_from_linear_w(layer.output.dense, 
                #                            weight_out[i] / weight_count)
            else:
                pass
                #tt_weight = WeightedTTLinear(weight_out[i] / weight_count, 
                #                             hidden_dim, token_dim, ranks, 
                #                             output_dims, input_dims)
                #tt_weight.set_from_linear(layer.output.dense)
        else: 
            tt_weight = TTCompressedLinear.from_linear(layer.output, 
                                                       rank=ranks[0], shape=(output_dims, input_dims))
        layer.output = tt_weight

    return model


