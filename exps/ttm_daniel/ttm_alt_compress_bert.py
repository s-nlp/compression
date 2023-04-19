from tqdm.auto import tqdm

import torch
import torch.nn as nn
import tntorch as tn

from .modules import SVDCompressedLinear, TTCompressedLinear, FWSVDCompressedLinear, FWTTCompressedLinear, InvasiveFWTTCompressedLinear

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

    for i, layer in enumerate(tqdm(encoder.layer, desc="Running TTm compression")):
        if weight_int is not None:
            if invasive:
                ttclass = InvasiveFWTTCompressedLinear
            else:
                ttclass = FWTTCompressedLinear
            
            tt_weight = ttclass.from_linear(layer.intermediate.dense, 
                weight_int[i] / weight_count,
                rank=ranks[0], shape=(input_dims, output_dims)
            )
        else: 
            tt_weight = TTCompressedLinear.from_linear(layer.intermediate.dense, 
                                                       rank=ranks[0], shape=(input_dims, output_dims))
        layer.intermediate.dense = tt_weight

        # second linear layerhas reversed dimensions,
        # so we swap input_dims and output_dims
        
        if weight_out is not None:
            if invasive:
                ttclass = InvasiveFWTTCompressedLinear
            else:
                ttclass = FWTTCompressedLinear
            
            tt_weight = ttclass.from_linear(layer.output.dense, 
                weight_out[i] / weight_count,
                rank=ranks[0], shape=(output_dims, input_dims)
            )
        else: 
            tt_weight = TTCompressedLinear.from_linear(layer.output.dense, 
                                                       rank=ranks[0], shape=(output_dims, input_dims))
        layer.output.dense = tt_weight

    return model


def svd_alt_compress_bert_ffn(model, rank,
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
                raise NotImplementedError()
            else:
                svd_weight = FWSVDCompressedLinear.from_linear(layer.intermediate.dense, 
                                                    weight_int[i] / weight_count, 
                                                    rank=rank)
        else:
            svd_weight = SVDCompressedLinear.from_linear(layer.intermediate.dense, 
                                                    rank=rank)
        layer.intermediate.dense = svd_weight

        # second linear layerhas reversed dimensions,
        # so we swap input_dims and output_dims
        
        if weight_out is not None:
            if invasive:
                raise NotImplementedError()
            else:
                svd_weight = FWSVDCompressedLinear.from_linear(layer.output.dense, 
                                                    weight_out[i] / weight_count, 
                                                    rank=rank)
        else:
            svd_weight = SVDCompressedLinear.from_linear(layer.output.dense, 
                                                    rank=rank)
        layer.output.dense = svd_weight

    return model