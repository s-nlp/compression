import tntorch as tn
import torch
import torch.nn as nn

from .modules import (
    FWSVDCompressedLinear,
    FWTTCompressedLinear,
    SVDCompressedLinear,
    TTCompressedLinear,
)


def ttm_alt_compress_bart_ffn(
    model,
    ranks,
    input_dims,
    output_dims,
    with_checkpoints=True,
    weight_int=None,
    weight_out=None,
    weight_count=None,
    invasive=False,
):
    if hasattr(model.model, "encoder") and hasattr(model.model, "decoder"):
        encoder = model.model.encoder
        decoder = model.model.decoder
    for part in (encoder, decoder):
        for i, layer in enumerate(part.layers):
            if weight_int is None:
                tt_weight = TTCompressedLinear.from_linear(
                    layer.fc1, rank=ranks[0], shape=(input_dims, output_dims)
                )
            elif not invasive:
                tt_weight = FWTTCompressedLinear.from_linear(
                    layer.fc1,
                    weight_int[i] / weight_count,
                    rank=ranks[0],
                    shape=(input_dims, output_dims),
                )
            layer.fc1 = tt_weight

            # second linear layerhas reversed dimensions,
            # so we swap input_dims and output_dims

            if weight_out is None:
                tt_weight = TTCompressedLinear.from_linear(
                    layer.fc2, rank=ranks[0], shape=(output_dims, input_dims)
                )
            elif not invasive:
                tt_weight = FWTTCompressedLinear.from_linear(
                    layer.fc2,
                    weight_out[i] / weight_count,
                    rank=ranks[0],
                    shape=(output_dims, input_dims),
                )
            layer.fc2 = tt_weight
    return model


def ttm_alt_compress_bert_ffn(
    model,
    ranks,
    input_dims,
    output_dims,
    with_checkpoints=True,
    weight_int=None,
    weight_out=None,
    weight_count=None,
    invasive=False,
):
    if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
        encoder = model.bert.encoder
    else:
        raise ValueError("Expected model to have attribute 'encoder' or 'bert.encoder'.")

    for i, layer in enumerate(encoder.layer):
        if weight_int is not None:
            if invasive:
                pass
                # tt_weight = TTLinear(token_dim, hidden_dim, ranks,
                #                     input_dims, output_dims,)
                # tt_weight.set_from_linear_w(layer.intermediate.dense,
                #                            weight_int[i] / weight_count)
            else:
                tt_weight = FWTTCompressedLinear.from_linear(
                    layer.intermediate.dense,
                    weight_int[i] / weight_count,
                    rank=ranks[0],
                    shape=(input_dims, output_dims),
                )
        else:
            tt_weight = TTCompressedLinear.from_linear(
                layer.intermediate.dense, rank=ranks[0], shape=(input_dims, output_dims)
            )
        layer.intermediate.dense = tt_weight

        # second linear layerhas reversed dimensions,
        # so we swap input_dims and output_dims

        if weight_out is not None:
            if invasive:
                pass
                # tt_weight = TTLinear(hidden_dim, token_dim, ranks,
                #                     output_dims, input_dims,)
                # tt_weight.set_from_linear_w(layer.output.dense,
                #                            weight_out[i] / weight_count)
            else:
                tt_weight = FWTTCompressedLinear.from_linear(
                    layer.output.dense,
                    weight_out[i] / weight_count,
                    rank=ranks[0],
                    shape=(output_dims, input_dims),
                )
        else:
            tt_weight = TTCompressedLinear.from_linear(
                layer.output.dense, rank=ranks[0], shape=(output_dims, input_dims)
            )
        layer.output.dense = tt_weight

    return model


def svd_alt_compress_bert_ffn(
    model, rank, weight_int=None, weight_out=None, weight_count=None, invasive=False
):
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
            else:
                svd_weight = FWSVDCompressedLinear.from_linear(
                    layer.intermediate.dense, weight_int[i] / weight_count, rank=rank
                )
        else:
            svd_weight = SVDCompressedLinear.from_linear(
                layer.intermediate.dense, rank=rank
            )
        layer.intermediate.dense = svd_weight

        # second linear layerhas reversed dimensions,
        # so we swap input_dims and output_dims

        if weight_out is not None:
            if invasive:
                pass
            else:
                svd_weight = FWSVDCompressedLinear.from_linear(
                    layer.output.dense, weight_out[i] / weight_count, rank=rank
                )
        else:
            svd_weight = SVDCompressedLinear.from_linear(layer.output.dense, rank=rank)
        layer.output.dense = svd_weight

    return model
