from .TTLinear import TTLinear


def ttm_compress_bert_ffn(model, ranks, input_dims, output_dims):
    if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
        encoder = model.bert.encoder
    elif hasattr(model, "encoder"):
        encoder = model.encoder
    else:
        raise ValueError("Expected model to have attribute 'encoder' or 'bert.encoder'.")

    for layer in encoder.layer:
        token_dim, hidden_dim = layer.intermediate.dense.weight.T.shape
        tt_weight = TTLinear(token_dim, hidden_dim, ranks, input_dims, output_dims)

        tt_weight.set_from_linear(layer.intermediate.dense)
        layer.intermediate.dense = tt_weight

        # second linear layerhas reversed dimensions,
        # so we swap input_dims and output_dims
        tt_weight = TTLinear(hidden_dim, token_dim, ranks, output_dims, input_dims)
        tt_weight.set_from_linear(layer.output.dense)
        layer.output.dense = tt_weight

    return model
