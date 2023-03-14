import argparse
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoConfig,
    BertForSequenceClassification,
    BertModel,
    BertPreTrainedModel,
    RobertaForSequenceClassification,
    RobertaModel,
    RobertaPreTrainedModel,
)
from transformers.modeling_utils import prune_linear_layer


def log_all_parameters(
    logger: logging.Logger,
    model_args: argparse.Namespace,
    data_args: argparse.Namespace,
    training_args: argparse.Namespace,
    additional_args: argparse.Namespace,
) -> None:
    """
    Logs all arguments passed to the script using the provided logger.

    Args:
        logger (logging.Logger): logger object to log the arguments
        model_args (argparse.Namespace): argparse Namespace object containing the model arguments
        data_args (argparse.Namespace): argparse Namespace object containing the data arguments
        training_args (argparse.Namespace): argparse Namespace object containing the training arguments
        additional_args (argparse.Namespace): argparse Namespace object containing any additional arguments
    """
    logger.info("Model Arguments:")
    for arg in vars(model_args):
        logger.info(f"{arg} = {getattr(model_args, arg)}")

    logger.info("Data Arguments:")
    for arg in vars(data_args):
        logger.info(f"{arg} = {getattr(data_args, arg)}")

    logger.info("Training Arguments:")
    for arg in vars(training_args):
        logger.info(f"{arg} = {getattr(training_args, arg)}")

    logger.info("Additional Arguments:")
    for arg in vars(additional_args):
        logger.info(f"{arg} = {getattr(additional_args, arg)}")


def load_from_tsv(file):
    lines = open(file, "r").readlines()
    data = [line.strip().split("\t") for line in lines[1:]]
    headers = lines[0].strip().split("\t")
    d = defaultdict(list)
    for i, head in enumerate(headers):
        for dd in data:
            d[head].append(dd[i])

    return Dataset.from_dict(d)


def calculate_parameters(module: nn.Module) -> int:
    """
    Returns the number of parameters of a given module
    (excluding embedding, layer_transformation,
    classifier and pooler parameters)

    Args:
    - module (nn.Module): a PyTorch module.

    Returns:
    - int: number of parameters in the module.

    """
    keys = ["embedding", "layer_transformation", "classifier", "pooler"]
    return sum(
        p.numel()
        for n, p in module.named_parameters()
        if all(key not in n for key in keys)
    )


def edit_config(config: AutoConfig, additional_args: Union[bool, None]) -> None:
    """
    Edits the configuration object based on additional arguments.

    Args:
    - config (AutoConfig): configuration object from transformers library.
    - additional_args (Union[bool, None]): additional arguments to modify configuration.
    """
    if additional_args is not None:
        config.transform_embedding = additional_args.transform_embedding
        config.do_distill = additional_args.do_distill
        config.do_layer_distill = additional_args.do_layer_distill


def initialize_layer_transformation(model: BertForSequenceClassification) -> None:
    """
    Initializes the layer_transformation weights as an identity matrix and bias as zero.

    Args:
    - model (BertForSequenceClassification): model with layer_transformation attribute.

    """
    model.layer_transformation.weight.data.copy_(
        torch.eye(len(model.layer_transformation.weight))
    )
    model.layer_transformation.bias.data.fill_(0)


def load_model_with_zs(
    model_path: str,
    model_class: Union[BertPreTrainedModel, RobertaPreTrainedModel],
    zs: dict,
) -> Union[BertPreTrainedModel, RobertaPreTrainedModel]:
    """
    Loads a model with pretrained weights from given path, then prunes the model based on provided sparsity.

    Args:
    - model_path (str): path to the directory with pretrained model files.
    - model_class (BertForSequenceClassification): model class from transformers library.
    - zs (dict): dictionary containing pruning sparsity for each weight matrix.

    Returns:
    - BertForSequenceClassification: a pruned BERT model with pretrained weights.

    """
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        config = AutoConfig.from_pretrained(model_path)
    model = model_class.from_pretrained(model_path, config=config)
    loaded_weights = torch.load(
        os.path.join(model_path, "pytorch_model.bin"), map_location="cpu"
    )
    model.load_state_dict(loaded_weights)
    print(f"Load weights from {model_path}")

    update_params(model, zs)
    print(f"Model Size before pruning: {calculate_parameters(model)}")
    prune_model_with_z(zs, model)
    print(f"Model Size after pruning: {calculate_parameters(model)}")
    return model


def load_model(
    model_path: str, model_class: type, zs: Optional[torch.Tensor] = None
) -> torch.nn.Module:
    """Load a PyTorch model from a given path with a set of hyperparameters.

    Args:
        model_path (str): The path to the saved model file.
        model_class (type): The PyTorch model class.
        zs (torch.Tensor, optional): The hyperparameters of the model. Defaults to None.

    Returns:
        torch.nn.Module: The loaded PyTorch model.
    """
    assert zs is not None
    model = load_model_with_zs(model_path, model_class, zs)
    print(f"Model Size: {calculate_parameters(model)}")
    return model


# load the l0 module
def load_l0_module(model_path: str) -> Optional[torch.nn.Module]:
    """Load the L0 module of a PyTorch model from a given path.

    Args:
        model_path (str): The path to the saved model file.

    Returns:
        torch.nn.Module: The loaded L0 module of the PyTorch model, if it exists; otherwise, None.
    """
    l0_module_path = os.path.join(model_path, "l0_module.pt")
    if os.path.exists(l0_module_path):
        return torch.load(l0_module_path, map_location=torch.device("cpu"))
    else:
        return None


# z values could be in [0, 1), we update the parameters accordingly with z values
def update_params(
    model: Union[
        BertModel,
        RobertaModel,
        BertForSequenceClassification,
        RobertaForSequenceClassification,
    ],
    zs: Dict[str, Union[torch.Tensor, None]],
) -> None:
    """
    Update the parameters of a BERT or RoBERTa model based on the given scaling factors.

    Args:
        model: A BERT or RoBERTa model to update.
        zs: A dictionary containing scaling factors for various parts of the model. The dictionary may contain the
            following keys:

            - "intermediate_z": A tensor of shape (num_layers,) containing scaling factors for the intermediate
              layers of the model.
            - "mlp_z": A tensor of shape (num_layers, hidden_dims) containing scaling factors for the feedforward
              layers of the model.
            - "head_z": A tensor of shape (num_layers, num_heads) containing scaling factors for the attention heads
              of the model.
            - "head_layer_z": A tensor of shape (num_layers, hidden_dims) containing scaling factors for the final
              output layer of the attention heads of the model.
            - "hidden_z": A tensor of shape (hidden_dims,) containing scaling factors for the embedding layer of the
              model.
    """
    bert = model.bert if hasattr(model, "bert") else model.roberta

    config = model.config
    hidden_dims = config.hidden_size
    num_heads = config.num_attention_heads
    dims_per_head = hidden_dims // num_heads
    num_layers = config.num_hidden_layers

    if zs is not None:
        if "intermediate_z" in zs:
            for layer in range(num_layers):
                intermediate_z = zs["intermediate_z"][layer].cpu().squeeze().clone()
                bert.encoder.layer[layer].output.dense.weight.data = bert.encoder.layer[
                    layer
                ].output.dense.weight.data.mul(intermediate_z)
                if "mlp_z" in zs:
                    mlp_z = zs["mlp_z"][layer].cpu()
                    bert.encoder.layer[layer].output.dense.weight.data = (
                        bert.encoder.layer[layer]
                        .output.dense.weight.data.transpose(0, 1)
                        .mul(mlp_z)
                        .transpose(0, 1)
                    )
                    bert.encoder.layer[
                        layer
                    ].output.dense.bias.data = bert.encoder.layer[
                        layer
                    ].output.dense.bias.data.mul(
                        mlp_z
                    )

        if "head_z" in zs:
            for layer in range(num_layers):
                head_z = zs["head_z"][layer].cpu().squeeze().clone()
                head_z = torch.repeat_interleave(head_z, dims_per_head)
                bert.encoder.layer[layer].attention.self.value.weight.data = (
                    bert.encoder.layer[layer]
                    .attention.self.value.weight.transpose(0, 1)
                    .data.mul(head_z)
                    .transpose(0, 1)
                )
                bert.encoder.layer[
                    layer
                ].attention.self.value.bias.data = bert.encoder.layer[
                    layer
                ].attention.self.value.bias.data.mul(
                    head_z
                )
                if "head_layer_z" in zs:
                    head_layer_z = zs["head_layer_z"][layer].cpu()
                    bert.encoder.layer[layer].attention.output.dense.weight.data = (
                        bert.encoder.layer[layer]
                        .attention.output.dense.weight.transpose(0, 1)
                        .data.mul(head_layer_z)
                        .transpose(0, 1)
                    )
                    bert.encoder.layer[
                        layer
                    ].attention.output.dense.bias.data = bert.encoder.layer[
                        layer
                    ].attention.output.dense.bias.data.mul(
                        head_layer_z
                    )

        if "hidden_z" in zs:
            hidden_z = zs["hidden_z"].cpu().squeeze().clone()
            bert.embeddings.word_embeddings.weight.data = (
                bert.embeddings.word_embeddings.weight.data.mul(hidden_z)
            )
            bert.embeddings.position_embeddings.weight.data = (
                bert.embeddings.position_embeddings.weight.data.mul(hidden_z)
            )
            bert.embeddings.token_type_embeddings.weight.data = (
                bert.embeddings.token_type_embeddings.weight.data.mul(hidden_z)
            )
            for layer in range(num_layers):
                bert.encoder.layer[
                    layer
                ].attention.self.key.weight.data = bert.encoder.layer[
                    layer
                ].attention.self.key.weight.data.mul(
                    hidden_z
                )
                bert.encoder.layer[
                    layer
                ].attention.self.query.weight.data = bert.encoder.layer[
                    layer
                ].attention.self.query.weight.data.mul(
                    hidden_z
                )
                bert.encoder.layer[
                    layer
                ].attention.self.value.weight.data = bert.encoder.layer[
                    layer
                ].attention.self.value.weight.data.mul(
                    hidden_z
                )
                bert.encoder.layer[layer].attention.output.dense.weight.data = (
                    bert.encoder.layer[layer]
                    .attention.output.dense.weight.data.transpose(0, 1)
                    .mul(hidden_z)
                    .transpose(0, 1)
                )
                bert.encoder.layer[
                    layer
                ].attention.output.dense.bias.data = bert.encoder.layer[
                    layer
                ].attention.output.dense.bias.data.mul(
                    hidden_z
                )
                bert.encoder.layer[
                    layer
                ].intermediate.dense.weight.data = bert.encoder.layer[
                    layer
                ].intermediate.dense.weight.data.mul(
                    hidden_z
                )
                bert.encoder.layer[layer].output.dense.weight.data = (
                    bert.encoder.layer[layer]
                    .output.dense.weight.data.transpose(0, 1)
                    .mul(hidden_z)
                    .transpose(0, 1)
                )
            if hasattr(bert.pooler, "dense"):
                bert.pooler.dense.weight.data = bert.pooler.dense.weight.data.mul(
                    hidden_z
                )
            if hasattr(model, "qa_outputs"):
                model.qa_outputs.weight.data = model.qa_outputs.weight.data.mul(hidden_z)


def prune_model_with_z(
    zs: Optional[Dict[str, torch.Tensor]], model: nn.Module
) -> Tuple[Optional[Dict[int, List[int]]], Optional[Dict[int, List[int]]]]:
    """
    Prune a model with given structured sparsity pattern.

    Args:
        zs: The structured sparsity pattern for the model. It should be a dictionary with keys: 'head_z', 'head_layer_z', 'intermediate_z', 'mlp_z', and 'hidden_z'.
        model: The PreTrainedModel to be pruned.

    Returns:
        A tuple of two dictionaries, containing the indices of pruned heads and intermediate dimensions, respectively. If 'zs' is None, then both dictionaries will be None.
    """
    if zs is None:
        return None, None
    bert = model.bert if hasattr(model, "bert") else model.roberta

    if "head_z" in zs:
        head_z = zs.get("head_z", None)
        head_layer_z = zs.get("head_layer_z", None)

        prune_heads = {}
        for layer in range(len(head_z)):
            head_z_layer = head_z[layer].cpu().squeeze().clone()
            if head_layer_z is not None:
                head_z_layer *= head_layer_z[layer]
            index = torch.where(head_z_layer == 0)[0].tolist()
            prune_heads[layer] = index

            print(f"Layer {layer}, heads {' '.join([str(i) for i in index])} pruned.")
        model.prune_heads(prune_heads)

    kept_intermediate_dims = None
    if "intermediate_z" in zs:
        kept_intermediate_dims = {}
        intermediate_zs = zs["intermediate_z"]
        mlp_z = zs.get("mlp_z", None)
        for layer in range(len(intermediate_zs)):
            intermediate_z_layer = intermediate_zs[layer].squeeze()
            intermediate_z_layer = intermediate_z_layer.cpu().clone()
            if mlp_z is not None:
                intermediate_z_layer *= mlp_z[layer]
            kept_intermediate_dims[layer] = (
                intermediate_z_layer.nonzero().reshape(-1).tolist()
            )

    def prune_layer_norm(layernorm, index: int):
        layernorm.weight = torch.nn.parameter.Parameter(
            layernorm.weight.index_select(0, index)
        )
        layernorm.bias = torch.nn.parameter.Parameter(
            layernorm.bias.index_select(0, index)
        )
        layernorm.normalized_shape = (len(index),)

    def prune_layer(layer, index: int, dim: int):
        layer = prune_linear_layer(layer, index, dim=dim)
        return layer

    if "hidden_z" in zs:
        hidden_zs = zs["hidden_z"]
        index = torch.LongTensor(hidden_zs.squeeze().nonzero().squeeze().tolist())
        index = index.to(model.device)

        bert.embeddings.word_embeddings.weight = torch.nn.parameter.Parameter(
            bert.embeddings.word_embeddings.weight.index_select(1, index)
            .clone()
            .detach()
        )
        bert.embeddings.word_embeddings.embedding_dim = index.shape[0]
        bert.embeddings.position_embeddings.weight = torch.nn.parameter.Parameter(
            bert.embeddings.position_embeddings.weight.index_select(1, index)
            .clone()
            .detach()
        )
        bert.embeddings.position_embeddings.embedding_dim = index.shape[0]
        bert.embeddings.token_type_embeddings.weight = torch.nn.parameter.Parameter(
            bert.embeddings.token_type_embeddings.weight.index_select(1, index)
            .clone()
            .detach()
        )
        bert.embeddings.token_type_embeddings.embedding_dim = index.shape[0]
        prune_layer_norm(bert.embeddings.LayerNorm, index)

        for layer in range(0, 12):
            if bert.encoder.layer[layer].attention.self.query is not None:
                bert.encoder.layer[layer].attention.self.query = prune_layer(
                    bert.encoder.layer[layer].attention.self.query, index, dim=1
                )
                bert.encoder.layer[layer].attention.self.key = prune_layer(
                    bert.encoder.layer[layer].attention.self.key, index, dim=1
                )
            if bert.encoder.layer[layer].attention.self.value is not None:
                bert.encoder.layer[layer].attention.self.value = prune_layer(
                    bert.encoder.layer[layer].attention.self.value, index, dim=1
                )
                bert.encoder.layer[layer].attention.output.dense = prune_layer(
                    bert.encoder.layer[layer].attention.output.dense, index, dim=0
                )
                prune_layer_norm(
                    bert.encoder.layer[layer].attention.output.LayerNorm, index
                )
            if bert.encoder.layer[layer].intermediate.dense is not None:
                bert.encoder.layer[layer].intermediate.dense = prune_layer(
                    bert.encoder.layer[layer].intermediate.dense, index, dim=1
                )
                bert.encoder.layer[layer].output.dense = prune_layer(
                    bert.encoder.layer[layer].output.dense, index, dim=0
                )
                prune_layer_norm(bert.encoder.layer[layer].output.LayerNorm, index)

        # accommodate for different models
        if hasattr(model, "classifier") and hasattr(model.classifier, "dense"):
            model.classifier.dense = prune_linear_layer(
                model.classifier.dense, index, dim=1
            )
        if hasattr(model, "cls") and hasattr(model.cls, "dense"):
            model.cls.dense = prune_linear_layer(model.classifier.dense, index, dim=1)
        if hasattr(bert.pooler, "dense"):
            bert.pooler.dense = prune_linear_layer(bert.pooler.dense, index, dim=1)
        if hasattr(model, "qa_outputs"):
            model.qa_outputs = prune_linear_layer(model.qa_outputs, index, dim=1)
        if getattr(model, "layer_transformation", None) is not None:
            model.layer_transformation = prune_linear_layer(
                model.layer_transformation, index, dim=1
            )
            print("layer transformation", model.layer_transformation.weight.shape)
        if getattr(model, "mha_layer_transformation", None) is not None:
            model.mha_layer_transformation = prune_linear_layer(
                model.mha_layer_transformation, index, dim=1
            )
            print(
                "layer mha_layer_transformation",
                model.mha_layer_transformation.weight.shape,
            )

    if kept_intermediate_dims is not None:
        prune_intermediate_layers(model, kept_intermediate_dims)

    # for layer in range(0, 12):
    #     print("Layer:", layer)
    #     if bert.encoder.layer[layer].attention.self.query is not None:
    #         print("query:", bert.encoder.layer[layer].attention.self.query.weight.shape)
    #         print("key:", bert.encoder.layer[layer].attention.self.key.weight.shape)
    #     else:
    #         print("query:", None)
    #         print("key:", None)
    #     if bert.encoder.layer[layer].attention.self.value is not None:
    #         print("value:", bert.encoder.layer[layer].attention.self.value.weight.shape)
    #         print(
    #             "output:", bert.encoder.layer[layer].attention.output.dense.weight.shape
    #         )
    #     else:
    #         print("value:", None)
    #         print("output:", None)
    #     if bert.encoder.layer[layer].intermediate.dense is not None:
    #         print("up:", bert.encoder.layer[layer].intermediate.dense.weight.shape)
    #         print("down:", bert.encoder.layer[layer].output.dense.weight.shape)
    #     else:
    #         print("up", None)
    #         print("down", None)


def prune_intermediate_layers(model: nn.Module, keep_dims: Dict[int, List[int]]) -> None:
    """
    Prune intermediate layers of a BERT or RoBERTa model.

    Args:
        model (nn.Module): The BERT or RoBERTa model to prune.
        keep_dims (Dict[int, List[int]]): A dictionary where the keys are integers
            representing the layer number and the values are lists of integers
            representing the indices of the neurons to keep in the intermediate and
            output linear layers for that layer number. If the list of indices is
            empty, the corresponding linear layers will be removed.

    Returns:
        None.
    """
    bert = model.bert if hasattr(model, "bert") else model.roberta
    device = model.device
    for layer in keep_dims:
        if len(keep_dims[layer]) == 0:
            bert.encoder.layer[layer].intermediate.dense = None
            bert.encoder.layer[layer].output.dense = None
        else:
            bert.encoder.layer[layer].intermediate.dense = prune_linear_layer(
                bert.encoder.layer[layer].intermediate.dense,
                index=torch.LongTensor(keep_dims[layer]).to(device),
                dim=0,
            )
            bert.encoder.layer[layer].output.dense = prune_linear_layer(
                bert.encoder.layer[layer].output.dense,
                index=torch.LongTensor(keep_dims[layer]).to(device),
                dim=1,
            )


def load_zs(model_path: str) -> Optional[torch.Tensor]:
    """
    Load the zero shot (zs) tensor from a given model path.

    Args:
        model_path (str): Path to the model or zs.pt file.

    Returns:
        Optional[torch.Tensor]: The loaded zero shot tensor, or None if the file does not exist.
    """
    if model_path.endswith("zs.pt"):
        zs_path = model_path
    else:
        zs_path = os.path.join(model_path, "zs.pt")

    if os.path.exists(zs_path):
        zs = torch.load(zs_path, map_location="cpu")
        if zs is None:
            model_path = os.path.dirname(model_path)
            l0_module = torch.load(
                os.path.join(model_path, "l0_module.pt"), map_location="cpu"
            )
            zs = l0_module.forward(training=False)
        return zs
    else:
        return None


def load_pruned_model(model: nn.Module, weights: dict) -> nn.Module:
    """
    Loads the pruned model with the given weights.

    Args:
        model (PreTrainedModel): The pre-trained model to be pruned.
        weights (dict): The weights of the pruned model.

    Returns:
        PreTrainedModel: The pruned model with the given weights.
    """
    config = model.config
    dim_per_head = config.hidden_size // config.num_attention_heads
    zs = {}

    architecture = config.architectures[0].lower()
    bert_name = "roberta" if "roberta" in architecture else "bert"

    hidden_z = torch.zeros(config.hidden_size)
    hidden_z[: weights[f"{bert_name}.embeddings.word_embeddings.weight"].shape[1]] = 1
    zs["hidden_z"] = hidden_z

    head_z = torch.zeros(config.num_hidden_layers, config.num_attention_heads)
    head_layer_z = torch.zeros(config.num_hidden_layers)
    for i in range(config.num_hidden_layers):
        key = f"{bert_name}.encoder.layer.{i}.attention.output.dense.weight"
        if key in weights:
            remaining_heads = weights[key].shape[-1] // dim_per_head
            head_z[i, :remaining_heads] = 1
            head_layer_z[i] = 1
    zs["head_z"] = head_z
    zs["head_layer_z"] = head_layer_z

    int_z = torch.zeros(config.num_hidden_layers, config.intermediate_size)
    mlp_z = torch.zeros(config.num_hidden_layers)
    for i in range(config.num_hidden_layers):
        key = f"bert.encoder.layer.{i}.output.dense.weight"
        if key in weights:
            remaining_int_dims = weights[key].shape[-1]
            int_z[i, :remaining_int_dims] = 1
            mlp_z[i] = 1
    zs["intermediate_z"] = int_z
    zs["mlp_z"] = mlp_z

    prune_model_with_z(zs, model)
    model.load_state_dict(weights, strict=False)
    return model


def get_full_model_size(model_class: nn.Module, model_name: str) -> int:
    """
    Args:
        model_class (class): The class of the model to be instantiated.
        model_name (str): The name of the pre-trained model to load the weights from.

    Returns:
        int: The total number of parameters in the model
    """
    return calculate_parameters(model_class.from_pretrained(model_name))
