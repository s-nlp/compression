import logging
import math
import os
from typing import List, Optional, Tuple, Union

import torch
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from transformers import AutoConfig
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    SequenceClassifierOutput,
)
from transformers.modeling_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.models.bert.modeling_bert import (
    BertAttention,
    BertEmbeddings,
    BertEncoder,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertLayer,
    BertModel,
    BertOutput,
    BertSelfAttention,
    BertSelfOutput,
    QuestionAnsweringModelOutput,
)

from utils.cofi_utils import (
    initialize_layer_transformation,
    load_from_tsv,
    load_l0_module,
    load_model,
    load_model_with_zs,
    load_pruned_model,
    load_zs,
    prune_intermediate_layers,
    prune_model_with_z,
    update_params,
)

# from transformers.utils.hub import cached_path, hf_bucket_url


logger = logging.getLogger(__name__)


class CoFiLayerNorm(torch.nn.LayerNorm):
    """
    This class implements a custom layer normalization function that can selectively apply layer normalization to only certain parts of the input.

    Parameters:
    normalized_shape (int or tuple of ints): Shape of the input tensor.
    eps (float, optional): A value added to the denominator for numerical stability. Default is 1e-5.
    elementwise_affine (bool, optional): If True, this module has learnable per-element affine parameters. Default is True.

    Methods:
    forward(input, hidden_z=None):
    Performs a forward pass through the layer normalization function to normalize the input tensor.
    If a binary mask hidden_z is provided,
    normalization is only applied to the unmasked elements of the input tensor.

    Returns:
    The normalized output tensor.
    """

    def __init__(
        self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True
    ) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine)

    def forward(
        self, input: torch.Tensor, hidden_z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Given an input tensor and an optional binary mask, applies layer normalization on the input tensor only at the positions specified by the mask.

        Args:
            input (torch.Tensor): The input tensor.
            hidden_z (torch.Tensor, optional): A binary mask indicating the positions to normalize in the input tensor. Defaults to None.

        Returns:
            A tensor with the same shape as the input tensor after applying layer normalization only at the masked positions.
        """
        if hidden_z is not None:
            remaining_index = torch.where(~hidden_z.eq(0))[0]
            compressed_input = torch.index_select(input, dim=-1, index=remaining_index)
            compressed_weight = self.weight[remaining_index]
            compressed_bias = self.bias[remaining_index]
            normalized_shape = len(remaining_index)
            normed_input = F.layer_norm(
                compressed_input,
                [normalized_shape],
                compressed_weight,
                compressed_bias,
                self.eps,
            )
            output = input.clone()
            output[:, :, remaining_index] = normed_input
        else:
            output = F.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps
            )
        return output


class CoFiBertForSequenceClassification(BertForSequenceClassification):
    """
    A BERT-based model with an additional option for layer distillation.

    Inherits from `BertForSequenceClassification` and includes an additional option for layer distillation.
    """

    def __init__(self, config):
        super().__init__(config)
        self.bert = CoFiBertModel(config)

        self.do_layer_distill = getattr(config, "do_layer_distill", False)

        if self.do_layer_distill:
            self.layer_transformation = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.layer_transformation = None

        # @classmethod
        # def from_pretrained(
        #     cls,
        #     pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        #     *model_args,
        #     **kwargs
        # ):
        # """
        # Instantiates the model from a pre-trained model configuration and checkpoint.

        # Args:
        #     pretrained_model_name_or_path: The path or name of the pre-trained model to load.
        #     *model_args: Additional arguments to pass to the underlying model.
        #     **kwargs: Additional keyword arguments to pass to the underlying model.

        # Returns:
        #     The instantiated `CoFiBertForSequenceClassification` model.

        # """
        # if os.path.exists(pretrained_model_name_or_path):
        #     weights = torch.load(
        #         os.path.join(pretrained_model_name_or_path, "pytorch_model.bin"),
        #         map_location=torch.device("cpu"),
        #     )
        # else:
        #     archive_file = hf_bucket_url(
        #         pretrained_model_name_or_path, filename="pytorch_model.bin"
        #     )
        #     resolved_archive_file = cached_path(archive_file)
        #     weights = torch.load(resolved_archive_file, map_location="cpu")

        # # Convert old format to new format if needed from a PyTorch state_dict
        # old_keys = []
        # new_keys = []
        # for key in weights.keys():
        #     new_key = None
        #     if "gamma" in key:
        #         new_key = key.replace("gamma", "weight")
        #     if "beta" in key:
        #         new_key = key.replace("beta", "bias")
        #     if new_key:
        #         old_keys.append(key)
        #         new_keys.append(new_key)
        # for old_key, new_key in zip(old_keys, new_keys):
        #     weights[new_key] = weights.pop(old_key)

        # if "config" not in kwargs:
        #     config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        #     config.do_layer_distill = False
        # else:
        #     config = kwargs["config"]

        # model = cls(config)

        # load_pruned_model(model, weights)
        # return model

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        head_z: Optional[torch.Tensor] = None,
        head_layer_z: Optional[torch.Tensor] = None,
        intermediate_z: Optional[torch.Tensor] = None,
        mlp_z: Optional[torch.Tensor] = None,
        hidden_z: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, ...], SequenceClassifierOutput,]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            head_z=head_z,
            head_layer_z=head_layer_z,
            intermediate_z=intermediate_z,
            mlp_z=mlp_z,
            hidden_z=hidden_z,
        )  # [32, 68, 768]

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [32, 3]

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class CoFiBertEmbeddings(BertEmbeddings):
    """
    Inherit from BertEmbeddings to allow CoFiLayerNorm.

    Args:
        config (BertConfig): The config class to initialize the embeddings.
    """

    def __init__(self, config):
        super().__init__(config)
        self.layernorm = CoFiLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        hidden_z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the embeddings layer.

        Args:
            input_ids (torch.Tensor, optional): The input token IDs of shape
                `(batch_size, sequence_length)`. Defaults to None.
            token_type_ids (torch.Tensor, optional): The token type IDs of shape
                `(batch_size, sequence_length)`. Defaults to None.
            position_ids (torch.Tensor, optional): The position IDs of shape
                `(batch_size, sequence_length)`. Defaults to None.
            inputs_embeds (torch.Tensor, optional): The input embeddings of shape
                `(batch_size, sequence_length, hidden_size)`. Defaults to None.
            hidden_z (torch.Tensor, optional): The hidden activation values of the
                previous layer of shape `(batch_size, sequence_length, hidden_size)`.
                Defaults to None.

        Returns:
            embeddings (torch.Tensor): The output embeddings of shape
                `(batch_size, sequence_length, hidden_size)`.
        """
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if position_ids is None:
            seq_length = input_shape[1]

            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=self.position_ids.device
            )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings

        if hidden_z is not None:
            embeddings = embeddings.mul(hidden_z)
        embeddings = self.layernorm(embeddings, hidden_z)
        embeddings = self.dropout(embeddings)

        if hidden_z is not None:
            embeddings = embeddings.mul(hidden_z)
        return embeddings


class CoFiBertModel(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = CoFiBertEncoder(config)
        self.embeddings = CoFiBertEmbeddings(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        head_layer_z=None,
        head_z=None,
        intermediate_z=None,
        mlp_z=None,
        hidden_z=None,
    ):

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            hidden_z=hidden_z,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            intermediate_z=intermediate_z,
            head_z=head_z,
            mlp_z=mlp_z,
            head_layer_z=head_layer_z,
            hidden_z=hidden_z,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CoFiBertEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList(
            [CoFiBertLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
        head_z=None,
        head_layer_z=None,
        intermediate_z=None,
        mlp_z=None,
        hidden_z=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                output_attentions,
                intermediate_z=intermediate_z[i] if intermediate_z is not None else None,
                head_z=head_z[i] if head_z is not None else None,
                mlp_z=mlp_z[i] if mlp_z is not None else None,
                head_layer_z=head_layer_z[i] if head_layer_z is not None else None,
                hidden_z=hidden_z,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class CoFiBertLayer(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = CoFiBertAttention(config)
        self.output = CoFiBertOutput(config)
        self.config = config

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        head_z=None,
        head_layer_z=None,
        intermediate_z=None,
        mlp_z=None,
        hidden_z=None,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            head_z=head_z,
            head_layer_z=head_layer_z,
            hidden_z=hidden_z,
        )

        attention_output = self_attention_outputs[0]
        # add self attentions if we output attention weights
        outputs = self_attention_outputs[1:]

        if self.intermediate.dense is None:
            layer_output = attention_output
        else:
            self.intermediate_z = intermediate_z
            self.mlp_z = mlp_z
            self.hidden_z = hidden_z
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                attention_output,
            )
        outputs = (layer_output,) + outputs + (attention_output,)
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        if self.intermediate_z is not None:
            intermediate_output = intermediate_output.mul(self.intermediate_z)
        layer_output = self.output(
            intermediate_output, attention_output, self.mlp_z, self.hidden_z
        )
        return layer_output


class CoFiBertAttention(BertAttention):
    """
    This class overrides the BertAttention module to implement CoFiBERT attention mechanism.
    CoFiBERT is a collaborative filtering based BERT architecture for personalized language
    understanding.

    Inherits from:
        `BertAttention`: The base class for all attention modules in BERT models.

    Args:
        config: A `BertConfig` object containing the configuration of the BERT model.

    Attributes:
        self: A `CoFiBertSelfAttention` object representing the self-attention mechanism.
        output: A `CoFiBertSelfOutput` object representing the output layer for the self-attention
            mechanism.
        config: A `BertConfig` object containing the configuration of the BERT model.

    """

    def __init__(self, config):
        super().__init__(config)
        self.self = CoFiBertSelfAttention(config)
        self.output = CoFiBertSelfOutput(config)
        self.config = config

    def prune_heads(self, heads: List[int]) -> None:
        """
        Prunes heads from the self-attention mechanism of the CoFiBERT module.

        Args:
            heads: A list of integers representing the heads to prune.

        """
        len_heads = len(heads)
        if len_heads == 0:
            return

        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        if len(index) == 0:
            self.self.query = None
            self.self.key = None
            self.self.value = None
            self.output.dense = None
        else:
            self.self.query = prune_linear_layer(self.self.query, index)
            self.self.key = prune_linear_layer(self.self.key, index)
            self.self.value = prune_linear_layer(self.self.value, index)
            self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        head_z: Optional[torch.Tensor] = None,
        head_layer_z: Optional[torch.Tensor] = None,
        hidden_z: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Applies the CoFiBertAttention module to the input tensor.

        Args:
            hidden_states (torch.Tensor): The input tensor of shape `(batch_size, sequence_length, hidden_size)`.
            attention_mask (torch.Tensor, optional): The attention mask tensor of shape `(batch_size, sequence_length)`.
                Defaults to `None`.
            output_attentions (bool, optional): Whether to output the attention probabilities tensor. Defaults to `False`.
            head_z (torch.Tensor, optional): The head-wise scaling factors tensor of shape `(batch_size, num_attention_heads)`.
                Defaults to `None`.
            head_layer_z (torch.Tensor, optional): The head-layer-wise scaling factors tensor of shape `(batch_size, num_attention_heads, num_hidden_layers)`.
                Defaults to `None`.
            hidden_z (torch.Tensor, optional): The hidden-state-wise scaling factors tensor of shape `(batch_size, hidden_size)`.
                Defaults to `None`.

        Returns:
            Tuple
        """
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            output_attentions,
            head_z=head_z,
        )

        attention_output = self.output(
            self_outputs[0], hidden_states, head_layer_z=head_layer_z, hidden_z=hidden_z
        )
        return (attention_output,) + self_outputs[1:]


class CoFiBertSelfAttention(BertSelfAttention):
    def __init__(self, config):
        """
        Initializes the CoFiBertSelfAttention layer.

        Args:
            config: a dictionary-like object containing the configuration parameters
                of the CoFiBertSelfAttention layer.

        Raises:
            ValueError: If the hidden size is not a multiple of the number of attention heads.
        """
        super().__init__(config)
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.config = config

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transposes the input tensor for applying the attention scores.

        Args:
            x: a tensor with shape (batch_size, seq_len, hidden_size).

        Returns:
            A tensor with shape (batch_size, num_heads, seq_len, head_size).
        """
        x_shape = x.size()
        last_dim = x_shape[-1]
        size_per_head = last_dim // self.num_attention_heads
        new_x_shape = x_shape[:-1] + (self.num_attention_heads, size_per_head)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        head_z: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Applies the CoFiBertSelfAttention layer.

        Args:
            hidden_states: a tensor with shape (batch_size, seq_len, hidden_size).
            attention_mask: an optional tensor with shape (batch_size, seq_len, seq_len).
            output_attentions: a bool that indicates if the attention weights should
                be returned.
            head_z: an optional tensor with shape (batch_size, num_heads) used to weight
                the attention scores.

        Returns:
            A tuple with one or two elements:
                - A tensor with shape (batch_size, seq_len, hidden_size).
                - A tensor with shape (batch_size, num_heads, seq_len, seq_len).
                  Only returned if output_attentions=True.
        """
        if self.value is None:
            return (None, None) if output_attentions else (None,)

        query_hidden_states = hidden_states
        mixed_query_layer = self.query(query_hidden_states)

        key_hidden_states = hidden_states
        mixed_key_layer = self.key(key_hidden_states)

        value_hidden_states = hidden_states
        mixed_value_layer = self.value(value_hidden_states)

        batch_size, seq_length, _ = hidden_states.shape

        if not hasattr(self, "ones"):
            self.ones = (
                torch.ones(batch_size, seq_length, seq_length)
                .float()
                .to(hidden_states.device)
            )

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        value_layer = self.transpose_for_scores(mixed_value_layer)
        context_layer = torch.matmul(attention_probs, value_layer)
        if head_z is not None:
            context_layer *= head_z

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            context_layer.shape[-1] * context_layer.shape[-2],
        )
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )
        return outputs


class CoFiBertSelfOutput(BertSelfOutput):
    """
    A self-attention output layer with CoFiLayerNorm.

    Inherits from `BertSelfOutput`.

    Args:
        config (BertConfig): The config class to initialize the layer.

    Attributes:
        dense (nn.Linear): The dense layer.
        layernorm (CoFiLayerNorm): The CoFiLayerNorm layer.
        dropout (nn.Dropout): The dropout layer.
        config (BertConfig): The layer's configuration.
    """

    def __init__(self, config):
        super().__init__(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layernorm = CoFiLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
        head_layer_z: Optional[torch.Tensor] = None,
        hidden_z: Optional[torch.Tensor] = None,
        inference: bool = False,
    ) -> torch.Tensor:
        """
        Perform a forward pass through the layer.

        Args:
            hidden_states (torch.Tensor): The hidden states tensor.
            input_tensor (torch.Tensor): The input tensor.
            head_layer_z (Optional[torch.Tensor]): The head layer z tensor.
            hidden_z (Optional[torch.Tensor]): The hidden z tensor.
            inference (bool): Whether inference mode is on.

        Returns:
            The tensor resulting from the forward pass through the layer.
        """
        if hidden_states is None:
            return input_tensor
        hidden_states = self.dense(hidden_states)
        if head_layer_z is not None:
            hidden_states = hidden_states.mul(head_layer_z)
        if not inference and hidden_states.sum().eq(0).item():
            hidden_states = hidden_states + input_tensor
        else:
            if hidden_z is not None:
                hidden_states = hidden_states.mul(hidden_z)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.layernorm(hidden_states + input_tensor, hidden_z)
            if hidden_z is not None:
                hidden_states = hidden_states.mul(hidden_z)
        return hidden_states


class CoFiBertOutput(BertOutput):
    """
    A modified BertOutput layer that applies CoFiLayerNorm instead of LayerNorm.
    """

    def __init__(self, config):
        """
        Initialize the CoFiBertOutput layer.

        Args:
            config (BertConfig): The configuration of the Bert model.
        """
        super().__init__(config)
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layernorm = CoFiLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
        mlp_z: Optional[torch.Tensor] = None,
        hidden_z: Optional[torch.Tensor] = None,
        inference: bool = False,
    ) -> torch.Tensor:
        """
        Apply the CoFiBertOutput layer to the input tensor.

        Args:
            hidden_states (torch.Tensor): The hidden states from the previous layer.
            input_tensor (torch.Tensor): The input tensor to the CoFiBertOutput layer.
            mlp_z (Optional[torch.Tensor]): The coefficient tensor for the MLP output.
            hidden_z (Optional[torch.Tensor]): The coefficient tensor for the hidden states.
            inference (bool): Whether the model is in inference mode.

        Returns:
            torch.Tensor: The output tensor from the CoFiBertOutput layer.
        """
        hidden_states = self.dense(hidden_states)
        if mlp_z is not None:
            hidden_states *= mlp_z
        if not inference and hidden_states.sum().eq(0).item():
            return hidden_states + input_tensor
        else:
            if hidden_z is not None:
                hidden_states = hidden_states.mul(hidden_z)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.layernorm(hidden_states + input_tensor, hidden_z)
            if hidden_z is not None:
                hidden_states = hidden_states.mul(hidden_z)
        return hidden_states


class CoFiBertForQuestionAnswering(BertForQuestionAnswering):
    def __init__(self, config):
        super().__init__(config)
        self.bert = CoFiBertModel(config)
        self.do_layer_distill = getattr(config, "do_layer_distill", False)

        if self.do_layer_distill:
            self.layer_transformation = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.layer_transformation = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        **kwargs
    ):
        if os.path.exists(pretrained_model_name_or_path):
            weights = torch.load(
                os.path.join(pretrained_model_name_or_path, "pytorch_model.bin"),
                map_location=torch.device("cpu"),
            )
        else:
            archive_file = hf_bucket_url(
                pretrained_model_name_or_path, "pytorch_model.bin"
            )
            resolved_archive_file = cached_path(archive_file)
            weights = torch.load(resolved_archive_file, map_location="cpu")

        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in weights.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            weights[new_key] = weights.pop(old_key)

        drop_weight_names = ["layer_transformation.weight", "layer_transformation.bias"]
        for name in drop_weight_names:
            if name in weights:
                weights.pop(name)

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        config.do_layer_distill = False
        model = cls(config)

        load_pruned_model(model, weights)
        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        intermediate_z=None,
        head_z=None,
        mlp_z=None,
        head_layer_z=None,
        hidden_z=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            intermediate_z=intermediate_z,
            head_z=head_z,
            head_layer_z=head_layer_z,
            mlp_z=mlp_z,
            hidden_z=hidden_z,
        )

        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
