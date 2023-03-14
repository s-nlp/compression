from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import (
    BertModel,
    BertPreTrainedModel,
    DistilBertModel,
    DistilBertPreTrainedModel,
    PreTrainedModel,
    RobertaModel,
    RobertaPreTrainedModel,
)


class BertForSpanClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSpanClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.num_spans = config.num_spans  # number of spans per input

        self.model = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(
            config.num_spans * config.hidden_size, config.num_labels
        )

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        spans=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
            spans (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_spans, 2)`):
                Labels for position (index) of the start and end of the labelled span.
                We assume a fixed number of spans per input and max pool over the span
                to get a span representation.
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
                Labels for computing the sequence classification/regression loss.
                Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
                If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
                If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        Returns:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
            loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
                Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
            start_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
                Span-start scores (before SoftMax).
            end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
                Span-end scores (before SoftMax).
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.
                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.
        Examples::
            from transformers import BertTokenizer, BertForQuestionAnswering
            import torch
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
            question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
            input_ids = tokenizer.encode(question, text)
            token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
            start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
            all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
            answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
            assert answer == "a nice puppet"
        """

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        # extract representations correponding to the spans
        span_reps = []
        for batch_spans in spans.transpose(0, 1):
            # spans: n_spans x bz x 2 (start, end)

            span_mask = torch.zeros_like(sequence_output)
            for batch_idx, (start, end) in enumerate(batch_spans):
                span_mask[batch_idx, start : end + 1] = 1

            # pool within spans
            masked_output = sequence_output * span_mask
            span_rep, _ = masked_output.max(dim=1)
            span_reps.append(span_rep)

        # combine multiple spans
        span_reps = torch.cat(span_reps, dim=1)
        # feed to ff classifier
        span_reps = self.dropout(span_reps)
        logits = self.classifier(span_reps)
        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        # print(torch.argmax(outputs[1].detach().cpu(), dim=-1))
        # print(labels)
        return outputs  # (loss), loegits, (hidden_states), (attentions)


class DistilBertForSpanClassification(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels: int = config.num_labels
        self.num_spans: int = config.num_spans
        self.model: DistilBertModel = DistilBertModel(config)
        self.dropout: nn.Dropout = nn.Dropout(config.dropout)
        self.classifier: nn.Linear = nn.Linear(
            config.num_spans * config.hidden_size, config.num_labels
        )
        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        spans: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        r"""
            spans (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_spans, 2)`):
                Labels for position (index) of the start and end of the labelled span.
                We assume a fixed number of spans per input and max pool over the span
                to get a span representation.
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
                Labels for computing the sequence classification/regression loss.
                Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
                If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
                If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        Returns:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
            loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
                Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
            start_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
                Span-start scores (before SoftMax).
            end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
                Span-end scores (before SoftMax).
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.
                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.
        """

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        # extract representations correponding to the spans
        span_reps = []
        for batch_spans in spans.transpose(0, 1):
            # spans: n_spans x bz x 2 (start, end)

            span_mask = torch.zeros_like(sequence_output)
            for batch_idx, (start, end) in enumerate(batch_spans):
                span_mask[batch_idx, start : end + 1] = 1

            # pool within spans
            masked_output = sequence_output * span_mask
            span_rep, _ = masked_output.max(dim=1)
            span_reps.append(span_rep)

        # combine multiple spans
        span_reps = torch.cat(span_reps, dim=1)
        # feed to ff classifier
        span_reps = self.dropout(span_reps)
        logits = self.classifier(span_reps)
        outputs = (logits,) + outputs[
            2:
        ]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class RobertaForSpanClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RobertaForSpanClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.num_spans = config.num_spans  # number of spans per input

        self.model = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob or config.dropout)
        self.classifier = nn.Linear(
            config.num_spans * config.hidden_size, config.num_labels
        )

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        spans=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
            spans (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_spans, 2)`):
                Labels for position (index) of the start and end of the labelled span.
                We assume a fixed number of spans per input and max pool over the span
                to get a span representation.
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
                Labels for computing the sequence classification/regression loss.
                Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
                If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
                If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        Returns:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
            loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
                Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
            start_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
                Span-start scores (before SoftMax).
            end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
                Span-end scores (before SoftMax).
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.
                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.
        Examples::
            from transformers import DistilbertTokenizer, DistilbertForQuestionAnswering
            import torch
            tokenizer = DistilbertTokenizer.from_pretrained('bert-base-uncased')
            model = DistilbertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
            question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
            input_ids = tokenizer.encode(question, text)
            token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
            start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
            all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
            answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
            assert answer == "a nice puppet"
        """

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        # extract representations correponding to the spans
        span_reps = []
        for batch_spans in spans.transpose(0, 1):
            # spans: n_spans x bz x 2 (start, end)

            span_mask = torch.zeros_like(sequence_output)
            for batch_idx, (start, end) in enumerate(batch_spans):
                span_mask[batch_idx, start : end + 1] = 1

            # pool within spans
            masked_output = sequence_output * span_mask
            span_rep, _ = masked_output.max(dim=1)
            span_reps.append(span_rep)

        # combine multiple spans
        span_reps = torch.cat(span_reps, dim=1)
        # feed to ff classifier
        span_reps = self.dropout(span_reps)
        logits = self.classifier(span_reps)
        outputs = (logits,) + outputs[
            2:
        ]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class FCLayer(nn.Module):
    """
    A fully connected layer with an optional activation function and dropout.

    Args:
        input_dim (int): The number of input features.
        output_dim (int): The number of output features.
        dropout_rate (float, optional): The dropout rate to use. Defaults to 0.0.
        use_activation (bool, optional): Whether to use an activation function. Defaults to True.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout_rate: float = 0.0,
        use_activation: bool = True,
    ):
        super().__init__()

        layers = [nn.Dropout(dropout_rate)]
        if use_activation:
            layers.append(nn.Tanh())
        layers.append(nn.Linear(input_dim, output_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Perform a forward pass through the fully connected layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.layers(x)


class SpanClassificationModel(nn.Module):
    """
    Model for span-level relation classification.

    Args:
        backbone (PreTrainedModel): Pre-trained model from transformers library.
        num_labels (int): Number of possible relation labels.
    """

    def __init__(self, backbone: PreTrainedModel, num_labels: int):
        super().__init__()
        self.backbone = backbone
        self.num_labels = num_labels
        self.config = backbone.config

        hidden_size = self.config.hidden_size

        if hasattr(self.config, "classifier_dropout"):
            dropout_rate = (
                self.config.classifier_dropout
                if self.config.classifier_dropout is not None
                else self.config.hidden_dropout_prob
            )
        else:
            dropout_rate = self.config.classifier_dropout_prob

        self.cls_fc_layer = FCLayer(hidden_size, hidden_size, dropout_rate)
        self.e1_fc_layer = FCLayer(hidden_size, hidden_size, dropout_rate)
        self.e2_fc_layer = FCLayer(hidden_size, hidden_size, dropout_rate)
        self.label_classifier = FCLayer(
            hidden_size * 3, num_labels, dropout_rate, use_activation=False
        )

    @staticmethod
    def _entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)

        Args:
            hidden_output: [batch_size, j-i+1, dim]
            e_mask: [batch_size, seq_len]
            (e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0])
        Returns:
            torch.Tensor: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(2)  # [batch_size, seq_len, 1]
        length_tensor = (e_mask_unsqueeze != 0).sum(dim=1)  # [batch_size, 1]

        sum_vector = (hidden_output * e_mask_unsqueeze).sum(1)
        return sum_vector / length_tensor

    def forward(
        self, input_ids, attention_mask, e1_mask, e2_mask, token_type_ids, labels=None
    ):
        """
        Compute the forward pass of the SpanClassificationModel.

        Args:
            input_ids: torch.Tensor[batch_size, seq_len]
            attention_mask: torch.Tensor[batch_size, seq_len]
            e1_mask: torch.Tensor[batch_size, seq_len]
            e2_mask: torch.Tensor[batch_size, seq_len]
            token_type_ids: torch.Tensor[batch_size, seq_len], optional
            labels: torch.Tensor[batch_size, num_labels], optional
        Returns:
            Tuple of (loss, logits, hidden_states, attentions), where:
            - loss (optional): scalar tensor representing the cross-entropy loss
            - logits: [batch_size, num_labels] tensor of unnormalized predictions
            - hidden_states (optional): tuple of hidden states from the transformer backbone
            - attentions (optional): tuple of attention matrices from the transformer backbone
        """
        # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output, pooled_output, *outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False,
        )

        # Average
        hidden_first = self._entity_average(sequence_output, e1_mask)
        hidden_second = self._entity_average(sequence_output, e2_mask)

        # Dropout -> tanh -> fc_layer
        pooled_output = self.cls_fc_layer(pooled_output)
        hidden_first = self.e1_fc_layer(hidden_first)
        hidden_second = self.e2_fc_layer(hidden_second)

        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, hidden_first, hidden_second], dim=-1)
        logits = self.label_classifier(concat_h)

        outputs = (logits,) + tuple(
            outputs
        )  # add hidden states and attention if they are here

        # Softmax
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class EntityChoiceModel(nn.Module):
    """
    A PyTorch module for entity choice task.

    Args:
        backbone (PreTrainedModel): The pre-trained language model to use as backbone.
    """

    def __init__(self, backbone: PreTrainedModel):
        super().__init__()
        self.backbone = backbone

        hidden_size = backbone.config.hidden_size

        if hasattr(backbone.config, "classifier_dropout"):
            dropout_rate = (
                backbone.config.classifier_dropout
                if backbone.config.classifier_dropout is not None
                else backbone.config.hidden_dropout_prob
            )
        else:
            dropout_rate = backbone.config.classifier_dropout_prob

        self.clf = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 1),
        )

        self.loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self, input_ids, attention_mask, entity_mask, token_type_ids, labels=None
    ):
        """
        Perform forward pass for entity choice task.

        Args:
            input_ids (torch.Tensor): Input IDs of shape (batch_size, seq_len).
            attention_mask (torch.Tensor): Attention mask of shape (batch_size, seq_len).
            entity_mask (torch.Tensor): Entity mask of shape (batch_size, num_entities, seq_len).
            token_type_ids (torch.Tensor): Token type IDs of shape (batch_size, seq_len).
            labels (torch.Tensor, optional): Target labels of shape (batch_size, num_entities).

        Returns:
            Tuple[torch.Tensor, ...]:
                If `labels` is None, returns a tuple containing:
                - logits (torch.Tensor): Logits of shape (batch_size, num_entities).
                - other outputs from backbone (if any).

                If `labels` is not None, returns a tuple containing:
                - reduced_loss (torch.Tensor): The average loss across entities in the batch.
                - other outputs from backbone (if any).
        """
        sequence_output, _, *outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False,
        )

        entity_mask = entity_mask.unsqueeze(3)
        entity_lengths = entity_mask.sum(2)

        # [batch_size, num_entities, seq_len, hid_dim]
        embeds_for_entities = sequence_output.unsqueeze(1) * entity_mask
        aggregated_embeds = embeds_for_entities.sum(2) / (entity_lengths + 1e-8)

        # [batch_size, num_entities]
        logits = self.clf(aggregated_embeds).squeeze(2)

        # due to padding, we might have rows without entity indices at all
        # thus, we need to mask the logits/loss for them

        # [batch_size, num_entities]
        present_entities = (entity_lengths.squeeze(2) != 0).to(logits.dtype)
        logits = logits * present_entities - 10000.0 * (1 - present_entities)

        outputs = (logits,) + tuple(
            outputs
        )  # add hidden states and attention if they are here

        if labels is not None:
            # do not penalize predictions for padded entities
            label_mask = labels != -1

            # BCEWithLogitsLoss requires targets to be from 0 to 1
            labels[~label_mask] = 0

            loss = self.loss(logits, labels.to(logits.dtype))

            reduced_loss = (
                (loss * label_mask).sum(1) / (label_mask.sum(1) + 1e-8)
            ).mean()
            outputs = (reduced_loss,) + outputs

        return outputs
