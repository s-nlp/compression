import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import (
    BertConfig,
    BertModel,
    BertPreTrainedModel,
    DistilBertConfig,
    DistilBertModel,
    DistilBertPreTrainedModel,
    RobertaConfig,
    RobertaModel,
    RobertaPreTrainedModel
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


class DistilBertForSpanClassification(DistilBertPreTrainedModel):
    def __init__(self, config):
        super(DistilBertForSpanClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.num_spans = config.num_spans  # number of spans per input

        self.model = DistilBertModel(config)
        self.dropout = nn.Dropout(config.dropout)
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
            # position_ids=position_ids,
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
