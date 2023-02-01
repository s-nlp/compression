'''

Bert compression via Product Quantization method.

'''


import copy
import torch.nn as nn

from src.product_quantization.pq_modules import PQEmbedding, PQLinear


def replace_linear_layers(model, dim=None, num_clusters=256):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_linear_layers(module, dim, num_clusters)
            
        if isinstance(module, nn.Linear):
            dim = module.weight.data.size()[0] if dim is None else dim
            setattr(model, n, PQLinear(module, dim=dim, num_clusters=num_clusters))


def quantize_bert_linears(model):
    quantized_model = copy.deepcopy(model)
    replace_linear_layers(quantized_model)
    return quantized_model


def quantize_bert_vocab(model, dim=256):
    quantized_model = copy.deepcopy(model)
    quantized_model.bert.embeddings.word_embeddings = PQEmbedding(model.bert.embeddings.word_embeddings, dim=dim)
    return quantized_model


def quantize_bert(model, quant_vocab=True, quant_linears=True, half_precision=True):
    '''

    quant_vocab: if True, compress ONLY word embeddings
    quant_linears: if True, compress ALL linear layers
    half_precision: if True, use the half precision (float16)

    Usage Example: 

        model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased')
        quantinized_model = quantize_bert(model)

    '''
    
    if quant_vocab:
        model = quantize_bert_vocab(model)

    if quant_linears:
        model = quantize_bert_linears(model)

    if half_precision:
        quantized_model = copy.deepcopy(model).half()
        model = quantized_model

    return model
