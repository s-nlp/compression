from .lobert_utils import lobert_student
from .dummy_model import dummy_func
from .head_pruning import random_head_pruning_model
from .svd_bert_model import svd_bert
from .bert_pruning_l1 import bert_prune

from collections import OrderedDict
MODEL_NAMES = OrderedDict(
    [
        # Base model mapping
        ("dummy", "dummy_self"),
        ("random_head", "random_head_pruning"),
        ("svd_selfatt", "lobert_self_svd"),
        ("svd_fnn", "lobert_self_ffn"),
        ("pruning_l1", "pruning_l1_self"),
        ("svd_bert", "svd_bert_self"),
    ])


def dummy_self(model):
    model = dummy_func(model)
    return model

def random_head_pruning(model):
    model = random_head_pruning_model(model)
    return model

def lobert_self_svd(model):
    model = lobert_student(model.config, 
        model, 
        student_mode = "self_svd", 
        num_hidden_layers = model.config.num_hidden_layers)
    return model

def lobert_self_ffn(model):
    model = lobert_student(model.config, 
        model, 
        student_mode = "ffn_svd", 
        num_hidden_layers = model.config.num_hidden_layers)
    return model
    
def pruning_l1_self(model):
    mdl = bert_prune(model)
    return mdl
    
def svd_bert_self(model):
    mdl = svd_bert(model)
    return mdl
