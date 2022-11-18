from .lobert_utils import lobert_student
from .dummy_model import dummy_func
from .head_pruning import random_head_pruning_model
from .simple_svd import simple_svd_func
from .unstructured_prune import uns_prune
from .structured_prune import str_prune

from collections import OrderedDict
MODEL_NAMES = OrderedDict(
    [
        # Base model mapping
        ("dummy", "dummy_self"),
        ("random_head", "random_head_pruning"),
        ("svd_selfatt", "lobert_self_svd"),
        ("svd_fnn", "lobert_self_ffn"),
        ("our_fnn", "simple_svd_model"),
        ("str_prune", "structured_pruning"),
        ("uns_prune", "unstructured_pruning")
    ])

def structured_pruning(model):
    model = str_prune(model, 0.6)
    return model

def unstructured_pruning(model):
    model = uns_prune(model, 0.6)
    return model

def simple_svd_model(model, rank = 150):
    model = simple_svd_func(model, rank)
    return model


def dummy_self(model):
    model = dummy_func(model)
    return model

def random_head_pruning(model):
    model = random_head_pruning_model(model)
    return model

def lobert_self_svd(model, rank = 150):
    model = lobert_student(model.config, 
        model, 
        student_mode = "self_svd", rank=rank,
        num_hidden_layers = model.config.num_hidden_layers)
    return model

def lobert_self_ffn(model):
    model = lobert_student(model.config, 
        model, 
        student_mode = "ffn_svd", 
        num_hidden_layers = model.config.num_hidden_layers)
    return model