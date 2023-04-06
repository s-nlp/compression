from .lobert_utils import lobert_student
from .dummy_model import dummy_func
from .head_pruning import random_head_pruning_model
from .drone.drone import drone_bert
from .simple_svd import simple_svd_func, w_svd_func, w_svd_func_T
from .unstructured_prune import uns_prune
from .structured_prune import str_prune
from .ttm.ttm_compress_bert import ttm_compress_bert_ffn
from .ttm_daniel.ttm_alt_compress_bert import (ttm_alt_compress_bert_ffn, 
                                               svd_alt_compress_bert_ffn,
                                            )


from collections import OrderedDict
MODEL_NAMES = OrderedDict(
    [
        # Base model mapping
        ("dummy", "dummy_self"),
        ("random_head", "random_head_pruning"),
        ("svd_selfatt", "lobert_self_svd"),
        ("svd_ffn", "lobert_self_ffn"),
        ("our_ffn", "simple_svd_model"),
        ("str_prune", "structured_pruning"),
        ("uns_prune", "unstructured_pruning"),
        ("ttm_ffn", "apply_ttm_compress_bert_ffn"),
        ("weighted_ttm_ffn", "apply_ttm_compress_bert_ffn"),
        ("fwsvd_ffn", "apply_fwsvd_compress_bert_ffn"),
        ("svd_ffn_w", "weight_svd_model"),
        ("svd_ffn_w_T", "weight_svd_T_model"),
        ("uns_prune", "unstructured_pruning"),
        ("drone", "apply_drone_compress_bert")
        ("ttm_ffn", "apply_ttm_compress_bert_ffn"),
        ("ttm_ffn_alt", "apply_alt_ttm_compress_bert_ffn"),
        ("ttm_ffn_alt_w", "apply_alt_ttm_w_compress_bert_ffn"),
        
        ("ttm_ffn_w_inv", "apply_ttm_compress_bert_ffn_w_inv"),
        ("ttm_ffn_w", "apply_ttm_compress_bert_ffn_w"),
        
        ("svd_ffn_alt", "apply_alt_svd_compress_bert_ffn"),
        ("svd_ffn_alt_w", "apply_alt_w_svd_invasive_compress_bert_ffn"),
    ])

#dummy model
def dummy_self(model, *args, **kwargs):
    model = dummy_func(model)
    return model

#simple TTM from vika source code
def apply_ttm_compress_bert_ffn(model, ranks, input_dims, output_dims):
    model = ttm_compress_bert_ffn(model, ranks, input_dims, output_dims, 
                                  with_checkpoints=False)
    return model

#simple TTM from daniel source code
def apply_alt_ttm_compress_bert_ffn(model, ranks, input_dims, output_dims):
    model = ttm_alt_compress_bert_ffn(model, ranks, input_dims, output_dims)
    return model

#TTM weight non-invasive from daniel source code
def apply_alt_ttm_w_compress_bert_ffn(model, ranks, input_dims, output_dims, 
                                weight_int, weight_out, weight_count):
    model = ttm_alt_compress_bert_ffn(model, ranks, input_dims, output_dims, False,
                                weight_int, weight_out, weight_count, invasive=False)
    return model

#simple svd from daniel source code
def apply_alt_svd_compress_bert_ffn(model, ranks, *args, **kwargs):
    model = svd_alt_compress_bert_ffn(model, ranks)
    return model

#weighted non-invasive svd from daniel source code
def apply_alt_w_svd_invasive_compress_bert_ffn(model, ranks, 
                                weight_int, weight_out, weight_count):
    model = svd_alt_compress_bert_ffn(model, ranks,
                                      weight_int, weight_out, weight_count, invasive=False)
    return model

#weighted invasive TTM from vika source code
#TODO:not working now, but should?
def apply_ttm_compress_bert_ffn_w_inv(model, ranks, input_dims, output_dims, 
                                weight_int, weight_out, weight_count):
    
    model = ttm_compress_bert_ffn(model, ranks, input_dims, output_dims, False,
                                weight_int, weight_out, weight_count, invasive=True)
    return model

#weighted non-invasive TTM from vika source code
def apply_ttm_compress_bert_ffn_w(model, ranks, input_dims, output_dims, 
                                weight_int, weight_out, weight_count):
    
    model = ttm_compress_bert_ffn(model, ranks, input_dims, output_dims, False,
                                weight_int, weight_out, weight_count, invasive=False)
    return model

def structured_pruning(model):
    model = str_prune(model, 0.6)
    return model

def unstructured_pruning(model):
    model = uns_prune(model, 0.6)
    return model

def simple_svd_model(model, rank = 150, *args, **kwargs):
    model = simple_svd_func(model, rank)
    return model

def apply_ttm_compress_bert_ffn(model, ranks, input_dims, output_dims, dataloader=None):
    model = ttm_compress_bert_ffn(model, ranks, input_dims, output_dims, dataloader=dataloader)
    return model

def weight_svd_model(model, rank = 150, weight_int=None, weight_out=None, weight_count=None):
    model = w_svd_func(model, rank, weight_int, weight_out, weight_count)
    return model

def apply_drone_compress_bert(model, dataloader, rank, device, dir_for_activations):
    model = drone_bert(model, dataloader=dataloader, rank=rank, device=device, dir_for_activations=dir_for_activations)
    return model

def dummy_self(model, *args):
    model = dummy_func(model)
    return model

def weight_svd_T_model(model, rank = 150, weight_int=None, weight_out=None, weight_count=None):
    model = w_svd_func_T(model, rank, weight_int, weight_out, weight_count)
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

def lobert_self_ffn(model, rank):
    model = lobert_student(model.config, 
        model, 
        student_mode = "ffn_svd", rank=rank,
        num_hidden_layers = model.config.num_hidden_layers)
    return model