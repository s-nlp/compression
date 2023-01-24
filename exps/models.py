from .lobert_utils import lobert_student
from .dummy_model import dummy_func
from .head_pruning import random_head_pruning_model
#from .simple_svd import simple_svd_func
#from .unstructured_prune import uns_prune
#from .structured_prune import str_prune
from .ttm.ttm_compress_bert import ttm_compress_bert_ffn
from .fwsvd_compress_bert import compute_nd_replace_dense

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
        ("uns_prune", "unstructured_pruning"),
        ("ttm_ffn", "apply_ttm_compress_bert_ffn"),
        ("fwsvd_ffn", "apply_fwsvd_compress_bert_ffn")
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

def apply_ttm_compress_bert_ffn(model, ranks, input_dims, output_dims):
    model = ttm_compress_bert_ffn(model, ranks, input_dims, output_dims)
    return model

def apply_fwsvd_compress_bert_ffn(model, dataloader, rank, device, use_baseline=False, low_rank_method="row-sum-weighted-svd"):
    print("="*50 + "\n\n\n")
    print("RUNNING FWSVD COMPRESSION")
    model = compute_nd_replace_dense(model, dataloader=dataloader, rank=rank, device=device, 
                                     use_baseline=use_baseline, low_rank_method=low_rank_method, 
                                     compute_full=(low_rank_method != "row-sum-weighted-svd"))
    print("DONE FWSVD COMPRESSION\n\n\n")
    print("="*50)
    return model

def dummy_self(model, *args):
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

def lobert_self_ffn(model, rank):
    model = lobert_student(model.config, 
        model, 
        student_mode = "ffn_svd", rank=rank,
        num_hidden_layers = model.config.num_hidden_layers)
    return model