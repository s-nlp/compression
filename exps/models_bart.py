from .bart_simple_svd import w_svd_func_T
from .ttm.ttm_compress_bert import ttm_compress_bart_ffn_W


from collections import OrderedDict
MODEL_NAMES = OrderedDict(
    [
        # Base model mapping
        ("svd_ffn_bart", "apply_svd_compress_bart"),
        ("svd_ffn_w_bart", "apply_svd_w_compress_bart"),
        ("ttm_ffn_bart", "apply_ttm_compress_bart"),
        ("ttm_ffn_w_bart", "apply_ttm_w_compress_bart"),
    ])

#dummy model
def dummy_self(model, *args, **kwargs):
    model = dummy_func(model)
    return model

def apply_svd_compress_bart(model, ranks):
    model = w_svd_func_T(model, ranks)
    return model

def apply_svd_w_compress_bart(model, ranks, full_dict):
    model = w_svd_func_T(model, ranks,
        weight_int_e = full_dict['weight_fc1_en'], 
        weight_out_e = full_dict['weight_fc2_en'], weight_int_d = full_dict['weight_fc1_de'], 
        weight_out_d = full_dict['weight_fc2_de'], weight_count = full_dict['weight_count']
                )
    return model

def apply_ttm_compress_bart(model, ranks, input_dims, output_dims):
    model = ttm_compress_bart_ffn_W(model, ranks, 
                                        input_dims, output_dims)
    return model

def apply_ttm_w_compress_bart(model, ranks, input_dims, output_dims, full_dict):
    model = ttm_compress_bart_ffn_W(model, ranks, input_dims, output_dims,
        weight_int_e = full_dict['weight_fc1_en'], 
        weight_out_e = full_dict['weight_fc2_en'], weight_int_d = full_dict['weight_fc1_de'], 
        weight_out_d = full_dict['weight_fc2_de'], weight_count = full_dict['weight_count']
                )
    return model