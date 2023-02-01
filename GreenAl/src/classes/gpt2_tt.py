from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel
from src.ttm_linear.ttm_linear import FactorizationTTMLinear
#from src.layers2.linear import TTMLinear

class GPT2_TT_Model(GPT2LMHeadModel):
    
    """
    The GPT2 Model transformer with a compressed fully-connected and projection layers. Model has a language modeling head on top (linear layer with weights tied to the input embeddings).
    Attributes
    ----------
    configuration : str
        Configuration file that stores parameter for GPT2 model.
    rank : int 
        A rank o af the TTMLinear layer - TT representation of FC layers. 
        
    See Also
    --------
    transformers.GPT2LMHeadModel
    src.layers2.linear.TTMLinear
    
    """
    
    def __init__(self, configuration, rank):
        """
        
        Initializes a GPT2_TT_Model object.
        During the initialization GPT2LMHeadModel is created using parameters from configuration. 
        Every fully-connected and projection layers from the created GPT2LMHeadModel is represented as a TT layer TTMLinear with rank rank.
        If rank = 0, the layers remain unchanged.
        
        Parameters
        ----------
        
        configuration : str
            Configuration file that stores parameter for GPT2 model.
        rank : int 
            A rank of the TTMLinear layer - TT representation of FC layers. 
        
        """       
        super().__init__(configuration)
        for i in range(len(self.transformer.h)):
            # fc part
            old_layer = self.transformer.h[i].mlp.c_fc
            (in_, out_) = old_layer.weight.shape
            layer = FactorizationTTMLinear(in_, out_, rank=rank, max_core_dim_product = rank)
            self.transformer.h[i].mlp.c_fc = layer

            # projection
            old_layer = self.transformer.h[i].mlp.c_proj
            (in_, out_) = old_layer.weight.shape
            layer = FactorizationTTMLinear(in_, out_, rank=rank, max_core_dim_product = rank)
            #drop_layer = TTDropout(layer, proba = 0.8, min_dim = 2, rank=128)
            #layer = drop_layer
            self.transformer.h[i].mlp.c_proj = layer
            
        