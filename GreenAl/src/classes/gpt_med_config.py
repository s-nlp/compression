from transformers.configuration_utils import PretrainedConfig


class GPT2MedConfig(PretrainedConfig):
    
    
    """
    Configuration class for GPT2Medium model, similar to GPT2Config
    
    Attributes
    ----------
    
    vocab_size: int
        Vocabulary size of the GPT-2 model.
    n_positions: int
        The maximum sequence length that this model might ever be used with. 
    n_embd: int
        Dimensionality of the embeddings and hidden states.
    n_layer: int
        Number of hidden layers in the Transformer encoder.
    n_head: int
        Number of attention heads for each attention layer in the Transformer encoder.
    n_inner: int
        Dimensionality of the inner feed-forward layers. None will set it to 4 times n_embd
    activation_function: str
        Activation function
    resid_pdrop: float
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    embd_pdrop: int
        The dropout ratio for the embeddings.
    attn_pdrop: float
        The dropout ratio for the attention.
    layer_norm_epsilon: float
        The epsilon to use in the layer normalization layers.
    initializer_range: float)
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    summary_type: string
        Argument used when doing sequence summary, used in the DoubleHeadsModel GPT2.
    summary_use_proj: bool
        Argument used when doing sequence summary, used in the models GPT2DoubleHeadsModel and TFGPT2DoubleHeadsModel.
    summary_activation: str
        Argument used when doing sequence summary. Used in for the multiple choice head in GPT2DoubleHeadsModel.
    summary_proj_to_labels: bool 
        Argument used when doing sequence summary, used in the models GPT2DoubleHeadsModel and TFGPT2DoubleHeadsModel.
   summary_first_dropout: float  
       Argument used when doing sequence summary, used in the models GPT2DoubleHeadsModel and TFGPT2DoubleHeadsModel.
   scale_attn_weights: bool
       Scale attention weights by dividing by sqrt(hidden_size)
   scale_attn_by_inverse_layer_idx: bool
       Whether to additionally scale attention weights by 1 / layer_idx + 1.
   reorder_and_upcast_attn: bool
       Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention dot-product/softmax to float() when training with mixed precision.

    
    See Also:
    
    ----------
    -transformers.GPT2Config
    
    """

    model_type = "gpt2-medium"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(     
        self,
        vocab_size=50257,
        n_positions=1024,
        n_embd=1024,
        n_ctx=1024, 
        n_layer=24,
        n_head=16,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_ctx = n_ctx
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)    