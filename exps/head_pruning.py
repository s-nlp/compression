import torch
import numpy as np

def random_head_pruning_model(model):
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    head_mask = (torch.randint(0,101,(n_layers, n_heads)) > 50) *1

    #we need at least one alive head at layer
    for i in np.where(np.all(np.isclose(head_mask, 0), axis=1))[0]:
        head_mask[i][0] = 1.0
    heads_to_prune = dict(
        (layer, torch.atleast_1d((1 - head_mask[layer].long()).nonzero().squeeze()).tolist()) for layer in range(len(head_mask))
    )
    model.prune_heads(heads_to_prune)
    return model