import torch.nn.utils.prune as prune
import torch
import numpy as np

def uns_prune(model, perc = 0.6):
    #choose modules
    parameters_to_prune = [
        (module, "weight") for module in filter(lambda m: type(m) == torch.nn.Linear, model.modules())
        ]
    #do perf
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=perc,
        )
    #do it permament
    for (module_, _ ) in parameters_to_prune:
        prune.remove(module_, 'weight')
    return model