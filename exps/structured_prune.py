import torch.nn.utils.prune as prune
import torch
import numpy as np

def str_prune(model, perc = 0.6):
    #choose modules
    parameters_to_prune = [
        (module, "weight") for module in filter(lambda m: type(m) == torch.nn.Linear, model.modules())
        ]
    #do perf
    for i,j in parameters_to_prune:
        prune.ln_structured(i, name=j, amount=perc, n=2, dim=1)
    #do it permament
    for (module_, _ ) in parameters_to_prune:
        prune.remove(module_, 'weight')
    return model