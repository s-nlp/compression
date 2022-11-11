import torch
import torch.nn.utils.prune as prune

def bert_prune(model):
    for name, module in model.named_modules() :
        amount = 0.3
        condition = isinstance(module, torch.nn.Linear) and (name.endswith('intermediate.dense') or name.endswith('output.dense') or name.startswith('pooler.'))
        condition = condition and (not name.endswith('attention.output.dense')) and (not name.endswith('pooler.dense'))
        if condition:
            # print(name, module)
            prune.l1_unstructured(module, name='weight', amount=amount)#0.3 nice
            prune.remove(module, 'weight')
    return model
