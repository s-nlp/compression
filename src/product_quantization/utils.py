'''

Auxiliary tools for benchmarks. May be omitted later and transfer to 'pq_bert.ipynb' notebook.

'''


import os
import numpy as np

import torch


def model_size(model):
    torch.save(model.state_dict(), 'tmp.p')
    size_mb = os.path.getsize('tmp.p') / (1024 ** 2)
    os.remove('tmp.p')
    return size_mb


def cosine_similarity(x, y):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = y / np.linalg.norm(y, axis=1, keepdims=True)
    similarity = (x * y).sum(axis=1)
    return similarity
