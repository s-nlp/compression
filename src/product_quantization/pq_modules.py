'''

Implementation of Standard PyTorch Modules via Product Quantization (PQ). 
Currently supports only nn.Embedding and nn.Linear.

'''


import torch
import torch.nn as nn

from pqkmeans.encoder import PQEncoder


def quantinize_matrix(matrix, dim, num_clusters):

    encoder = PQEncoder(num_subdim=dim, Ks=num_clusters)
    encoder.fit(matrix)

    indexes = encoder.transform(matrix)
    vectors = encoder.codewords

    return indexes, vectors


class PQEmbedding(nn.Module):
    def __init__(self, module, dim, num_clusters=256):
        super().__init__()

        assert isinstance(module, nn.Embedding), 'An embedding layer is expected'

        weight = module.weight.data.numpy()
        _, self.original_dim = weight.shape

        indexes, vectors = quantinize_matrix(weight, dim, num_clusters)
        indexes = torch.from_numpy(indexes)
        vectors = torch.from_numpy(vectors).to(module.weight.dtype)

        self.register_buffer('indexes', indexes)
        self.register_buffer('vectors', vectors)

        dims = torch.arange(dim)
        self.register_buffer('dims', dims)

    def forward(self, idx):
        embeddings = self.vectors[self.dims, self.indexes[idx.flatten()].to(torch.long)]
        embeddings = embeddings.reshape(*idx.size(), self.original_dim)
        return embeddings
    

class PQLinear(nn.Module):
    def __init__(self, module, dim, num_clusters=256):
        super().__init__()

        assert isinstance(module, nn.Linear), 'A linear layer is expected'

        weight = module.weight.data.numpy().T
        self.in_dim, self.out_dim = weight.shape

        indexes, vectors = quantinize_matrix(weight, dim, num_clusters)
        indexes = torch.from_numpy(indexes)
        vectors = torch.from_numpy(vectors).to(module.weight.dtype)

        self.register_buffer('indexes', indexes)
        self.register_buffer('vectors', vectors)

        dims = torch.arange(dim)
        in_dims = torch.arange(self.in_dim)
        
        self.register_buffer('dims', dims)
        self.register_buffer('in_dims', in_dims)

        self.bias = module.bias

    def forward(self, x):
        weight = self.vectors[self.dims, self.indexes[self.in_dims].to(torch.long)].reshape(self.in_dim, self.out_dim)
        x = torch.matmul(x, weight) + self.bias
        return x
