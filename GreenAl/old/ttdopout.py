import torch
from torch import nn
from linear import TTMLinear
import copy

import tensorly as tl
tl.set_backend('pytorch')

class TTDropout(nn.Module):
    def __init__(self, old_layer, proba, min_dim, rank):
        super().__init__()
        self.proba = proba
        self.min_dim = min_dim
        self.old_layer = copy.deepcopy(old_layer)
        self.layer = copy.deepcopy(old_layer)
        self.rank = rank
        
    #def create_zero_mask(self):
    #def forward(self, inpt):
               
    def apply_tensor_dropout1(self, tt_tensor, training=True):
        if (not self.proba) or ((not training)):
            return tt_tensor

        device = tt_tensor.ttm.tt.cores[0].device

        sampled_indices = []
        for i, rank in enumerate(tt_tensor.ttm.tt.ranks):
            if rank > self.min_dim:
                idx = tl.arange(rank, device=device, dtype=torch.int64)
                idx = idx[torch.bernoulli(torch.ones(rank, device=device)*(1 - self.proba),
                                          out=torch.empty(rank, device=device, dtype=torch.bool))]
                if len(idx) == 0:
                    idx = torch.randint(0, rank, size=(min_values, ), device=device, dtype=torch.int64)
            else:
                idx = tl.arange(rank, device=device, dtype=torch.int64).tolist()

            sampled_indices.append(idx)

        sampled_factors = []
        if training:
            scaling = 1/(1 - self.proba)
        else:
            scaling = 1
            
        for i, f in enumerate(tt_tensor.ttm.tt.cores):
            if i == 0:
                ax = len(tt_tensor.ttm.tt.cores[0].shape) - 1
                d = torch.index_select(f, ax, sampled_indices[i])
                sampled_factors.append(torch.clone(torch.index_select(f, ax, sampled_indices[i])*scaling))
            elif i == (len(tt_tensor.ttm.tt.cores) - 1):
                ax = 0
                d = torch.index_select(f, ax, sampled_indices[i - 1])
                sampled_factors.append(torch.clone(torch.index_select(f, ax, sampled_indices[i - 1])*scaling))
            else:
                ax_0 = 0
                ax_end = len(tt_tensor.ttm.tt.cores[0].shape) - 1
                new_tensor = torch.index_select(f, ax_0, sampled_indices[i - 1])
                new_tensor = torch.index_select(new_tensor, ax_end, sampled_indices[i])*scaling
                sampled_factors.append(torch.clone(new_tensor))

        return nn.ParameterList(sampled_factors)
    
    def forward(self, inpt):
        if self.training:
            #for i, f in enumerate(self.layer.ttm.tt.cores):  
                #print ("self layer shapes before", f.shape)
            #print ("\n\n\n")
            self.layer.ttm.tt.cores = self.apply_tensor_dropout1(self.old_layer, training=True)
            
            #for i, f in enumerate(self.layer.ttm.tt.cores):  
                #print ("self layer shapes after", f.shape)
            #print ("\n\n\n")
            return self.layer(inpt)
            
        else:
            #print ("else")
            return self.layer(inpt)
        