import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class SVD_dec(nn.Module):
    def __init__(self,input_size, output_size, r):
        super(SVD_dec, self).__init__()
        self.lin0 = nn.Linear(in_features=input_size, out_features=r, bias=False)
        self.lin1 = nn.Linear(in_features=r, out_features=r, bias=False)
        self.lin2 = nn.Linear(in_features=r, out_features=output_size, bias=True)

    def forward(self, x):
        # print(f'x_shape{x.size()}, self.lin0:{self.lin0.weight}')
        output = self.lin0(x)
        # print(f'output_shape{output.size()}, self.lin0:{self.lin1}')
        output = self.lin1(output)
        output = self.lin2(output)
        return output

def svd_bert(model):
    model_svd = BertModel.from_pretrained('bert-base-uncased')
    sd=model.bert.state_dict()
    model_svd.load_state_dict(sd)
    r = 750
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i in range(12):
        model_svd.encoder.layer[i].intermediate.dense = SVD_dec(768,3072,r)#nn.ModuleList([nn.Linear(in_features=768, out_features=r, bias=False),
                                                              # nn.Linear(in_features=r, out_features=r, bias=False),
                                                              # nn.Linear(in_features=r, out_features=3072, bias=True)])
        u, s, v = torch.linalg.svd(model.bert.encoder.layer[i].intermediate.dense.weight)
        bias = model.bert.encoder.layer[i].intermediate.dense.bias
        U = u[:,:r]
        S = s[:r]
        V = v[:r,:]
        model_svd.encoder.layer[i].intermediate.dense.lin0.weight = torch.nn.Parameter(torch.Tensor(V).to(device))
        model_svd.encoder.layer[i].intermediate.dense.lin1.weight = torch.nn.Parameter(torch.Tensor(torch.diag(S)).to(device))
        model_svd.encoder.layer[i].intermediate.dense.lin2.weight = torch.nn.Parameter(torch.Tensor(U).to(device))
        model_svd.encoder.layer[i].intermediate.dense.lin2.bias = torch.nn.Parameter(bias.to(device))
        
        model_svd.encoder.layer[i].output.dense = SVD_dec(3072,768,r)#nn.ModuleList([nn.Linear(in_features=3072, out_features=r, bias=False),
                                                              # nn.Linear(in_features=r, out_features=r, bias=False),
                                                              # nn.Linear(in_features=r, out_features=768, bias=True)])
        u, s, v = torch.linalg.svd(model.bert.encoder.layer[i].output.dense.weight)
        bias = model.bert.encoder.layer[i].output.dense.bias
        U = u[:,:r]
        S = s[:r]
        V = v[:r,:]
        model_svd.encoder.layer[i].output.dense.lin0.weight = torch.nn.Parameter(torch.Tensor(V).to(device))
        model_svd.encoder.layer[i].output.dense.lin1.weight = torch.nn.Parameter(torch.Tensor(torch.diag(S)).to(device))
        model_svd.encoder.layer[i].output.dense.lin2.weight = torch.nn.Parameter(torch.Tensor(U).to(device))
        model_svd.encoder.layer[i].output.dense.lin2.bias = torch.nn.Parameter(bias.to(device))
    model.bert = model_svd
    return model
