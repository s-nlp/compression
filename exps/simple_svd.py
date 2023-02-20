import torch
import torch.nn as nn


class SVD_dec(nn.Module):
    def __init__(self, input_size, output_size, r):
        super(SVD_dec, self).__init__()
        self.lin0 = nn.Linear(in_features=input_size, out_features=r, bias=False)
        # self.lin1 = nn.Linear(in_features=r, out_features=r, bias=False)
        self.lin2 = nn.Linear(in_features=r, out_features=output_size, bias=True)

    def forward(self, x):
        # print(f'x_shape{x.size()}, self.lin0:{self.lin0.weight}')
        output = self.lin0(x)
        # print(f'output_shape{output.size()}, self.lin0:{self.lin1}')
        # output = self.lin1(output)
        output = self.lin2(output)
        return output


def our_svd(model_to, svd_rank=51):
    for i in range(model_to.config.num_hidden_layers):
        bias = model_to.bert.encoder.layer[i].intermediate.dense.bias
        U, S, Vt = torch.linalg.svd(
            model_to.bert.encoder.layer[i].intermediate.dense.weight,
            full_matrices=False,
        )
        # truncate SVD and fuse Sigma matrix
        w1 = torch.mm(torch.diag(torch.sqrt(S[:svd_rank])), Vt[0:svd_rank, :])
        w2 = torch.mm(U[:, 0:svd_rank], torch.diag(torch.sqrt(S[:svd_rank])))

        model_to.bert.encoder.layer[i].intermediate.dense = SVD_dec(
            model_to.config.hidden_size, model_to.config.intermediate_size, svd_rank
        )

        # model_to.bert.encoder.layer[i].intermediate.dense.lin0.weight.data = w1
        # model_to.bert.encoder.layer[i].intermediate.dense.lin2.weight.data = w2
        # model_to.bert.encoder.layer[i].intermediate.dense.lin2.bias = bias
        # -----------
        model_to.bert.encoder.layer[i].intermediate.dense.lin0.weight.data.copy_(w1)
        model_to.bert.encoder.layer[i].intermediate.dense.lin2.weight.data.copy_(w2)
        model_to.bert.encoder.layer[i].intermediate.dense.lin2.bias.data.copy_(bias)

        bias = model_to.bert.encoder.layer[i].output.dense.bias
        U, S, Vt = torch.linalg.svd(
            model_to.bert.encoder.layer[i].output.dense.weight, full_matrices=False
        )
        # truncate SVD and fuse Sigma matrix
        w1 = torch.mm(torch.diag(torch.sqrt(S[:svd_rank])), Vt[0:svd_rank, :])
        w2 = torch.mm(U[:, 0:svd_rank], torch.diag(torch.sqrt(S[:svd_rank])))

        model_to.bert.encoder.layer[i].output.dense = SVD_dec(
            model_to.config.intermediate_size, model_to.config.hidden_size, svd_rank
        )

        # model_to.bert.encoder.layer[i].output.dense.lin0.weight.data = w1
        # model_to.bert.encoder.layer[i].output.dense.lin2.weight.data = w2
        # model_to.bert.encoder.layer[i].output.dense.lin2.bias = bias
        # -----------
        model_to.bert.encoder.layer[i].output.dense.lin0.weight.data.copy_(w1)
        model_to.bert.encoder.layer[i].output.dense.lin2.weight.data.copy_(w2)
        model_to.bert.encoder.layer[i].output.dense.lin2.bias.data.copy_(bias)
    return model_to


def w_svd_func(
    model_to, svd_rank=51, weight_int=None, weight_out=None, weight_count=None
):
    for i in range(model_to.config.num_hidden_layers):
        if weight_int is not None:
            weight_int_ = weight_int[i].sum(1)
            # weight_int_ /= weight_int_.max()
            # weight_int_ =(weight_int[i]**2).sum(1)
            # print(weight_int_)
            I = torch.diag(torch.sqrt(weight_int_ / weight_count)).to(model_to.device)
        else:
            I = torch.eye(model_to.config.intermediate_size).to(model_to.device)
        # I_slash = torch.inverse(I)
        bias = model_to.bert.encoder.layer[i].intermediate.dense.bias
        U, S, Vt = torch.linalg.svd(
            I @ model_to.bert.encoder.layer[i].intermediate.dense.weight,
            full_matrices=False,
        )
        # truncate SVD and fuse Sigma matrix
        w1 = torch.mm(torch.diag(torch.sqrt(S[0:svd_rank])), Vt[0:svd_rank, :])
        w2 = torch.mm(
            torch.linalg.lstsq(I, U[:, 0:svd_rank]).solution,
            torch.diag(torch.sqrt(S[0:svd_rank])),
        )

        # w2 = torch.mm(I_slash @ U[:, 0:svd_rank], torch.diag(torch.sqrt(S[0:svd_rank])))

        model_to.bert.encoder.layer[i].intermediate.dense = SVD_dec(
            model_to.config.hidden_size, model_to.config.intermediate_size, svd_rank
        )

        model_to.bert.encoder.layer[i].intermediate.dense.lin0.weight.data.copy_(w1)
        model_to.bert.encoder.layer[i].intermediate.dense.lin2.weight.data.copy_(w2)
        model_to.bert.encoder.layer[i].intermediate.dense.lin2.bias.data.copy_(bias)
    for i in range(model_to.config.num_hidden_layers):
        if weight_out is not None:
            weight_out_ = weight_out[i].sum(1)
            # weight_out_ /= weight_out_.max()
            # weight_out_ = (weight_out[i]**2).sum(1)
            I = torch.diag(torch.sqrt(weight_out_ / weight_count)).to(model_to.device)
        else:
            I = torch.eye(model_to.config.hidden_size).to(model_to.device)
        # I_slash = torch.inverse(I)
        bias = model_to.bert.encoder.layer[i].output.dense.bias

        U, S, Vt = torch.linalg.svd(
            I @ model_to.bert.encoder.layer[i].output.dense.weight, full_matrices=False
        )
        w1 = torch.mm(torch.diag(torch.sqrt(S[0:svd_rank])), Vt[0:svd_rank, :])
        w2 = torch.mm(
            torch.linalg.lstsq(I, U[:, 0:svd_rank]).solution,
            torch.diag(torch.sqrt(S[0:svd_rank])),
        )

        model_to.bert.encoder.layer[i].output.dense = SVD_dec(
            model_to.config.intermediate_size, model_to.config.hidden_size, svd_rank
        )

        model_to.bert.encoder.layer[i].output.dense.lin0.weight.data.copy_(w1)
        model_to.bert.encoder.layer[i].output.dense.lin2.weight.data.copy_(w2)
        model_to.bert.encoder.layer[i].output.dense.lin2.bias.data.copy_(bias)
    return model_to


def w_svd_func_inv(
    model_to, svd_rank=51, weight_int=None, weight_out=None, weight_count=None
):
    for i in range(model_to.config.num_hidden_layers):
        if weight_int is not None:
            weight_int_ = weight_int[i].sum(0)
            # weight_int_ = 1 - torch.nn.functional.normalize(weight_int[i].sum(0), dim=0)
            # weight_int_ = torch.softmax(weight_int[i].sum(0), dim=0) **2

            I = torch.diag(torch.sqrt(weight_int_ / weight_count)).to(model_to.device)
        else:
            I = torch.eye(model_to.config.hidden_size).to(model_to.device)
        # I_slash = torch.inverse(I)
        bias = model_to.bert.encoder.layer[i].intermediate.dense.bias
        U, S, Vt = torch.linalg.svd(
            (I @ model_to.bert.encoder.layer[i].intermediate.dense.weight.T).T,
            full_matrices=False,
        )
        # truncate SVD and fuse Sigma matrix
        # w1 = (I_slash @ torch.mm(torch.diag(torch.sqrt(S[0:svd_rank])),Vt[0:svd_rank, :]).T).T
        w1 = torch.linalg.lstsq(
            I, torch.mm(torch.diag(torch.sqrt(S[0:svd_rank])), Vt[0:svd_rank, :]).T
        ).solution.T
        w2 = torch.mm(U[:, 0:svd_rank], torch.diag(torch.sqrt(S[0:svd_rank])))

        model_to.bert.encoder.layer[i].intermediate.dense = SVD_dec(
            model_to.config.hidden_size, model_to.config.intermediate_size, svd_rank
        )

        model_to.bert.encoder.layer[i].intermediate.dense.lin0.weight.data.copy_(w1)
        model_to.bert.encoder.layer[i].intermediate.dense.lin2.weight.data.copy_(w2)
        model_to.bert.encoder.layer[i].intermediate.dense.lin2.bias.data.copy_(bias)
    for i in range(model_to.config.num_hidden_layers):
        if weight_out is not None:
            weight_out_ = weight_out[i].sum(0)
            # weight_out_ = 1 - torch.nn.functional.normalize(weight_out[i].sum(0), dim=0)

            I = torch.diag(torch.sqrt(weight_out_ / weight_count)).to(model_to.device)
        else:
            I = torch.eye(model_to.config.intermediate_size).to(model_to.device)
        # I_slash = torch.inverse(I)
        bias = model_to.bert.encoder.layer[i].output.dense.bias

        U, S, Vt = torch.linalg.svd(
            (I @ model_to.bert.encoder.layer[i].output.dense.weight.T).T,
            full_matrices=False,
        )
        w1 = torch.linalg.lstsq(
            I, torch.mm(torch.diag(torch.sqrt(S[0:svd_rank])), Vt[0:svd_rank, :]).T
        ).solution.T
        # w1 = (I_slash @ torch.mm(torch.diag(torch.sqrt(S[0:svd_rank])),Vt[0:svd_rank, :]).T).T
        w2 = torch.mm(U[:, 0:svd_rank], torch.diag(torch.sqrt(S[0:svd_rank])))

        model_to.bert.encoder.layer[i].output.dense = SVD_dec(
            model_to.config.intermediate_size, model_to.config.hidden_size, svd_rank
        )

        model_to.bert.encoder.layer[i].output.dense.lin0.weight.data.copy_(w1)
        model_to.bert.encoder.layer[i].output.dense.lin2.weight.data.copy_(w2)
        model_to.bert.encoder.layer[i].output.dense.lin2.bias.data.copy_(bias)
    return model_to


def simple_svd_func(model, svd_rank=51):
    return our_svd(model, svd_rank)
