import torch
import torch.nn as nn
import tntorch as tn

from time import time

from TTLinear import TTLinear
from ttm_compress_bert import Checkpointed
from forward_backward import forward, full_matrix_backward, forward_backward_module

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def fake_train(model, d, batch_size, seq_length, device, n_steps=10, seed=0):
    torch.manual_seed(seed)

    start = time()

    for step in range(n_steps):
        x = torch.rand(batch_size, seq_length, d, requires_grad=True).to(device)
        pred = model(x)
        loss = pred.norm()
        loss.backward()
    
    torch.cuda.synchronize()
    end = time()

    samples_per_second = n_steps * batch_size / (end - start) 

    return model, end - start

def run_benchmark(model, d, batch_size, seq_length, device):
    print("N parameters:", count_parameters(model) // 1000, "k")

    print("Before:", torch.cuda.max_memory_allocated() // 1024**2, "Mb")
    model, train_time = fake_train(model, d, batch_size, seq_length, device)
    print("After:", torch.cuda.max_memory_allocated() // 1024**2, "Mb")
    print(f"Train time: {train_time:.3f} s\n")

    del model
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.empty_cache()


def make_linear_block(d):
    return nn.Sequential(
        nn.Linear(d, 4 * d),
        nn.ReLU(),
        nn.Linear(4 * d, d),
        nn.ReLU(),
    )


def make_ttlinear_block(d, **tt_args):
    input_dims = tt_args.pop("input_dims")
    output_dims = tt_args.pop("output_dims")
    return nn.Sequential(
        TTLinear(d, 4 * d, input_dims=input_dims, output_dims=output_dims, **tt_args),
        nn.ReLU(),
        TTLinear(4 * d, d, input_dims=output_dims, output_dims=input_dims, **tt_args),
        nn.ReLU(),
    )


def make_checkpointed_ttlinear_block(d, **tt_args):
    input_dims = tt_args.pop("input_dims")
    output_dims = tt_args.pop("output_dims")
    return nn.Sequential(
        Checkpointed(TTLinear(d, 4 * d, input_dims=input_dims, output_dims=output_dims, **tt_args)),
        nn.ReLU(),
        Checkpointed(TTLinear(4 * d, d, input_dims=output_dims, output_dims=input_dims, **tt_args)),
        nn.ReLU(),
    )


def main():
    d = 768
    batch_size = 128
    seq_length = 2
    n_blocks = 12
    ranks = [100, 100, 100]
    input_dims = [4, 6, 8, 4]
    output_dims = [8, 8, 6, 8]

    device = "cuda:0"
    assert torch.cuda.is_available()

    print(f"Testing with batch size {batch_size}, seq_length = {seq_length}, d = {d}, n_blocks = {n_blocks}, ranks = {ranks}")

    print("Usual Linear layer")
    model = nn.Sequential(
        *[make_linear_block(d) for _ in range(n_blocks)]
    ).to(device)

    run_benchmark(model, d, batch_size, seq_length, device)

    print("TTm Linear, checkpoint every layer")
    tt_args = {
        "ranks": ranks,
        "input_dims": input_dims,
        "output_dims": output_dims,
    }
    model = nn.Sequential(
        *[make_checkpointed_ttlinear_block(d, **tt_args) for _ in range(n_blocks)]
    ).to(device)

    run_benchmark(model, d, batch_size, seq_length, device)

    print("TTm Linear")
    tt_args = {
        "ranks": ranks,
        "input_dims": input_dims,
        "output_dims": output_dims,
    }
    model = nn.Sequential(
        *[make_ttlinear_block(d, **tt_args) for _ in range(n_blocks)]
    ).to(device)

    run_benchmark(model, d, batch_size, seq_length, device)

    # TTm Linear
    print("TTm Linear, backward by hands")

    tt_args = {
        "ranks": ranks,
        "input_dims": input_dims,
        "output_dims": output_dims,
        "forward_fn": forward_backward_module(forward, full_matrix_backward(forward))
    }
    model = nn.Sequential(
        *[make_ttlinear_block(d, **tt_args) for _ in range(n_blocks)]
    ).to(device)

    run_benchmark(model, d, batch_size, seq_length, device)

    # TTm Linear
    print("TTm Linear, backward by hands + checkpoints")

    tt_args = {
        "ranks": ranks,
        "input_dims": input_dims,
        "output_dims": output_dims,
        "forward_fn": forward_backward_module(forward, full_matrix_backward(forward))
    }
    model = nn.Sequential(
        *[make_checkpointed_ttlinear_block(d, **tt_args) for _ in range(n_blocks)]
    ).to(device)

    run_benchmark(model, d, batch_size, seq_length, device)
    
main()
