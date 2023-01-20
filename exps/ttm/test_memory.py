import torch
import torch.nn as nn
import tntorch as tn

from TTLinear import TTLinear
from ttm_compress_bert import Checkpointed

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def fake_train(model, d, batch_size, device, n_steps=5, seed=0):
    torch.manual_seed(seed)

    for step in range(n_steps):
        x = torch.rand(batch_size, d, requires_grad=True).to(device)
        pred = model(x)
        loss = pred.norm()
        loss.backward()

    return model


def print_results(model, d, batch_size, device):
    print("N parameters:", count_parameters(model) // 1000, "k")

    print("Before:", torch.cuda.max_memory_allocated() // 1024**2, "Mb")
    model = fake_train(model, d, batch_size, device)
    print("After:", torch.cuda.max_memory_allocated() // 1024**2, "Mb\n")


def main():
    d = 768
    batch_size = 256
    ranks = [100, 100, 100]
    input_dims = [4, 6, 8, 4]
    output_dims = [8, 8, 6, 8]

    device = "cuda:0"
    assert torch.cuda.is_available()

    print(f"Testing with batch size {batch_size}, d = {d}, ranks = {ranks}")
    # Linear
    print("Usual Linear layer")
    model = nn.Sequential(
        nn.Linear(d, 4 * d),
        nn.ReLU(),
        nn.Linear(4 * d, d),
        nn.ReLU(),
        nn.Linear(d, 4 * d),
        nn.ReLU(),
        nn.Linear(4 * d, d),
    ).to(device)

    print_results(model, d, batch_size, device)

    del model
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.empty_cache()

    # TTm Linear
    print("TTm Linear")

    model = nn.Sequential(
        TTLinear(d, 4 * d, ranks, input_dims, output_dims),
        nn.ReLU(),
        TTLinear(4 * d, d, ranks, output_dims, input_dims),
        nn.ReLU(),
        TTLinear(d, 4 * d, ranks, input_dims, output_dims),
        nn.ReLU(),
        TTLinear(4 * d, d, ranks, output_dims, input_dims),
    ).to(device)

    print_results(model, d, batch_size, device)

    del model
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.empty_cache()

    # TTm Linear, checkpoints
    print("TTm Linear, checkpoint every layer")

    model = nn.Sequential(
        Checkpointed(TTLinear(d, 4 * d, ranks, input_dims, output_dims)),
        nn.ReLU(),
        Checkpointed(TTLinear(4 * d, d, ranks, output_dims, input_dims)),
        nn.ReLU(),
        Checkpointed(TTLinear(d, 4 * d, ranks, input_dims, output_dims)),
        nn.ReLU(),
        Checkpointed(TTLinear(4 * d, d, ranks, output_dims, input_dims)),
    ).to(device)

    print_results(model, d, batch_size, device)

    del model
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.empty_cache()

    # TTm Linear, checkpoints
    print("TTm Linear, checkpoint all")

    model = Checkpointed(
        TTLinear(d, 4 * d, ranks, input_dims, output_dims),
        nn.ReLU(),
        TTLinear(4 * d, d, ranks, output_dims, input_dims),
        nn.ReLU(),
        TTLinear(d, 4 * d, ranks, input_dims, output_dims),
        nn.ReLU(),
        TTLinear(4 * d, d, ranks, output_dims, input_dims),
    ).to(device)

    print_results(model, d, batch_size, device)

    del model
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.empty_cache()

    # TTm Linear, checkpoints
    print("TTm Linear, 2 checkpoints")

    model = nn.Sequential(
        Checkpointed(
            TTLinear(d, 4 * d, ranks, input_dims, output_dims),
            nn.ReLU(),
            TTLinear(4 * d, d, ranks, output_dims, input_dims),
        ),
        nn.ReLU(),
        Checkpointed(
            TTLinear(d, 4 * d, ranks, input_dims, output_dims),
            nn.ReLU(),
            TTLinear(4 * d, d, ranks, output_dims, input_dims),
        ),
    ).to(device)

    print_results(model, d, batch_size, device)

    del model
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.empty_cache()

    # TTm Linear, checkpoints
    print("TTm Linear, checkpoint every layer, opt_einsum")

    model = nn.Sequential(
        Checkpointed(TTLinear(d, 4 * d, ranks, input_dims, output_dims, use_opt_einsum=True)),
        nn.ReLU(),
        Checkpointed(TTLinear(4 * d, d, ranks, output_dims, input_dims, use_opt_einsum=True)),
        nn.ReLU(),
        Checkpointed(TTLinear(d, 4 * d, ranks, input_dims, output_dims, use_opt_einsum=True)),
        nn.ReLU(),
        Checkpointed(TTLinear(4 * d, d, ranks, output_dims, input_dims, use_opt_einsum=True)),
    ).to(device)

    print_results(model, d, batch_size, device)

    del model
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.empty_cache()


main()
