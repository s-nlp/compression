import os
import torch
from collections import defaultdict
from tqdm import tqdm

from .rrr import reduced_rank_regression

QUERIES = "queries"
KEYS = "keys"
VALUES = "values"
ATTN_OUT = "attn-out"
MLP_INTERMEDIATE = "mlp-intermediate"
MLP_OUTPUT = "mlp-output"
EMBEDS = "embeds"
ATTN_MASK = "attn-mask"


@torch.inference_mode()
def compute_and_save_bert_activations(model, dataloader, output_dir: str, device: str, subsample: float = 0.01):
    total_len = len(dataloader)
    
    if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
        encoder = model.bert.encoder
    elif hasattr(model, "deberta") and hasattr(model.deberta, "encoder"):
        encoder = model.deberta.encoder
    elif hasattr(model, "encoder"):
        encoder = model.encoder
    else:
        raise ValueError("Expected model to have attribute 'encoder' or 'bert.encoder'.")

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="computing activations", total=total_len * subsample)):
        if batch_idx >= subsample * total_len:
            break

        batch = {key: value.to(device) for key, value in batch.items()}

        assert isinstance(batch, dict) and "attention_mask" in batch

        hidden_states = model(**batch, output_hidden_states=True).hidden_states

        attention_mask = encoder.get_attention_mask(batch["attention_mask"])
        rel_embeddings = encoder.get_rel_embedding()

        torch.save(hidden_states[0], output_dir + f"/batch_{batch_idx}_layer_{0}_{EMBEDS}.pth")
        torch.save(attention_mask.bool(), output_dir + f"/batch_{batch_idx}_layer_{0}_{ATTN_MASK}.pth")

        for layer_idx in range(len(hidden_states) - 1):
            x = hidden_states[layer_idx]
            layer = encoder.layer[layer_idx]
            
            # Recompute Attention activations
            attention = layer.attention.self
            q_target = attention.query_proj(x)
            k_target = attention.key_proj(x)
            v_target = attention.value_proj(x)
            
            x = attention(x, attention_mask, rel_embeddings=rel_embeddings, output_attentions=False)

            attn_out_target = layer.attention.output.dense(x)
            x = layer.attention.output(x, hidden_states[layer_idx])
            attn_output = x

            # Recompute first linear layer in MLP activations
            intermediate_target = layer.intermediate.dense(x)
            x = layer.intermediate(x)

            # Recompute second linear layer in MLP activations
            output_target = layer.output.dense(x)

            torch.save(q_target, output_dir + f"/batch_{batch_idx}_layer_{layer_idx}_{QUERIES}.pth")
            torch.save(k_target, output_dir + f"/batch_{batch_idx}_layer_{layer_idx}_{KEYS}.pth")
            torch.save(v_target, output_dir + f"/batch_{batch_idx}_layer_{layer_idx}_{VALUES}.pth")
            torch.save(attn_out_target, output_dir + f"/batch_{batch_idx}_layer_{layer_idx}_{ATTN_OUT}.pth")
            torch.save(intermediate_target, output_dir + f"/batch_{batch_idx}_layer_{layer_idx}_{MLP_INTERMEDIATE}.pth")
            torch.save(output_target, output_dir + f"/batch_{batch_idx}_layer_{layer_idx}_{MLP_OUTPUT}.pth")

            # Sanity check
            x = layer.output(x, attn_output)
            assert torch.allclose(x, hidden_states[layer_idx + 1])


def concat_activations_by_layers(dir_with_activations: str):

    layer_and_part_2_filenames = defaultdict(list)

    for filename in sorted(os.listdir(dir_with_activations)):
        # asserting filename has structure "batch_{batch_idx}_layer_{layer_idx}_..._{part}.pth"
        if not filename.endswith(".pth"):
            continue
        splitted = filename.split("_")
        assert splitted[0] == "batch" and splitted[2] == "layer"

        layer_idx = int(splitted[3])
        part = splitted[-1][:-4]

        layer_and_part_2_filenames[(layer_idx, part)].append(filename)
    
    for (layer_idx, part), list_of_filenames in layer_and_part_2_filenames.items():
        activations = torch.cat([torch.load(dir_with_activations + name, map_location="cpu") for name in list_of_filenames])
        #print(f"Total activations shape, layer {layer_idx}, part {part}", activations.shape)
        torch.save(activations, dir_with_activations + f"/layer_{layer_idx}_{part}.pth")


class LowRankLinear(torch.nn.Module):
    def __init__(self, first_weight, second_weight, bias):
        super().__init__()
        self.first = torch.nn.Linear(first_weight.shape[0], first_weight.shape[1], bias=False)
        self.second = torch.nn.Linear(second_weight.shape[0], second_weight.shape[1])

        self.first.weight.data = first_weight.clone().T
        self.second.weight.data = second_weight.clone().T
        self.second.bias.data = bias.clone()
    
    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x

    def __repr__(self):
        return f"LowRankLinear({self.first.weight.shape[1]}_{self.first.weight.shape[0]}_{self.second.weight.shape[0]})"


@torch.no_grad()
def run_drone_compression_for_bert(model, rank, dir_with_activations: str, device: str, batch_size=128):
    def _load(layer_idx: int, part: str):
        return torch.load(dir_with_activations + f"/layer_{layer_idx}_{part}.pth", map_location=device)

    def _build_compressed_layer(x: torch.Tensor, layer_idx: int, part: str):
        """
        Loads precomputed activations for specific part (queries, mlp_output, ...) for specific layer,
        runs reduced rank regression to compute best low-rank approximation (in terms of reconstructing activations).
        """
        target = _load(layer_idx, part)

        w, u, bias = reduced_rank_regression(x.flatten(0, 1), target.flatten(0, 1), rank, pinverse_device=device)
        #print("x", x.flatten(0, 1).shape, "y", target.flatten(0, 1).shape, "first matrix", w.shape, "second matrix", u.shape, "bias", bias.shape)

        error = target.flatten(0, 1) - bias - x.flatten(0, 1) @ w @ u
        
        print(f"{part:<16} relative error: {torch.linalg.matrix_norm(error).item() / torch.linalg.matrix_norm(target.flatten(0, 1)).item():.5f}")

        return LowRankLinear(w, u, bias)


    if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
        encoder = model.bert.encoder
    elif hasattr(model, "deberta") and hasattr(model.deberta, "encoder"):
        encoder = model.deberta.encoder
    elif hasattr(model, "encoder"):
        encoder = model.encoder
    else:
        raise ValueError("Expected model to have attribute 'encoder' or 'bert.encoder'.")

    rel_embeddings = encoder.get_rel_embedding()

    x = _load(0, EMBEDS)
    attention_mask = _load(0, ATTN_MASK)

    for layer_idx in tqdm(range(len(encoder.layer)), desc="running drone"):
        layer = encoder.layer[layer_idx]

        # q_target = _load(layer_idx, QUERIES)

        # q_rec = torch.cat([
        #     layer.attention.self.query_proj(
        #         x[i:i + batch_size]
        #     ) for i in range(0, x.shape[0], batch_size)
        # ])
        #print(torch.linalg.matrix_norm(q_rec.flatten(0,1) - q_target.flatten(0,1)).item() / torch.linalg.matrix_norm(q_target.flatten(0,1)).item())
        #assert torch.allclose(q_rec, q_target)

# Self-attention query/value/key
        attn_skip_connection = x.clone()
        
        nexx = torch.cat([
            layer.attention.self(
                x[i:i + batch_size], 
                attention_mask[i:i + batch_size], 
                rel_embeddings=rel_embeddings, 
                output_attentions=False
            )
            for i in range(0, x.shape[0], batch_size)]
        )

        layer.attention.self.query_proj = _build_compressed_layer(x, layer_idx, QUERIES)
        layer.attention.self.value_proj = _build_compressed_layer(x, layer_idx, VALUES)
        layer.attention.self.key_proj = _build_compressed_layer(x, layer_idx, KEYS)

        x = nexx

# Self-attention output
        nexx = torch.cat([
            layer.attention.output(x[i:i + batch_size], attn_skip_connection[i:i + batch_size])
            for i in range(0, x.shape[0], batch_size)]
        )

        #layer.attention.output.dense = _build_compressed_layer(x, layer_idx, ATTN_OUT)

        x = nexx

# MLP Intermediate
        mlp_skip_connection = x.clone()

        nexx = torch.cat([
            layer.intermediate(x[i:i + batch_size])
            for i in range(0, x.shape[0], batch_size)]
        )
        
        layer.intermediate.dense = _build_compressed_layer(x, layer_idx, MLP_INTERMEDIATE)

        x = nexx
# MLP Output        

        nexx = torch.cat([
            layer.output(x[i:i + batch_size], mlp_skip_connection[i:i + batch_size])
            for i in range(0, x.shape[0], batch_size)
        ])
        
        #layer.output.dense = _build_compressed_layer(x, layer_idx, MLP_OUTPUT)

        x = nexx

    
    return model


def drone_bert(model, dataloader, rank, device, dir_for_activations):
    if not os.path.exists(dir_for_activations):
        os.mkdir(dir_for_activations)
        compute_and_save_bert_activations(model, dataloader, dir_for_activations, device, subsample=0.1)
        concat_activations_by_layers(dir_for_activations)
    return run_drone_compression_for_bert(model, rank, dir_for_activations, device)