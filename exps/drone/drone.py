import os
import torch
from collections import defaultdict

from rrr import reduced_rank_regression

@torch.inference_mode()
def compute_and_save_activations(model, dataloader, path_to_save: str, subsample: float = 0.01):
    total_len = len(dataloader)
    
    if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
        encoder = model.bert.encoder
    elif hasattr(model, "encoder"):
        encoder = model.encoder
    else:
        raise ValueError("Expected model to have attribute 'encoder' or 'bert.encoder'.")

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= subsample * total_len:
            break

        assert isinstance(batch, dict) and "attention_mask" in batch

        attention_mask = batch["attention_mask"]
        hidden_states = model(**batch, output_hidden_states=True).hidden_states

        for layer_idx in range(len(hidden_states) - 1):
            x = hidden_states[layer_idx]
            layer = encoder.layer[layer_idx]
            
            attention = layer.attention.self
            q_target = attention.query_proj(x)
            k_target = attention.key_proj(x)
            v_target = attention.value_proj(x)
            torch.save(q_target.half(), path_to_save + f"/batch_{batch_idx}_layer_{layer_idx}_q.pth")
            torch.save(k_target.half(), path_to_save + f"/batch_{batch_idx}_layer_{layer_idx}_k.pth")
            torch.save(v_target.half(), path_to_save + f"/batch_{batch_idx}_layer_{layer_idx}_v.pth")
            x = attention(x, attention_mask, output_attentions=False)

            attn_out_target = layer.attention.output.dense(x)
            torch.save(attn_out_target.half(), path_to_save + f"/batch_{batch_idx}_layer_{layer_idx}_attn-out.pth")
            x = layer.attention.output(x)

            intermediate_target = layer.intermediate.dense(x)
            torch.save(intermediate_target.half(), path_to_save + f"/batch_{batch_idx}_layer_{layer_idx}_mlp-intermediate.pth")
            x = layer.intermediate(x)

            output_target = layer.output.dense(x)
            torch.save(output_target.half(), path_to_save + f"/batch_{batch_idx}_layer_{layer_idx}_mlp-output.pth")            

            #torch.save(hidden.half(), path_to_save + f"/batch_{batch_idx}_layer_{layer_idx}_output_after_dropout_and_layernorm.pth")

            


def concat_activations_by_layers(path_to_save: str):

    layer2activations = defaultdict(list)

    for filename in os.listdir(path_to_save):
        if not filename.endswith(".pth"):
            continue
        activations = torch.load(filename)
        os.remove(filename)
        # asserting filename has structure "batch_{batch_idx}_layer_{layer_idx}_..._{part}.pth"
        splitted = filename.split("_")
        batch_idx = int(splitted[1])
        layer_idx = int(splitted[3])
        part = splitted[-1][:-4]

        print(part)

        layer2activations[(layer_idx, part)].append(activations)
    
    for (layer_idx, part), list_of_activations in layer2activations.items():
        activations = torch.cat(list_of_activations)
        print(f"Total activations shape, layer {layer_idx}, part {part}", activations.shape)
        torch.save(activations, path_to_save + f"layer_{layer_idx}_{part}.pth")