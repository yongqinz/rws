"""
Pilot: Measuring Quantization Dependency Horizon in MoE
Quantize one linear block in one MoE layer, measure error propagation to subsequent layers.
This tests the core hypothesis: does MoE quantization error propagate non-trivially across layers?
"""
import sys
sys.path.insert(0, '/data/zengyq/paper/baselines/MxMoE')

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
from pathlib import Path

MODEL_PATH = "/data/zengyq/models/Qwen1.5-MoE-A2.7B"
DEVICE = "cuda:0"  # Use CUDA_VISIBLE_DEVICES to control which physical GPU
N_LAYERS_TO_CHECK = 4  # Check propagation over 4 layers ahead
N_SAMPLES = 32  # Small pilot sample

def quantize_weight_rtn(weight, bits=4, group_size=128):
    """Simple RTN quantization for pilot testing."""
    orig_shape = weight.shape
    if group_size > 0:
        # Reshape for group quantization
        if orig_shape[0] % group_size != 0:
            weight = weight[:, :orig_shape[0] // group_size * group_size]
        w = weight.reshape(-1, group_size)
    else:
        w = weight.reshape(-1)

    wmin = w.min(dim=-1, keepdim=True).values
    wmax = w.max(dim=-1, keepdim=True).values
    scale = (wmax - wmin) / (2**bits - 1)
    scale = scale.clamp(min=1e-8)
    w_q = torch.round((w - wmin) / scale) * scale + wmin

    return w_q.reshape(orig_shape)

@torch.no_grad()
def measure_propagation(model, tokenizer, text_samples, perturb_layer_idx, perturb_block="gate_proj"):
    """
    Perturb one linear block in one MoE layer and measure how the error
    propagates to subsequent layers' outputs.
    """
    model.eval()

    # Get baseline hidden states at each layer
    def get_hidden_states(model, input_ids):
        """Hook to capture hidden states at each layer."""
        hidden_states = {}
        hooks = []

        # For Qwen1.5-MoE, find MoE layers
        model_layers = model.model.layers

        for i, layer in enumerate(model_layers):
            def make_hook(layer_idx):
                def hook(module, input, output):
                    # output is a tuple, first element is hidden states
                    if isinstance(output, tuple):
                        hidden_states[layer_idx] = output[0].detach().cpu()
                return hook
            h = layer.register_forward_hook(make_hook(i))
            hooks.append(h)

        outputs = model(input_ids, output_hidden_states=True)
        for h in hooks:
            h.remove()

        return hidden_states

    # Tokenize
    texts = text_samples[:N_SAMPLES]
    all_tokens = []
    for t in texts:
        tokens = tokenizer(t, return_tensors="pt", truncation=True, max_length=512)
        all_tokens.append(tokens.input_ids)

    input_ids = torch.cat(all_tokens, dim=1).to(DEVICE)

    print(f"  Measuring baseline hidden states...")
    baseline_hs = get_hidden_states(model, input_ids)

    # Now perturb one block
    print(f"  Perturbing layer {perturb_layer_idx}, block {perturb_block}...")
    model_layers = model.model.layers
    target_layer = model_layers[perturb_layer_idx]

    # Find the MoE block (mlp) and get the expert's linear block
    mlp = target_layer.mlp
    if hasattr(mlp, 'experts'):
        # Perturb expert 0's gate_proj as a representative
        expert = mlp.experts[0]
        block = getattr(expert, perturb_block)
        orig_weight = block.weight.data.clone()
        q_weight = quantize_weight_rtn(orig_weight, bits=4, group_size=128)
        block.weight.data.copy_(q_weight)
    else:
        print(f"  Layer {perturb_layer_idx} has no MoE experts, skipping")
        return None

    print(f"  Measuring perturbed hidden states...")
    perturbed_hs = get_hidden_states(model, input_ids)

    # Restore original weight
    block.weight.data.copy_(orig_weight)

    # Compute propagation metric: relative error at each subsequent layer
    results = {}
    n_layers = len(model_layers)

    for offset in range(1, N_LAYERS_TO_CHECK + 1):
        target_idx = perturb_layer_idx + offset
        if target_idx >= n_layers:
            break

        baseline = baseline_hs[target_idx].float()
        perturbed = perturbed_hs[target_idx].float()

        # Relative L2 error
        rel_error = (perturbed - baseline).norm().item() / (baseline.norm().item() + 1e-8)

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            baseline.flatten().unsqueeze(0),
            perturbed.flatten().unsqueeze(0)
        ).item()

        results[f"layer_{target_idx}"] = {
            "relative_l2_error": rel_error,
            "cosine_similarity": cos_sim,
            "offset_from_perturbed": offset,
        }

    return results


def main():
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map=DEVICE,
        trust_remote_code=True,
    )

    print("Loading calibration data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in dataset["text"] if len(t) > 100][:N_SAMPLES]

    # Determine which layers are MoE layers (Qwen1.5-MoE has MoE from layer 1 onward)
    model_layers = model.model.layers
    n_layers = len(model_layers)

    # Test perturbation at a few different MoE layers
    test_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4]

    all_results = {}

    for layer_idx in test_layers:
        if layer_idx >= n_layers:
            continue
        # Check if this layer has MoE
        if not hasattr(model_layers[layer_idx].mlp, 'experts'):
            print(f"Layer {layer_idx} is not MoE, skipping...")
            continue

        print(f"\n=== Testing perturbation at layer {layer_idx} ===")
        for block in ["gate_proj", "up_proj", "down_proj"]:
            print(f"\n  Block: {block}")
            try:
                result = measure_propagation(model, tokenizer, texts, layer_idx, block)
                if result:
                    key = f"layer{layer_idx}_{block}"
                    all_results[key] = result
                    for k, v in result.items():
                        print(f"    {k}: rel_error={v['relative_l2_error']:.6f}, cos_sim={v['cosine_similarity']:.6f}")
            except Exception as e:
                print(f"    Error: {e}")

    # Save results
    output_path = Path("/data/zengyq/paper/baselines/MxMoE/pilot_dependency_horizon.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n=== SUMMARY ===")
    for key, layers in all_results.items():
        print(f"\n{key}:")
        for layer_name, metrics in layers.items():
            print(f"  {layer_name} (offset +{metrics['offset_from_perturbed']}): "
                  f"rel_error={metrics['relative_l2_error']:.4f}, "
                  f"cos_sim={metrics['cosine_similarity']:.4f}")


if __name__ == "__main__":
    main()
