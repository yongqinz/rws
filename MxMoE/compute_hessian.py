"""Compute Hessian trace per expert for MoE models using Hutchinson's method.

For each expert in each MoE layer, computes Hessian trace for gate_proj,
up_proj, and down_proj weights, sums them as per-expert sensitivity,
then normalizes per layer.

Output: JSON file with per-expert Hessian sensitivity scores.
"""
import torch
import torch.nn as nn
import json
import argparse
import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoConfig

MODEL_INFO = {
    "qwen2_moe": {
        "num_routed": 60,
        "has_shared": True,
        "shared_name": "shared_expert",  # singular
        "skip_layer0": False,
        "weight_names": ["gate_proj", "up_proj", "down_proj"],
    },
    "deepseek_moe": {
        "num_routed": 64,
        "has_shared": True,
        "shared_name": "shared_experts",  # plural
        "skip_layer0": True,
        "weight_names": ["gate_proj", "up_proj", "down_proj"],
    },
    "mixtral": {
        "num_routed": 8,
        "has_shared": False,
        "shared_name": None,
        "skip_layer0": False,
        "weight_names": ["w1", "w3", "w2"],  # Mixtral uses w1,w2,w3
    },
}


def hessian_trace_hutchinson(weight, num_samples=50):
    """Approximate trace of Hessian using Hutchinson's method."""
    weight = weight.detach().float()
    trace_estimates = []

    for _ in range(num_samples):
        v = torch.randn_like(weight)
        v = v / torch.norm(v)

        weight.requires_grad_(True)
        loss = torch.norm(weight)
        grad1 = torch.autograd.grad(loss, weight, create_graph=True)[0]
        hvp = torch.autograd.grad(grad1, weight, grad_outputs=v)[0]

        trace_estimates.append(torch.sum(v * hvp).item())

    return sum(trace_estimates) / num_samples


def compute_expert_hessian(model, model_id, num_samples=50):
    """Compute Hessian sensitivity for all experts across all layers."""
    info = MODEL_INFO[model_id]
    wnames = info["weight_names"]

    layers = model.model.layers
    num_layers = len(layers)

    hessian_dict = {}

    for li in range(num_layers):
        if info["skip_layer0"] and li == 0:
            continue

        layer = layers[li]
        mlp = layer.mlp

        # Routed experts
        for ei in range(info["num_routed"]):
            key = f"L{li}_E{ei}"
            expert = mlp.experts[ei]

            total_hessian = 0.0
            for wname in wnames:
                w = getattr(expert, wname).weight
                h = hessian_trace_hutchinson(w, num_samples)
                total_hessian += h

            hessian_dict[key] = total_hessian
            print(f"  {key}: hessian={total_hessian:.4f}")

        # Shared expert
        if info["has_shared"]:
            shared = getattr(mlp, info["shared_name"], None)
            if shared is not None:
                key = f"L{li}_Eshared"
                total_hessian = 0.0
                for wname in wnames:
                    w = getattr(shared, wname).weight
                    h = hessian_trace_hutchinson(w, num_samples)
                    total_hessian += h
                hessian_dict[key] = total_hessian
                print(f"  {key}: hessian={total_hessian:.4f}")

    # Normalize per layer
    for li in range(num_layers):
        if info["skip_layer0"] and li == 0:
            continue

        layer_keys = [k for k in hessian_dict if k.startswith(f"L{li}_")]
        layer_sum = sum(hessian_dict[k] for k in layer_keys)

        if layer_sum > 0:
            for k in layer_keys:
                hessian_dict[k] /= layer_sum

    return hessian_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["qwen2_moe", "deepseek_moe", "mixtral"])
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    MODEL_PATHS = {
        "qwen2_moe": "/data/zengyq/models/Qwen1.5-MoE-A2.7B",
        "deepseek_moe": "/data/zengyq/models/deepseek-moe-16b-base",
        "mixtral": "/data/zengyq/models/Mixtral-8x7B-v0.1",
    }

    if args.save is None:
        args.save = f"calib/hessian/{args.model}_hessian.json"

    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATHS[args.model],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    print(f"Computing Hessian (num_samples={args.num_samples})...")
    hessian = compute_expert_hessian(model, args.model, args.num_samples)

    with open(args.save, "w") as f:
        json.dump(hessian, f, indent=2)
    print(f"Saved to {args.save}")
