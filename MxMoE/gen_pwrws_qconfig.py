"""Per-Weight Route-Weighted Sensitivity (PW-RWS) allocation for MoE quantization.

Key insight: Current MoE quantization assigns the same precision to all weight
matrices (gate_proj, up_proj, down_proj) within an expert. But these weights
have very different sensitivity — down_proj can be 15x more sensitive than
gate_proj within the same expert.

Our method: Per-weight granularity allocation
1. Score each weight matrix independently by route-weighted reconstruction loss
2. Rank ALL weight groups across all layers, experts, and matrices globally
3. Assign W4 to top-k weight groups, meeting total bit budget
4. This gives 3x finer granularity than per-expert allocation
"""
import json
import argparse
import os

W2 = {"w_bits": 2, "w_gsize": 128, "w_sym": False, "w_clip": [1.0, 1.0],
       "a_bits": 16, "a_gsize": 128, "a_sym": False, "a_clip": [1.0, 1.0]}
W4 = {"w_bits": 4, "w_gsize": -1, "w_sym": False, "w_clip": [1.0, 1.0],
       "a_bits": 16, "a_gsize": -1, "a_sym": False, "a_clip": [1.0, 1.0]}

MODEL_INFO = {
    "qwen2_moe":    {"n_routed": 60, "n_shared": 1, "skip_layer0": False},
    "deepseek_moe": {"n_routed": 64, "n_shared": 1, "skip_layer0": True},
    "mixtral":      {"n_routed": 8,  "n_shared": 0, "skip_layer0": False},
}

WEIGHT_NAMES = ["gate", "up", "down"]
W2_BITS = 2.25
W4_BITS = 4.0


def load_recon_loss(model):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    w2_file = f"{cur_dir}/calib/{model}-MOE-rtn-W2A16_g128_asym-wiki2-128-4096-layer_out_norm.json"
    w4_file = f"{cur_dir}/calib/{model}-MOE-rtn-W4A16_g-1_asym-wiki2-128-4096-layer_out_norm.json"
    with open(w2_file) as f:
        w2 = json.load(f)
    with open(w4_file) as f:
        w4 = json.load(f)
    return w2, w4


def get_layer_info(li, w2_data, n_shared):
    layer = str(li)
    n_total = len(w2_data[layer])
    if n_total <= 1:
        return 0, n_total
    return n_total - n_shared, n_shared


def generate_pwrws_qconfig(model, target_bits, gate_file, use_rws=True):
    w2_data, w4_data = load_recon_loss(model)
    with open(gate_file) as f:
        trace = json.load(f)

    info = MODEL_INFO[model]
    n_routed = info["n_routed"]
    n_shared = info["n_shared"]
    num_layers = len(w2_data)

    total_routed_weights = sum(
        get_layer_info(li, w2_data, n_shared)[0] * 3
        for li in range(num_layers)
    )
    total_shared_weights = num_layers * n_shared * 3
    total_weights = total_routed_weights + total_shared_weights
    shared_budget = total_shared_weights * W4_BITS
    total_budget = target_bits * total_weights
    routed_budget = total_budget - shared_budget
    n_w4 = (routed_budget - total_routed_weights * W2_BITS) / (W4_BITS - W2_BITS)
    total_n4 = max(0, round(n_w4))

    print(f"Target: {target_bits}b, Total weight groups: {total_routed_weights} routed + {total_shared_weights} shared")
    print(f"W4 budget: {total_n4} / {total_routed_weights} routed weight groups ({total_n4/total_routed_weights:.1%})")

    # Step 1: Score each weight group independently, normalized per layer per weight type
    raw_scores = {}
    for li in range(num_layers):
        layer = str(li)
        layer_n_routed, _ = get_layer_info(li, w2_data, n_shared)
        if layer_n_routed == 0:
            continue

        for wi in range(3):
            vals = []
            for ei in range(layer_n_routed):
                ei_str = str(ei)
                w2_loss = w2_data[layer][ei_str][wi]
                if layer in w4_data and ei_str in w4_data[layer]:
                    w4_loss = w4_data[layer][ei_str][wi]
                    sensitivity = w2_loss / w4_loss if w4_loss > 0 else 1.0
                else:
                    sensitivity = w2_loss
                vals.append(sensitivity)

            max_val = max(vals) if vals else 1.0
            for ei in range(layer_n_routed):
                raw_scores[(li, ei, wi)] = vals[ei] / max_val if max_val > 0 else 0

    # Step 2: Apply route weighting
    all_groups = []
    for li in range(num_layers):
        layer_n_routed, _ = get_layer_info(li, w2_data, n_shared)
        if layer_n_routed == 0:
            continue

        freq = trace[f"layer-{li}"]["access_freq"]
        max_freq = max(freq) if max(freq) > 0 else 1.0

        for ei in range(layer_n_routed):
            f_val = freq[ei] if ei < len(freq) else 0
            route_weight = f_val / (max_freq + 1e-8)

            for wi in range(3):
                sensitivity = raw_scores[(li, ei, wi)]
                if use_rws:
                    score = sensitivity * (1 + route_weight)
                else:
                    score = sensitivity
                all_groups.append((li, ei, wi, score))

    # Step 3: Global ranking
    all_groups.sort(key=lambda x: x[3], reverse=True)

    # Step 4: Assign W4 to top-k
    w4_set = set()
    for li, ei, wi, score in all_groups[:total_n4]:
        w4_set.add((li, ei, wi))

    # Step 5: Build qconfig
    qconfig = {"LT": {}}

    for li in range(num_layers):
        layer = str(li)
        layer_n_routed, layer_n_shared = get_layer_info(li, w2_data, n_shared)

        if layer_n_routed == 0:
            qconfig["LT"][layer] = [0, 0]
            qconfig[layer] = {"experts": {"0": {"gate": W4, "up": W4, "down": W4}}}
            continue

        experts = {}
        for ei in range(layer_n_routed):
            expert_cfg = {}
            for wi, wname in enumerate(WEIGHT_NAMES):
                cfg = W4 if (li, ei, wi) in w4_set else W2
                expert_cfg[wname] = cfg
            experts[str(ei)] = expert_cfg

        if layer_n_shared > 0:
            experts[str(layer_n_routed)] = {"gate": W4, "up": W4, "down": W4}

        qconfig[layer] = {"experts": experts}
        qconfig["LT"][layer] = [0, 0]

    # Summary
    print("\n=== Per-Weight Allocation Summary ===")
    for li in range(num_layers):
        layer_n_routed, layer_n_shared = get_layer_info(li, w2_data, n_shared)
        if layer_n_routed == 0:
            continue
        n_w4 = [sum(1 for ei in range(layer_n_routed) if (li, ei, wi) in w4_set) for wi in range(3)]
        if li % 6 == 0 or li == num_layers - 1:
            print(f"  L{li:2d}: gate={n_w4[0]:2d}W4  up={n_w4[1]:2d}W4  down={n_w4[2]:2d}W4")

    mixed = 0
    for li in range(num_layers):
        layer_n_routed, _ = get_layer_info(li, w2_data, n_shared)
        for ei in range(layer_n_routed):
            n_w4_ei = sum(1 for wi in range(3) if (li, ei, wi) in w4_set)
            if 0 < n_w4_ei < 3:
                mixed += 1
    print(f"\n  Mixed experts (not all-W4 or all-W2): {mixed}")

    return qconfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["qwen2_moe", "deepseek_moe", "mixtral"])
    parser.add_argument("--wbits", type=float, required=True)
    parser.add_argument("--use_rws", action="store_true", default=True)
    parser.add_argument("--no_rws", dest="use_rws", action="store_false")
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    gate_file = f"{CUR_DIR}/calib/gate/{args.model}/wiki2/4096/moe-gate.json"

    mode = "rws" if args.use_rws else "norews"
    if args.save is None:
        args.save = f"{CUR_DIR}/qconfigs/global_rws/{args.model}_rtn_pwrws_{mode}_wbits{args.wbits}.json"

    print(f"Generating PW-RWS qconfig: model={args.model}, target={args.wbits}b, rws={args.use_rws}")
    qconfig = generate_pwrws_qconfig(args.model, args.wbits, gate_file, args.use_rws)

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    with open(args.save, "w") as f:
        json.dump(qconfig, f, indent=2)
    print(f"Saved to {args.save}")
