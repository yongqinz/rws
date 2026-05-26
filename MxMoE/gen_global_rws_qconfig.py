"""Global Route-Weighted Sensitivity allocation for MoE quantization.

Key insight: Reconstruction loss varies 10-45x across layers, yet current methods
(MxMoE, Hessian, Freq-only) all allocate the same number of W4 experts per layer.

Our method: Route-Weighted Global Allocation (RWGA)
1. Score each expert by route-weighted reconstruction loss sensitivity
2. Rank ALL experts across ALL layers globally
3. Assign W4 to top-k experts globally (meeting total bit budget)
4. Floor constraint: at least 1 W4 routed expert per layer

This allows:
- Highly sensitive later layers to receive more W4 budget
- Robust early layers to contribute budget savings
- Much better overall bit efficiency than per-layer uniform allocation
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

W2_BITS = 2.25
W4_BITS = 4.0


def load_recon_loss(model):
    """Load per-expert reconstruction loss from MxMoE calibration."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    w2_file = f"{cur_dir}/calib/{model}-MOE-rtn-W2A16_g128_asym-wiki2-128-4096-layer_out_norm.json"
    w4_file = f"{cur_dir}/calib/{model}-MOE-rtn-W4A16_g-1_asym-wiki2-128-4096-layer_out_norm.json"

    with open(w2_file) as f:
        w2 = json.load(f)
    with open(w4_file) as f:
        w4 = json.load(f)

    return w2, w4


def compute_total_n4(target_bits, n_routed, n_shared, num_layers):
    """Compute total W4 experts needed across all layers to meet target bits."""
    total_experts_per_layer = n_routed + n_shared
    total_experts = total_experts_per_layer * num_layers
    shared_total = n_shared * num_layers  # shared always W4
    routed_total = n_routed * num_layers

    total_budget = target_bits * total_experts
    routed_budget = total_budget - shared_total * W4_BITS
    total_n4 = (routed_budget - routed_total * W2_BITS) / (W4_BITS - W2_BITS)
    return max(0, round(total_n4))


def generate_global_rws_qconfig(model, target_bits, gate_file, use_rws=True):
    """Generate qconfig using global route-weighted sensitivity allocation."""
    w2_data, w4_data = load_recon_loss(model)
    with open(gate_file) as f:
        trace = json.load(f)

    info = MODEL_INFO[model]
    n_routed = info["n_routed"]
    n_shared = info["n_shared"]
    num_layers = len(w2_data)

    # Determine per-layer expert counts from calibration data
    def get_layer_info(li):
        layer = str(li)
        n_total = len(w2_data[layer])
        if n_total <= 1:
            return 0, n_total  # skip layer or shared-only
        return n_total - n_shared, n_shared

    total_n4 = compute_total_n4(target_bits, n_routed, n_shared, num_layers)
    print(f"Total W4 budget: {total_n4} routed experts across {num_layers} layers")
    print(f"  Per-layer uniform would give: {total_n4 // num_layers} W4 per layer")

    # Step 1: Compute route-weighted sensitivity for every expert
    # First pass: compute raw sensitivity
    raw_scores = {}  # (li, ei) -> score
    for li in range(num_layers):
        layer = str(li)
        layer_n_routed, layer_n_shared = get_layer_info(li)
        if layer_n_routed == 0:
            continue

        for ei in range(layer_n_routed):
            ei_str = str(ei)
            w2_loss = sum(w2_data[layer][ei_str])
            if layer in w4_data and ei_str in w4_data[layer]:
                w4_loss = sum(w4_data[layer][ei_str])
                sensitivity = w2_loss / w4_loss if w4_loss > 0 else 1.0
            else:
                sensitivity = w2_loss
            raw_scores[(li, ei)] = sensitivity

    # Normalize per layer (so cross-layer comparison is meaningful)
    for li in range(num_layers):
        layer_n_routed, _ = get_layer_info(li)
        if layer_n_routed == 0:
            continue
        layer_vals = [raw_scores[(li, ei)] for ei in range(layer_n_routed)]
        layer_max = max(layer_vals) if layer_vals else 1.0
        if layer_max > 0:
            for ei in range(layer_n_routed):
                raw_scores[(li, ei)] /= layer_max

    # Second pass: apply route weighting
    all_experts = []
    for li in range(num_layers):
        layer_n_routed, _ = get_layer_info(li)
        if layer_n_routed == 0:
            continue

        freq = trace[f"layer-{li}"]["access_freq"]
        max_freq = max(freq) if max(freq) > 0 else 1.0

        for ei in range(layer_n_routed):
            sensitivity = raw_scores[(li, ei)]
            if use_rws and ei < len(freq):
                f_val = freq[ei]
                weight = f_val / (max_freq + 1e-8)
                score = sensitivity * (1 + weight)
            else:
                score = sensitivity

            all_experts.append((li, ei, score, sensitivity))

    # Step 2: Global ranking
    all_experts.sort(key=lambda x: x[2], reverse=True)

    # Step 3: Allocate W4 to top-k globally, with floor constraint
    # First pass: ensure minimum 1 W4 per layer
    w4_set = {}  # layer -> set of expert indices with W4
    active_layers = set()
    for li in range(num_layers):
        layer_n_routed, _ = get_layer_info(li)
        if layer_n_routed == 0:
            continue
        active_layers.add(li)
        w4_set[li] = set()

    # Floor: assign top-1 expert per layer first
    floor_assigned = 0
    for li in sorted(active_layers):
        layer_experts = [(li, ei, s, raw) for li_, ei, s, raw in all_experts if li_ == li]
        if layer_experts:
            top_expert = layer_experts[0]
            w4_set[li].add(top_expert[1])
            floor_assigned += 1

    remaining_budget = total_n4 - floor_assigned
    print(f"  Floor assignment: {floor_assigned} experts (1 per layer)")
    print(f"  Remaining budget: {remaining_budget} experts")

    # Second pass: fill remaining budget from global ranking
    if remaining_budget > 0:
        for li, ei, score, raw in all_experts:
            if ei in w4_set[li]:
                continue  # already assigned as floor
            if remaining_budget <= 0:
                break
            w4_set[li].add(ei)
            remaining_budget -= 1

    # Step 4: Build qconfig
    qconfig = {"LT": {}}

    for li in range(num_layers):
        layer = str(li)
        layer_n_routed, layer_n_shared = get_layer_info(li)
        if layer_n_routed == 0:
            # Skip layer (e.g. DeepSeek layer 0 = shared only)
            qconfig["LT"][layer] = [0, 0]
            qconfig[layer] = {"experts": {"0": {"gate": W4, "up": W4, "down": W4}}}
            continue

        experts = {}
        for ei in range(layer_n_routed):
            cfg = W4 if ei in w4_set.get(li, set()) else W2
            experts[str(ei)] = {"gate": cfg, "up": cfg, "down": cfg}

        # Shared expert always W4
        if layer_n_shared > 0:
            experts[str(layer_n_routed)] = {"gate": W4, "up": W4, "down": W4}

        qconfig[layer] = {"experts": experts}
        qconfig["LT"][layer] = [0, 0]

    # Print allocation summary
    print("\n=== Allocation Summary ===")
    for li in sorted(active_layers):
        layer_n_routed, layer_n_shared = get_layer_info(li)
        n_w4 = len(w4_set.get(li, set()))
        n_w2 = layer_n_routed - n_w4
        avg_bits = (n_w4 * W4_BITS + n_w2 * W2_BITS + layer_n_shared * W4_BITS) / (layer_n_routed + layer_n_shared)
        print(f"  Layer {li:2d}: {n_w4:2d} W4 + {n_w2:2d} W2 (avg {avg_bits:.2f}b)")
    print()

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
        args.save = f"{CUR_DIR}/qconfigs/global_rws/{args.model}_rtn_grws_{mode}_wbits{args.wbits}.json"

    print(f"Generating Global RWS qconfig: model={args.model}, target={args.wbits}b, rws={args.use_rws}")
    qconfig = generate_global_rws_qconfig(args.model, args.wbits, gate_file, args.use_rws)

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    with open(args.save, "w") as f:
        json.dump(qconfig, f, indent=2)
    print(f"Saved to {args.save}")
