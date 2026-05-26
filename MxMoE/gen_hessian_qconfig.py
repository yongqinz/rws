"""Generate qconfigs using Hessian sensitivity for bit allocation.

For each model + budget:
1. Load Hessian sensitivity per expert (normalized per layer)
2. Load gate routing frequency per expert
3. Compute RWS: weighted_sensitivity = hessian * (1 + freq/max_freq)
4. Assign W4 to top-k experts by weighted sensitivity
5. Save qconfig
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


def compute_n4(target_bits, n_routed, n_shared):
    total_experts = n_routed + n_shared
    shared_bits = n_shared * W4_BITS
    routed_bits_budget = target_bits * total_experts - shared_bits
    n4 = (routed_bits_budget - n_routed * W2_BITS) / (W4_BITS - W2_BITS)
    return max(0, min(n_routed, round(n4)))


def generate_qconfig(model, target_bits, hessian_file, gate_file, use_rws=True):
    with open(hessian_file) as f:
        hessian = json.load(f)
    with open(gate_file) as f:
        trace = json.load(f)

    info = MODEL_INFO[model]
    n_routed = info["n_routed"]
    n_shared = info["n_shared"]
    num_layers = trace["num_layers"]

    qconfig = {"LT": {}}
    n4 = compute_n4(target_bits, n_routed, n_shared)

    for li in range(num_layers):
        if info["skip_layer0"] and li == 0:
            qconfig["LT"][str(li)] = [0, 0]
            qconfig[str(li)] = {"experts": {"0": {"gate": W4, "up": W4, "down": W4}}}
            continue

        freq = trace[f"layer-{li}"]["access_freq"]
        max_freq = max(freq)

        # Compute score for each expert
        scores = []
        for ei in range(n_routed):
            h_key = f"L{li}_E{ei}"
            h_val = hessian.get(h_key, 1.0 / n_routed)
            f_val = freq[ei]

            if use_rws:
                # RWS: scale hessian by routing frequency
                w = f_val / (max_freq + 1e-8)
                score = h_val * (1 + w)
            else:
                # Baseline: raw hessian only (no routing)
                score = h_val

            scores.append((ei, score))

        # Sort by score descending, top-n4 get W4
        scores.sort(key=lambda x: x[1], reverse=True)
        w4_set = set(ei for ei, _ in scores[:n4])

        experts = {}
        for ei in range(n_routed):
            cfg = W4 if ei in w4_set else W2
            experts[str(ei)] = {"gate": cfg, "up": cfg, "down": cfg}

        # Shared expert always W4
        if n_shared > 0:
            experts[str(n_routed)] = {"gate": W4, "up": W4, "down": W4}

        qconfig[str(li)] = {"experts": experts}
        qconfig["LT"][str(li)] = [0, 0]

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
    hessian_file = f"{CUR_DIR}/calib/hessian/{args.model}_hessian.json"
    gate_file = f"{CUR_DIR}/calib/gate/{args.model}/wiki2/4096/moe-gate.json"

    mode = "rws" if args.use_rws else "hess_only"
    if args.save is None:
        args.save = f"{CUR_DIR}/qconfigs/hessian/{args.model}_rtn_hess_{mode}_wbits{args.wbits}.json"

    print(f"Generating Hessian qconfig: model={args.model}, target={args.wbits}b, rws={args.use_rws}")
    qconfig = generate_qconfig(args.model, args.wbits, hessian_file, gate_file, args.use_rws)

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    with open(args.save, "w") as f:
        json.dump(qconfig, f, indent=2)
    print(f"Saved to {args.save}")
