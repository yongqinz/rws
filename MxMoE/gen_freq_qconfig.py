"""Generate frequency-only allocation qconfigs.

Assigns W4 to the highest-traffic experts (all 3 blocks), W2 to the rest,
to achieve a target average bits/param. Shared experts always get W4.
"""
import json
import argparse
import os

W2 = {"w_bits": 2, "w_gsize": 128, "w_sym": False, "w_clip": [1.0, 1.0],
       "a_bits": 16, "a_gsize": 128, "a_sym": False, "a_clip": [1.0, 1.0]}
W4 = {"w_bits": 4, "w_gsize": -1, "w_sym": False, "w_clip": [1.0, 1.0],
       "a_bits": 16, "a_gsize": -1, "a_sym": False, "a_clip": [1.0, 1.0]}

MODEL_INFO = {
    "qwen2_moe":    {"n_routed": 60, "n_shared_equiv": 4,  "skip_layer0": False},
    "mixtral":      {"n_routed": 8,  "n_shared_equiv": 0,  "skip_layer0": False},
    "deepseek_moe": {"n_routed": 64, "n_shared_equiv": 2,  "skip_layer0": True},
}

W2_BITS = 2.25
W4_BITS = 4.0


def compute_n4(target_bits, n_routed, n_shared_equiv):
    """Compute how many routed experts get W4 to hit the target bits/param."""
    total_experts = n_routed + n_shared_equiv
    # shared always W4
    shared_bits = n_shared_equiv * W4_BITS
    routed_bits_budget = target_bits * total_experts - shared_bits
    n4 = (routed_bits_budget - n_routed * W2_BITS) / (W4_BITS - W2_BITS)
    return max(0, min(n_routed, round(n4)))


def generate_qconfig(model, target_bits, trace_file):
    with open(trace_file) as f:
        trace = json.load(f)

    info = MODEL_INFO[model]
    n_routed = info["n_routed"]
    n_shared = info["n_shared_equiv"]
    num_layers = trace["num_layers"]

    qconfig = {"LT": {}}

    for li in range(num_layers):
        if info["skip_layer0"] and li == 0:
            # DeepSeek layer 0: shared expert only, always W4
            qconfig["LT"][str(li)] = [0, 0]
            qconfig[str(li)] = {"experts": {"0": {"gate": W4, "up": W4, "down": W4}}}
            continue

        freq = trace[f"layer-{li}"]["access_freq"]
        # freq has n_routed entries
        assert len(freq) == n_routed, f"Expected {n_routed} experts, got {len(freq)}"

        n4 = compute_n4(target_bits, n_routed, n_shared)

        # Sort experts by frequency (descending), get top-n4 indices
        sorted_indices = sorted(range(n_routed), key=lambda i: freq[i], reverse=True)
        w4_set = set(sorted_indices[:n4])

        experts = {}
        for ei in range(n_routed):
            cfg = W4 if ei in w4_set else W2
            experts[str(ei)] = {"gate": cfg, "up": cfg, "down": cfg}

        # Shared expert (last one, always W4)
        if n_shared > 0:
            experts[str(n_routed)] = {"gate": W4, "up": W4, "down": W4}

        qconfig[str(li)] = {"experts": experts}
        qconfig["LT"][str(li)] = [0, 0]

        # Verify actual bits
        total_bits = sum(W4_BITS if ei in w4_set else W2_BITS for ei in range(n_routed))
        total_bits += n_shared * W4_BITS
        actual = total_bits / (n_routed + n_shared)
        if li == 0:
            print(f"  L0: n4={n4}/{n_routed}, target={target_bits}, actual={actual:.4f}")

    return qconfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--wbits", type=float, required=True)
    parser.add_argument("--trace_file", default=None)
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    if args.trace_file is None:
        args.trace_file = f"{CUR_DIR}/calib/gate/{args.model}/wiki2/4096/moe-gate.json"
    if args.save is None:
        args.save = f"{CUR_DIR}/qconfigs/freq_only/{args.model}_rtn_freq_wbits{args.wbits}.json"

    print(f"Generating freq-only qconfig: model={args.model}, target={args.wbits}b")
    qconfig = generate_qconfig(args.model, args.wbits, args.trace_file)

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    with open(args.save, "w") as f:
        json.dump(qconfig, f, indent=2)
    print(f"Saved to {args.save}")
