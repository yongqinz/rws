"""
RouteQEP: Route-Conditioned Error Propagation for MoE Mixed-Precision Quantization.

Core idea: MoE quantization error propagates through conditional expert routes,
not sequential layers. We compute a route-conditioned propagation operator that
replaces per-block local sensitivity with propagated sensitivity in the ILP.

Key objects:
  - Expert transition matrix P_{i,i'}^{(l)} = Pr(e_{l+1}=i' | e_l=i)
  - Route Heterogeneity Index RHI_l = ||P - 1*pi^T||_F
  - Propagated delta: delta_hat = g^{(l)} * delta_local
"""

import json
import math
import numpy as np
from pathlib import Path
from typing import Optional


def compute_corouting_mass(
    topk_idx_per_layer: list[list[np.ndarray]],
    topk_weight_per_layer: list[list[np.ndarray]],
    num_experts: int,
) -> list[np.ndarray]:
    """
    Compute co-routing mass C_{i,i'}^{(l)} for adjacent layers.

    C_{i,i'}^{(l)} = sum over tokens t of: sum_{k1 in topK_l} sum_{k2 in topK_{l+1}} rho_{t,i}^{(l)} * rho_{t,i'}^{(l+1)}

    This counts how often expert i at layer l and expert i' at layer l+1 are
    co-activated for the same token (weighted by routing weights).

    Args:
        topk_idx_per_layer: [layer][sample] -> ndarray[seqlen, topk]
        topk_weight_per_layer: [layer][sample] -> ndarray[seqlen, topk]
        num_experts: number of experts per layer

    Returns:
        co_routing: list of ndarray[num_experts, num_experts], one per layer transition
    """
    num_layers = len(topk_idx_per_layer)
    co_routing = []

    for l in range(num_layers - 1):
        C = np.zeros((num_experts, num_experts), dtype=np.float64)

        idx_l = topk_idx_per_layer[l]   # list of [seqlen, topk]
        idx_lp1 = topk_idx_per_layer[l + 1]
        wt_l = topk_weight_per_layer[l]
        wt_lp1 = topk_weight_per_layer[l + 1]

        # Skip if either layer has no routing (e.g., non-MoE layer)
        if len(idx_l) == 0 or len(idx_lp1) == 0:
            co_routing.append(C)
            continue

        for sample_idx in range(min(len(idx_l), len(idx_lp1))):
            # [seqlen, topk]
            experts_l = idx_l[sample_idx]    # experts selected at layer l
            experts_lp1 = idx_lp1[sample_idx]
            weights_l = wt_l[sample_idx]     # routing weights at layer l
            weights_lp1 = wt_lp1[sample_idx]

            seq_len = experts_l.shape[0]

            for tok in range(seq_len):
                for k1 in range(experts_l.shape[1]):
                    e_l = experts_l[tok, k1]
                    w_l = weights_l[tok, k1]
                    for k2 in range(experts_lp1.shape[1]):
                        e_lp1 = experts_lp1[tok, k2]
                        w_lp1 = weights_lp1[tok, k2]
                        C[e_l, e_lp1] += float(w_l) * float(w_lp1)

        co_routing.append(C)

    return co_routing


def compute_expert_transition_matrix(
    co_routing: list[np.ndarray],
) -> list[np.ndarray]:
    """
    Normalize co-routing mass to get conditional expert transition matrix.

    P_{i,i'}^{(l)} = C_{i,i'}^{(l)} / sum_{i'} C_{i,i'}^{(l)}

    Args:
        co_routing: list of ndarray[num_experts, num_experts]

    Returns:
        P: list of ndarray[num_experts, num_experts], row-stochastic
    """
    P_list = []
    for C in co_routing:
        row_sums = C.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)  # avoid division by zero
        P = C / row_sums
        P_list.append(P)
    return P_list


def compute_route_heterogeneity_index(P_list: list[np.ndarray]) -> list[float]:
    """
    Compute Route Heterogeneity Index for each layer transition.

    RHI_l = ||P^{(l)} - 1 * pi^{(l+1)T}||_F

    where pi^{(l+1)} is the marginal expert distribution at layer l+1.
    Dense QEP is equivalent to the rank-1 approximation 1 * pi^T.

    Higher RHI means route-aware propagation differs more from dense QEP.

    Args:
        P_list: list of transition matrices

    Returns:
        rhi: list of float, one per layer transition
    """
    rhi = []
    for P in P_list:
        # Marginal distribution at next layer: pi = row means of P
        pi = P.mean(axis=0)  # [num_experts]
        rank1_approx = np.ones_like(P) * pi[np.newaxis, :]
        rhi_val = np.linalg.norm(P - rank1_approx, 'fro')
        rhi.append(float(rhi_val))
    return rhi


def compute_propagation_gain(
    P_list: list[np.ndarray],
    delta_local: list[list[list[list[float]]]],
    num_hops: int = 1,
    lam: float = 1.0,
) -> list[np.ndarray]:
    """
    Compute route-conditioned propagation gain g^{(l)} for each (expert, block).

    Recursion: g^{(L)} = 1, g^{(l)} = 1 + lambda * M^{(l)} * g^{(l+1)}

    where M^{(l)} is the block-to-block propagation operator:
    M_{(i,j) -> (i',j')} = P_{i,i'} * a_{i',j'}

    and a_{i',j'} = average local delta for block j' in expert i' (downstream gain).

    Args:
        P_list: list of transition matrices [num_layers-1, num_experts, num_experts]
        delta_local: [layer][expert][block][strategy] local quantization losses
        num_hops: number of hops (1 or 2)
        lam: propagation scaling factor

    Returns:
        gain: list of ndarray[num_experts, num_blocks], one per layer
    """
    num_layers = len(delta_local)

    # Compute downstream block gains a_{i',j'} from local deltas
    # Use normalized delta: ratio of worst to best strategy sensitivity.
    # This prevents exponential amplification from raw delta magnitudes.
    downstream_gain = []  # [layer][expert][block]
    for l in range(num_layers):
        layer_deltas = []
        for e in range(len(delta_local[l])):
            expert_gain = []
            for n in range(len(delta_local[l][e])):
                deltas = delta_local[l][e][n]
                if len(deltas) >= 2:
                    # Ratio of worst to best strategy: how much more loss the cheaper option costs
                    ratio = max(deltas) / (min(deltas) + 1e-10)
                    expert_gain.append(float(ratio))
                else:
                    expert_gain.append(1.0)
            layer_deltas.append(expert_gain)
        # Normalize per layer so gains are relative, not absolute
        all_vals = [v for exp in layer_deltas for v in exp]
        layer_max = max(all_vals) if all_vals else 1.0
        layer_gain = [[v / layer_max for v in exp] for exp in layer_deltas]
        downstream_gain.append(layer_gain)

    # Compute propagation gain per layer (backward from last to first)
    # Full backward recursion: g^{(l)} = 1 + lambda * M^{(l)} * g^{(l+1)}
    # With H-hop truncation: g^{(l)} = 1 + lambda * M^{(l)} * g^{(l+1)},
    #   but g^{(l+h)} = 1 for h > num_hops
    gain = [np.ones((len(delta_local[l]), max(len(delta_local[l][e]), 1))) for l in range(num_layers)]

    for l in range(num_layers - 2, -1, -1):
        p_idx = min(l, len(P_list) - 1) if P_list else -1
        if p_idx < 0:
            continue
        P = P_list[p_idx]
        num_experts_l = len(delta_local[l])
        l_next = min(l + 1, num_layers - 1)

        new_gain = np.ones_like(gain[l])

        for i in range(min(num_experts_l, P.shape[0])):
            for j in range(gain[l].shape[1]):
                prop = 0.0
                for ip in range(P.shape[1]):
                    if P[i, ip] < 1e-8:
                        continue
                    if ip >= len(downstream_gain[l_next]) or ip >= gain[l_next].shape[0]:
                        continue
                    for jp in range(min(len(downstream_gain[l_next][ip]), gain[l_next].shape[1])):
                        a_ip_jp = downstream_gain[l_next][ip][jp]
                        g_lp1 = gain[l_next][ip, jp]
                        prop += P[i, ip] * a_ip_jp * g_lp1

                new_gain[i, j] = 1.0 + lam * prop

        gain[l] = new_gain

    # Apply hop truncation: zero out gain for layers beyond num_hops from end
    # For num_hops=1: every layer gets direct 1-hop gain (no chaining)
    # For num_hops=2: full backward chain
    if num_hops == 1:
        # Recompute with no chaining: each layer only sees its immediate next layer
        gain = [np.ones((len(delta_local[l]), max(len(delta_local[l][e]), 1))) for l in range(num_layers)]
        for l in range(num_layers - 2, -1, -1):
            p_idx = min(l, len(P_list) - 1) if P_list else -1
            if p_idx < 0:
                continue
            P = P_list[p_idx]
            num_experts_l = len(delta_local[l])
            l_next = min(l + 1, num_layers - 1)

            new_gain = np.ones_like(gain[l])
            for i in range(min(num_experts_l, P.shape[0])):
                for j in range(gain[l].shape[1]):
                    prop = 0.0
                    for ip in range(P.shape[1]):
                        if P[i, ip] < 1e-8:
                            continue
                        if ip >= len(downstream_gain[l_next]):
                            continue
                        for jp in range(min(len(downstream_gain[l_next][ip]), gain[l_next].shape[1])):
                            a_ip_jp = downstream_gain[l_next][ip][jp]
                            prop += P[i, ip] * a_ip_jp * 1.0  # g=1 for next layer (no chaining)
                    new_gain[i, j] = 1.0 + lam * prop
            gain[l] = new_gain
    # For num_hops >= 2: the full backward chain above is already correct

    return gain


def apply_propagation_to_delta(
    delta_local: list[list[list[list[float]]]],
    gain: list[np.ndarray],
) -> list[list[list[list[float]]]]:
    """
    Apply propagation gain to local delta: delta_hat = gain * delta_local.

    Args:
        delta_local: [layer][expert][block][strategy]
        gain: [layer] -> ndarray[num_experts, num_blocks]

    Returns:
        delta_hat: same shape as delta_local but with propagated values
    """
    delta_hat = []
    for l in range(len(delta_local)):
        layer_delta = []
        for e in range(len(delta_local[l])):
            expert_delta = []
            for n in range(len(delta_local[l][e])):
                block_delta = []
                g = gain[l][e, n] if e < gain[l].shape[0] and n < gain[l].shape[1] else 1.0
                for s in range(len(delta_local[l][e][n])):
                    block_delta.append(delta_local[l][e][n][s] * g)
                expert_delta.append(block_delta)
            layer_delta.append(expert_delta)
        delta_hat.append(layer_delta)
    return delta_hat


def load_transition_matrix_from_trace(trace_file: str) -> list[np.ndarray]:
    """
    Load expert transition matrix from a pre-computed trace file.

    If a co-routing trace file exists (e.g., calib/gate/{model}/corouting.json),
    load it. Otherwise, return identity matrices (no propagation).
    """
    corouting_file = trace_file.replace("moe-gate.json", "corouting.json")
    if Path(corouting_file).exists():
        with open(corouting_file, "r") as f:
            data = json.load(f)
        P_list = [np.array(p) for p in data.get("transition_matrices", [])]
        return P_list
    return []


def save_propagation_stats(
    save_path: str,
    P_list: list[np.ndarray],
    rhi: list[float],
    gain: list[np.ndarray],
):
    """Save propagation statistics for analysis."""
    data = {
        "route_heterogeneity_index": rhi,
        "transition_matrices": [P.tolist() for P in P_list],
        "propagation_gain_stats": [
            {
                "mean": float(g.mean()),
                "std": float(g.std()),
                "min": float(g.min()),
                "max": float(g.max()),
            }
            for g in gain
        ],
    }
    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Propagation stats saved to {save_path}")


def load_access_probs_from_trace(trace_file: str) -> list[list[float]]:
    """Load expert access probabilities from gate trace."""
    with open(trace_file, "r") as f:
        trace = json.load(f)
    access_probs = []
    for l in range(trace["num_layers"]):
        freq = trace[f"layer-{l}"]["access_freq"]
        total = sum(freq)
        if total == 0:
            probs = [1.0 / len(freq)] * len(freq)
        else:
            probs = [f / total for f in freq]
        access_probs.append(probs)
    return access_probs


def compute_route_weighted_delta(
    delta_local: list[list[list[list[float]]]],
    trace_file: str,
) -> tuple[list[list[list[list[float]]]], list[float]]:
    """
    Weight each expert's delta by its routing probability.
    High-frequency experts get amplified sensitivity, ensuring the ILP
    prioritizes their precision.

    This avoids the exponential amplification problem of cross-layer propagation.
    """
    access_probs = load_access_probs_from_trace(trace_file)
    num_layers = len(delta_local)

    # Shared expert (last expert index) gets max weight since it processes all tokens
    rhi_vals = []

    delta_hat = []
    for l in range(num_layers):
        probs = access_probs[l] if l < len(access_probs) else [1.0] * len(delta_local[l])
        max_prob = max(probs) if probs else 1.0
        layer_delta = []
        for e in range(len(delta_local[l])):
            # MoE experts: weight by access probability
            # Shared expert (last): weight by 1.0 (processes all tokens)
            if e < len(probs):
                weight = probs[e] / max_prob  # normalize to [0, 1]
            else:
                weight = 1.0  # shared expert
            expert_delta = []
            for n in range(len(delta_local[l][e])):
                block_delta = [d * (1.0 + weight) for d in delta_local[l][e][n]]
                expert_delta.append(block_delta)
            layer_delta.append(expert_delta)
        delta_hat.append(layer_delta)
        rhi_vals.append(float(np.std(probs) / (np.mean(probs) + 1e-10)))

    return delta_hat, rhi_vals


def compute_propagated_delta(
    delta_local: list[list[list[list[float]]]],
    trace_file: Optional[str] = None,
    P_list: Optional[list[np.ndarray]] = None,
    num_hops: int = 1,
    lam: float = 1.0,
) -> tuple[list[list[list[list[float]]]], list[float]]:
    """
    Main entry point: compute route-conditioned propagated delta.

    Args:
        delta_local: [layer][expert][block][strategy] local quantization losses
        trace_file: path to routing trace (for loading pre-computed transition matrices)
        P_list: pre-computed transition matrices (alternative to trace_file)
        num_hops: 0 (no propagation), 1, or 2
        lam: propagation scaling factor

    Returns:
        delta_hat: propagated delta (same shape as delta_local)
        rhi: Route Heterogeneity Index per layer transition
    """
    if num_hops == 0:
        return delta_local, []

    # Route-weighted mode: weight by access probability instead of cross-layer propagation
    if num_hops == 1 and trace_file is not None:
        print("Using route-weighted sensitivity (access probability weighting)")
        return compute_route_weighted_delta(delta_local, trace_file)

    # Cross-layer propagation mode (num_hops >= 2)
    if P_list is None and trace_file is not None:
        P_list = load_transition_matrix_from_trace(trace_file)

    if not P_list:
        print("WARNING: No transition matrix data found. Using local delta (no propagation).")
        return delta_local, [0.0]

    rhi = compute_route_heterogeneity_index(P_list)
    if not rhi:
        rhi = [0.0]
    print(f"Route Heterogeneity Index: mean={np.mean(rhi):.4f}, max={np.max(rhi):.4f}")

    gain = compute_propagation_gain(P_list, delta_local, num_hops=num_hops, lam=lam)
    delta_hat = apply_propagation_to_delta(delta_local, gain)

    return delta_hat, rhi
