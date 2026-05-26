"""
Paper Figure Generation Script for Route-Weighted Sensitivity in MoE Quantization.

Generates all analysis figures from completed experiment data.
Run: python plot_analysis.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
from pathlib import Path
from typing import Optional

BASE = Path("/data/zengyq/paper/baselines/MxMoE")
RESULTS = BASE / "results"
QCONFIGS = BASE / "qconfigs" / "w2a16_g128_asym+w4a16_g-1_asym"
GATE = BASE / "calib" / "gate"
FIGURES = BASE / "figures"
FIGURES.mkdir(exist_ok=True)

# Style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = {
    'qwen2_moe_baseline': '#1f77b4',
    'qwen2_moe_routewtd': '#ff7f0e',
    'deepseek_moe_baseline': '#2ca02c',
    'deepseek_moe_routewtd': '#d62728',
    'mixtral_baseline': '#9467bd',
    'mixtral_routewtd': '#8c564b',
    'fp16': '#7f7f7f',
    'baseline': '#1f77b4',
    'routewtd': '#ff7f0e',
}

MODELS = {
    'qwen2_moe': {
        'name': 'Qwen1.5-MoE-A2.7B',
        'short': 'Qwen-MoE',
        'num_layers': 24,
        'num_experts': 60,
        'num_shared': 1,
        'topk': 4,
        'prefix': '',
        'fp16_ppl': 6.790,
    },
    'deepseek_moe': {
        'name': 'DeepSeek-MoE-16B',
        'short': 'DeepSeek-MoE',
        'num_layers': 28,
        'num_experts': 64,
        'num_shared': 2,
        'topk': 6,
        'prefix': 'deepseek_moe_',
        'fp16_ppl': 6.116,
    },
    'mixtral': {
        'name': 'Mixtral-8x7B',
        'short': 'Mixtral',
        'num_layers': 32,
        'num_experts': 8,
        'num_shared': 0,
        'topk': 2,
        'prefix': 'mixtral_',
    },
}

BUDGETS = [2.5, 2.75, 3.0, 3.25]


def load_ppl(model_id: str, method: str, budget: float) -> Optional[float]:
    cfg = MODELS[model_id]
    prefix = cfg['prefix']
    if method == 'baseline':
        fname = f"{prefix}baseline_{budget}b_ppl.json" if prefix else f"baseline_{budget}b_ppl.json"
    else:
        fname = f"{prefix}routewtd_{budget}b_ppl.json" if prefix else f"routewtd_{budget}b_ppl.json"
    path = RESULTS / fname
    if path.exists():
        d = json.load(open(path))
        ppl = d.get('ppl')
        if ppl is not None and not (isinstance(ppl, float) and np.isnan(ppl)):
            return ppl
    return None


def load_tasks(model_id: str, method: str, budget: float) -> Optional[dict]:
    cfg = MODELS[model_id]
    prefix = cfg['prefix']
    if method == 'baseline':
        fname = f"{prefix}baseline_{budget}b_tasks.json" if prefix else f"baseline_{budget}b_tasks.json"
    else:
        fname = f"{prefix}routewtd_{budget}b_tasks.json" if prefix else f"routewtd_{budget}b_tasks.json"
    path = RESULTS / fname
    if path.exists():
        return json.load(open(path))
    return None


def load_gate(model_id: str) -> dict:
    path = GATE / model_id / "wiki2" / "4096" / "moe-gate.json"
    if path.exists():
        return json.load(open(path))
    return {}


def load_corouting(model_id: str) -> dict:
    path = GATE / model_id / "wiki2" / "4096" / "corouting.json"
    if path.exists():
        return json.load(open(path))
    return {}


def load_qconfig(model_id: str, method: str, budget: float) -> Optional[dict]:
    prop = "_prop1" if method == 'routewtd' else ""
    fname = f"{model_id}_rtn_Slayer_bs512_wbits{budget}_r1.0{prop}.json"
    path = QCONFIGS / fname
    if path.exists():
        return json.load(open(path))
    return None


def get_expert_bits(qconfig: dict, layer_idx: int, expert_idx: int) -> float:
    """Get average bit width for an expert across 3 blocks."""
    layer = qconfig[str(layer_idx)]
    if 'experts' in layer:
        expert_data = layer['experts'].get(str(expert_idx))
    else:
        expert_data = layer.get(str(expert_idx))
    if expert_data is None:
        return 0
    bits = []
    for block in ['gate', 'up', 'down']:
        w = expert_data[block]
        # 2-bit with group 128 = 2.25 bits, 4-bit with group -1 = 4 bits
        if w['w_bits'] == 2 and w['w_gsize'] == 128:
            bits.append(2.25)
        else:
            bits.append(float(w['w_bits']))
    return np.mean(bits)


# ===== Figure 1: PPL vs Bit Budget =====

def _load_baseline_ppl(model_id: str, method: str, wbits: int) -> Optional[float]:
    path = RESULTS / f"{model_id}_{method}_w{wbits}_ppl.json"
    if path.exists():
        d = json.load(open(path))
        ppl = d.get('ppl')
        if ppl is not None and ppl < 1e6:
            return ppl
    return None


def _load_baseline_tasks(model_id: str, method: str, wbits: int) -> Optional[dict]:
    path = RESULTS / f"{model_id}_{method}_w{wbits}_tasks.json"
    if path.exists():
        return json.load(open(path))
    return None


def fig_ppl_vs_budget():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    baseline_styles = {
        'RTN W4': {'method': 'rtn', 'wbits': 4, 'color': '#e377c2', 'ls': ':', 'lw': 1.5},
        'GPTQ W4': {'method': 'gptq', 'wbits': 4, 'color': '#17becf', 'ls': ':', 'lw': 1.5},
        'RTN W3': {'method': 'rtn', 'wbits': 3, 'color': '#bcbd22', 'ls': ':', 'lw': 1.5},
        'GPTQ W3': {'method': 'gptq', 'wbits': 3, 'color': '#aec7e8', 'ls': ':', 'lw': 1.5},
    }
    # Bit-width mapping for x-axis reference
    baseline_bits = {'RTN W4': 4.0, 'GPTQ W4': 4.0, 'RTN W3': 3.0, 'GPTQ W3': 3.0}

    for idx, (model_id, cfg) in enumerate([('qwen2_moe', MODELS['qwen2_moe']),
                                            ('deepseek_moe', MODELS['deepseek_moe'])]):
        ax = axes[idx]
        bl_ppls, rw_ppls = [], []
        bl_valid, rw_valid = [], []

        for b in BUDGETS:
            bl = load_ppl(model_id, 'baseline', b)
            rw = load_ppl(model_id, 'routewtd', b)
            bl_ppls.append(bl)
            rw_ppls.append(rw)
            if bl is not None: bl_valid.append((b, bl))
            if rw is not None: rw_valid.append((b, rw))

        # FP16 baseline
        fp16 = cfg.get('fp16_ppl')
        if fp16:
            ax.axhline(y=fp16, color=COLORS['fp16'], linestyle='--', linewidth=1.5, label='FP16', alpha=0.7)

        # Baseline methods (GPTQ, RTN) as horizontal lines
        for bname, bstyle in baseline_styles.items():
            bppl = _load_baseline_ppl(model_id, bstyle['method'], bstyle['wbits'])
            if bppl is not None:
                ax.axhline(y=bppl, color=bstyle['color'], linestyle=bstyle['ls'],
                           linewidth=bstyle['lw'], label=f'{bname} ({bppl:.2f})', alpha=0.7)

        if bl_valid:
            xs, ys = zip(*bl_valid)
            ax.plot(xs, ys, 'o-', color=COLORS[f'{model_id}_baseline'],
                    label='MxMoE Baseline', linewidth=2, markersize=8)
        if rw_valid:
            xs, ys = zip(*rw_valid)
            ax.plot(xs, ys, 's--', color=COLORS[f'{model_id}_routewtd'],
                    label='Route-Weighted', linewidth=2, markersize=8)

        ax.set_xlabel('Average Bit Width')
        ax.set_ylabel('WikiText2 PPL')
        ax.set_title(cfg['name'])
        ax.set_xticks(BUDGETS)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES / "fig1_ppl_vs_budget.pdf")
    plt.savefig(FIGURES / "fig1_ppl_vs_budget.png")
    plt.close()
    print("[OK] Fig 1: PPL vs Bit Budget (with baselines)")


# ===== Figure 2: RHI per Layer =====

def fig_rhi_per_layer():
    fig, ax = plt.subplots(figsize=(10, 4))

    for model_id, cfg in MODELS.items():
        corouting = load_corouting(model_id)
        rhi = corouting.get('route_heterogeneity_index', [])
        if rhi:
            layers = list(range(len(rhi)))
            ax.plot(layers, rhi, 'o-', label=f"{cfg['short']} (mean={np.mean(rhi):.2f})",
                    linewidth=1.5, markersize=4, alpha=0.8)

    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Route Heterogeneity Index (RHI)')
    ax.set_title('Route Heterogeneity Index Across Layers')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "fig2_rhi_per_layer.pdf")
    plt.savefig(FIGURES / "fig2_rhi_per_layer.png")
    plt.close()
    print("[OK] Fig 2: RHI per Layer")


# ===== Figure 3: Expert Access Frequency Heatmap =====

def fig_access_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (model_id, cfg) in enumerate([('qwen2_moe', MODELS['qwen2_moe']),
                                            ('deepseek_moe', MODELS['deepseek_moe'])]):
        ax = axes[idx]
        gate = load_gate(model_id)
        if not gate:
            continue

        num_layers = cfg['num_layers']
        num_experts = cfg['num_experts'] + cfg['num_shared']
        access_freq = np.zeros((num_layers, num_experts))

        for l in range(num_layers):
            layer_data = gate.get(f'layer-{l}', {})
            freq = layer_data.get('access_freq', [])
            for e in range(min(len(freq), num_experts)):
                access_freq[l, e] = freq[e]

        # Normalize per layer
        row_sums = access_freq.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1)
        access_prob = access_freq / row_sums

        im = ax.imshow(access_prob.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Expert')
        ax.set_title(f"{cfg['short']} Expert Access Probability")

        # Mark shared experts
        if cfg['num_shared'] > 0:
            for se in range(cfg['num_experts'], num_experts):
                ax.axhline(y=se - 0.5, color='blue', linestyle='--', linewidth=1, alpha=0.5)

        plt.colorbar(im, ax=ax, label='Access Probability', shrink=0.8)

    plt.tight_layout()
    plt.savefig(FIGURES / "fig3_access_heatmap.pdf")
    plt.savefig(FIGURES / "fig3_access_heatmap.png")
    plt.close()
    print("[OK] Fig 3: Expert Access Heatmap")


# ===== Figure 4: Bit Allocation Heatmap (2.5b budget) =====

def fig_allocation_heatmap(budget=2.5):
    """Show baseline vs route-weighted allocation difference."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    models = [
        ('qwen2_moe', MODELS['qwen2_moe']),
        ('deepseek_moe', MODELS['deepseek_moe']),
    ]

    for col, (model_id, cfg) in enumerate(models):
        bl_cfg = load_qconfig(model_id, 'baseline', budget)
        rw_cfg = load_qconfig(model_id, 'routewtd', budget)

        if bl_cfg is None or rw_cfg is None:
            continue

        int_keys = sorted([k for k in bl_cfg.keys() if k.isdigit()], key=int)
        num_layers = len(int_keys)
        # Find max experts across layers
        max_experts = 0
        for lk in int_keys:
            layer = bl_cfg[lk]
            if 'experts' in layer:
                max_experts = max(max_experts, len(layer['experts']))
            else:
                max_experts = max(max_experts, len(layer))

        bl_mat = np.full((num_layers, max_experts), np.nan)
        rw_mat = np.full((num_layers, max_experts), np.nan)

        for li, lk in enumerate(int_keys):
            layer = bl_cfg[lk]
            if 'experts' in layer:
                for e_str in layer['experts']:
                    e = int(e_str)
                    bl_mat[li, e] = get_expert_bits(bl_cfg, int(lk), e)
                    rw_mat[li, e] = get_expert_bits(rw_cfg, int(lk), e)

        # Baseline
        ax = axes[0, col]
        im = ax.imshow(bl_mat.T, aspect='auto', cmap='RdYlGn', vmin=2.0, vmax=4.5,
                       interpolation='nearest')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Expert')
        ax.set_title(f"{cfg['short']} Baseline ({budget}b)")
        plt.colorbar(im, ax=ax, label='Avg Bit Width', shrink=0.8)

        # Route-weighted
        ax = axes[1, col]
        im = ax.imshow(rw_mat.T, aspect='auto', cmap='RdYlGn', vmin=2.0, vmax=4.5,
                       interpolation='nearest')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Expert')
        ax.set_title(f"{cfg['short']} Route-Weighted ({budget}b)")
        plt.colorbar(im, ax=ax, label='Avg Bit Width', shrink=0.8)

    plt.tight_layout()
    plt.savefig(FIGURES / f"fig4_allocation_{budget}b.pdf")
    plt.savefig(FIGURES / f"fig4_allocation_{budget}b.png")
    plt.close()
    print(f"[OK] Fig 4: Bit Allocation Heatmap ({budget}b)")


# ===== Figure 5: Allocation Difference Heatmap =====

def fig_allocation_diff(budget=2.5):
    """Show the difference between route-weighted and baseline allocations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for col, (model_id, cfg) in enumerate([('qwen2_moe', MODELS['qwen2_moe']),
                                            ('deepseek_moe', MODELS['deepseek_moe'])]):
        ax = axes[col]
        bl_cfg = load_qconfig(model_id, 'baseline', budget)
        rw_cfg = load_qconfig(model_id, 'routewtd', budget)

        if bl_cfg is None or rw_cfg is None:
            continue

        int_keys = sorted([k for k in bl_cfg.keys() if k.isdigit()], key=int)
        num_layers = len(int_keys)
        max_experts = 0
        for lk in int_keys:
            layer = bl_cfg[lk]
            if 'experts' in layer:
                max_experts = max(max_experts, len(layer['experts']))
            else:
                max_experts = max(max_experts, len(layer))

        diff_mat = np.full((num_layers, max_experts), np.nan)

        for li, lk in enumerate(int_keys):
            layer = bl_cfg[lk]
            if 'experts' in layer:
                for e_str in layer['experts']:
                    e = int(e_str)
                    bl_bits = get_expert_bits(bl_cfg, int(lk), e)
                    rw_bits = get_expert_bits(rw_cfg, int(lk), e)
                    diff_mat[li, e] = rw_bits - bl_bits

        im = ax.imshow(diff_mat.T, aspect='auto', cmap='RdBu', vmin=-2, vmax=2,
                       interpolation='nearest')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Expert')
        ax.set_title(f"{cfg['short']} Allocation Change (RW - Baseline)")
        plt.colorbar(im, ax=ax, label='Bit Width Change', shrink=0.8)

    plt.tight_layout()
    plt.savefig(FIGURES / f"fig5_allocation_diff_{budget}b.pdf")
    plt.savefig(FIGURES / f"fig5_allocation_diff_{budget}b.png")
    plt.close()
    print(f"[OK] Fig 5: Allocation Difference ({budget}b)")


# ===== Figure 6: Downstream Task Comparison =====

def fig_downstream_tasks(budget=2.5):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    task_names = ['piqa', 'hellaswag', 'arc_easy', 'arc_challenge', 'winogrande',
                  'lambada_openai', 'lambada_standard']
    task_labels = ['PIQA', 'HellaSwag', 'ARC-E', 'ARC-C', 'WinoGrande', 'Lambada\nOpenAI', 'Lambada\nStd']

    # Include uniform baselines (RTN, GPTQ) as reference bars
    baseline_methods = [
        ('RTN W3', 'rtn', 3, '#bcbd22'),
        ('RTN W4', 'rtn', 4, '#e377c2'),
        ('GPTQ W3', 'gptq', 3, '#aec7e8'),
        ('GPTQ W4', 'gptq', 4, '#17becf'),
    ]

    for col, (model_id, cfg) in enumerate([('qwen2_moe', MODELS['qwen2_moe']),
                                            ('deepseek_moe', MODELS['deepseek_moe'])]):
        ax = axes[col]
        bl_tasks = load_tasks(model_id, 'baseline', budget)
        rw_tasks = load_tasks(model_id, 'routewtd', budget)

        if bl_tasks is None or rw_tasks is None:
            ax.set_title(f"{cfg['short']} — No data")
            continue

        bl_vals = [bl_tasks.get(t, 0) for t in task_names]
        rw_vals = [rw_tasks.get(t, 0) for t in task_names]

        x = np.arange(len(task_names))
        w = 0.18
        n_bars = 2 + sum(1 for _, m, b, _ in baseline_methods if _load_baseline_tasks(model_id, m, b))
        offset = -(n_bars - 1) * w / 2

        color_bl = COLORS[f'{model_id}_baseline']
        color_rw = COLORS[f'{model_id}_routewtd']

        ax.bar(x + offset, bl_vals, w, label='MxMoE Baseline', color=color_bl, alpha=0.8)
        offset += w
        ax.bar(x + offset, rw_vals, w, label='Route-Weighted', color=color_rw, alpha=0.8)
        offset += w

        for bname, bmethod, bwbits, bcolor in baseline_methods:
            btasks = _load_baseline_tasks(model_id, bmethod, bwbits)
            if btasks is not None:
                bvals = [btasks.get(t, 0) for t in task_names]
                ax.bar(x + offset, bvals, w, label=bname, color=bcolor, alpha=0.6)
                offset += w

        bars1 = ax.bar(x - w/2, bl_vals, w, label='Baseline', color=color_bl, alpha=0.8)
        bars2 = ax.bar(x + w/2, rw_vals, w, label='Route-Weighted', color=color_rw, alpha=0.8)

        ax.set_ylabel('Accuracy')
        ax.set_title(f"{cfg['short']} Downstream Tasks ({budget}b)")
        ax.set_xticks(x)
        ax.set_xticklabels(task_labels, fontsize=10)
        ax.legend()
        ax.set_ylim(0.3, 0.8)
        ax.grid(True, alpha=0.2, axis='y')

        # Add delta labels
        for i in range(len(task_names)):
            delta = rw_vals[i] - bl_vals[i]
            if abs(delta) > 0.001:
                color = 'green' if delta > 0 else 'red'
                ax.annotate(f'{delta:+.2f}', xy=(x[i] + w/2, rw_vals[i]),
                           xytext=(0, 5), textcoords='offset points',
                           ha='center', fontsize=8, color=color)

    plt.tight_layout()
    plt.savefig(FIGURES / f"fig6_downstream_{budget}b.pdf")
    plt.savefig(FIGURES / f"fig6_downstream_{budget}b.png")
    plt.close()
    print(f"[OK] Fig 6: Downstream Tasks ({budget}b)")


# ===== Figure 7: RHI vs PPL Improvement Correlation =====

def fig_rhi_vs_improvement():
    fig, ax = plt.subplots(figsize=(6, 5))

    points = []
    for model_id, cfg in MODELS.items():
        corouting = load_corouting(model_id)
        rhi = corouting.get('route_heterogeneity_index', [])
        if not rhi:
            continue
        mean_rhi = np.mean(rhi)

        for budget in BUDGETS:
            bl = load_ppl(model_id, 'baseline', budget)
            rw = load_ppl(model_id, 'routewtd', budget)
            if bl and rw:
                improvement = (bl - rw) / bl * 100  # positive = route-weighted better
                points.append((mean_rhi, improvement, cfg['short'], budget))

    if not points:
        print("[SKIP] Fig 7: No data for RHI vs improvement")
        return

    rhis, imps, labels, budgets = zip(*points)

    # Color by model
    model_colors = {'Qwen-MoE': '#1f77b4', 'DeepSeek-MoE': '#2ca02c', 'Mixtral': '#9467bd'}
    colors = [model_colors.get(l, '#333') for l in labels]

    sizes = [40 + 30 * (4.0 - b) for b in budgets]  # lower budget = larger point

    scatter = ax.scatter(rhis, imps, c=colors, s=sizes, alpha=0.8, edgecolors='black', linewidth=0.5)

    # Labels
    for r, imp, label, b in points:
        ax.annotate(f'{b}b', (r, imp), textcoords='offset points',
                   xytext=(5, 5), fontsize=8, alpha=0.7)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Mean Route Heterogeneity Index (RHI)')
    ax.set_ylabel('PPL Improvement (%)')
    ax.set_title('RHI vs Route-Weighting Benefit')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                              markersize=10, label=n)
                      for n, c in model_colors.items()]
    ax.legend(handles=legend_elements)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES / "fig7_rhi_vs_improvement.pdf")
    plt.savefig(FIGURES / "fig7_rhi_vs_improvement.png")
    plt.close()
    print("[OK] Fig 7: RHI vs PPL Improvement")


# ===== Figure 8: Per-layer PPL Degradation =====

def fig_per_layer_sensitivity():
    """Plot per-layer sensitivity from calibration data."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for col, (model_id, cfg) in enumerate([('qwen2_moe', MODELS['qwen2_moe']),
                                            ('deepseek_moe', MODELS['deepseek_moe'])]):
        ax = axes[col]
        for method, label, color in [('W2A16_g128_asym', 'W2.25A16', '#d62728'),
                                     ('W4A16_g-1_asym', 'W4A16', '#2ca02c')]:
            calib_file = BASE / "calib" / f"{model_id}-MOE-rtn-{method}-wiki2-128-4096-layer_out_norm.json"
            if not calib_file.exists():
                continue
            data = json.load(open(calib_file))
            # Each entry is layer -> expert -> block -> norm_diff
            layers = sorted([int(k) for k in data.keys()])
            # Average sensitivity per layer across all experts and blocks
            layer_sens = []
            for l in layers:
                layer_data = data[str(l)]
                vals = []
                for e_data in (layer_data.get('experts', {}).values() if 'experts' in layer_data else [layer_data]):
                    if isinstance(e_data, dict):
                        for b_data in e_data.values():
                            if isinstance(b_data, (int, float)):
                                vals.append(b_data)
                if vals:
                    layer_sens.append(np.mean(vals))
                else:
                    layer_sens.append(0)

            ax.plot(range(len(layer_sens)), layer_sens, 'o-', label=label,
                   color=color, linewidth=1.5, markersize=4)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Avg Sensitivity (Layer Output Norm)')
        ax.set_title(cfg['short'])
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES / "fig8_per_layer_sensitivity.pdf")
    plt.savefig(FIGURES / "fig8_per_layer_sensitivity.png")
    plt.close()
    print("[OK] Fig 8: Per-Layer Sensitivity")


# ===== Figure 9: Expert Heterogeneity Box Plot =====

def fig_expert_heterogeneity():
    """Show distribution of expert access frequency per layer."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for col, (model_id, cfg) in enumerate([('qwen2_moe', MODELS['qwen2_moe']),
                                            ('deepseek_moe', MODELS['deepseek_moe'])]):
        ax = axes[col]
        gate = load_gate(model_id)
        if not gate:
            continue

        num_layers = cfg['num_layers']
        num_experts = cfg['num_experts']

        # Collect per-layer expert access distributions (exclude shared)
        box_data = []
        for l in range(num_layers):
            freq = gate.get(f'layer-{l}', {}).get('access_freq', [])
            routed_freq = freq[:num_experts]  # exclude shared experts
            if routed_freq:
                total = sum(routed_freq)
                probs = [f / total for f in routed_freq] if total > 0 else [1/num_experts] * num_experts
                box_data.append(probs)

        if box_data:
            bp = ax.boxplot(box_data, positions=range(num_layers), widths=0.6,
                           showfliers=False, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('#aec7e8')
                patch.set_alpha(0.6)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Expert Access Probability')
        ax.set_title(f"{cfg['short']} Expert Access Distribution")
        ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES / "fig9_expert_heterogeneity.pdf")
    plt.savefig(FIGURES / "fig9_expert_heterogeneity.png")
    plt.close()
    print("[OK] Fig 9: Expert Heterogeneity Box Plot")


# ===== Figure 10: Summary Performance Table (as figure) =====

def fig_summary_table():
    """Generate a comprehensive summary table with all methods."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    for ax in axes:
        ax.axis('off')

    for col, (model_id, cfg) in enumerate([('qwen2_moe', MODELS['qwen2_moe']),
                                            ('deepseek_moe', MODELS['deepseek_moe'])]):
        ax = axes[col]
        fp16 = cfg.get('fp16_ppl', 0)
        rows = []

        # FP16 row
        rows.append(['FP16', f'{fp16:.3f}', '—', '—', '—'])

        # Uniform baselines (RTN, GPTQ)
        for method, label in [('rtn', 'RTN'), ('gptq', 'GPTQ')]:
            for wbits in [3, 4]:
                ppl = _load_baseline_ppl(model_id, method, wbits)
                tasks = _load_baseline_tasks(model_id, method, wbits)
                avg = tasks.get('acc_avg', tasks.get('zero_shot', {}).get('acc_avg', 0)) if tasks else 0
                bits_str = f'{wbits}.0'
                rows.append([f'{label} W{wbits}', f'{ppl:.3f}' if ppl else '—',
                             f'{avg:.4f}' if avg else '—', bits_str, 'Uniform'])

        # MxMoE methods
        for method, label in [('baseline', 'MxMoE'), ('routewtd', 'MxMoE-RW')]:
            for budget in BUDGETS:
                ppl = load_ppl(model_id, method, budget)
                tasks_d = load_tasks(model_id, method, budget)
                avg = tasks_d.get('acc_avg', tasks_d.get('zero_shot', {}).get('acc_avg', 0)) if tasks_d else 0
                rows.append([f'{label} {budget}b', f'{ppl:.3f}' if ppl else '—',
                             f'{avg:.4f}' if avg else '—', f'{budget}', 'Mixed'])

        headers = [f'{cfg["short"]}', 'PPL', 'Avg Acc', 'Bits', 'Type']
        table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.3)

        for j in range(len(headers)):
            table[0, j].set_facecolor('#404040')
            table[0, j].set_text_props(color='white', fontweight='bold')

        # Highlight best PPL (excluding FP16)
        ppl_vals = []
        for r in rows[1:]:
            try:
                ppl_vals.append(float(r[1]))
            except:
                ppl_vals.append(999)
        if ppl_vals:
            best_idx = ppl_vals.index(min(ppl_vals)) + 1
            table[best_idx, 1].set_facecolor('#c8e6c9')

    plt.tight_layout()
    plt.savefig(FIGURES / "fig10_summary_table.pdf")
    plt.savefig(FIGURES / "fig10_summary_table.png")
    plt.close()
    print("[OK] Fig 10: Summary Table (with baselines)")


# ===== Figure 11: Shared Expert Analysis =====

def fig_shared_expert_analysis(budget=2.5):
    """Compare allocation for shared vs routed experts."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for col, (model_id, cfg) in enumerate([('qwen2_moe', MODELS['qwen2_moe']),
                                            ('deepseek_moe', MODELS['deepseek_moe'])]):
        ax = axes[col]
        if cfg['num_shared'] == 0:
            continue

        bl_cfg = load_qconfig(model_id, 'baseline', budget)
        rw_cfg = load_qconfig(model_id, 'routewtd', budget)
        gate = load_gate(model_id)

        if not bl_cfg or not rw_cfg or not gate:
            continue

        num_layers = cfg['num_layers']
        shared_bits_bl, shared_bits_rw = [], []
        routed_bits_bl, routed_bits_rw = [], []

        for l in range(num_layers):
            # Shared experts
            for se in range(cfg['num_experts'], cfg['num_experts'] + cfg['num_shared']):
                bl_b = get_expert_bits(bl_cfg, l, se)
                rw_b = get_expert_bits(rw_cfg, l, se)
                shared_bits_bl.append(bl_b)
                shared_bits_rw.append(rw_b)

            # Routed experts (sample some)
            for e in range(min(10, cfg['num_experts'])):
                bl_b = get_expert_bits(bl_cfg, l, e)
                rw_b = get_expert_bits(rw_cfg, l, e)
                routed_bits_bl.append(bl_b)
                routed_bits_rw.append(rw_b)

        data_to_plot = [
            routed_bits_bl, routed_bits_rw,
            shared_bits_bl, shared_bits_rw,
        ]
        labels = ['Routed\nBaseline', 'Routed\nRoute-Wtd', 'Shared\nBaseline', 'Shared\nRoute-Wtd']

        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, showfliers=False)
        colors = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_ylabel('Average Bit Width')
        ax.set_title(f"{cfg['short']} ({budget}b)")
        ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES / f"fig11_shared_expert_{budget}b.pdf")
    plt.savefig(FIGURES / f"fig11_shared_expert_{budget}b.png")
    plt.close()
    print(f"[OK] Fig 11: Shared Expert Analysis ({budget}b)")


# ===== Figure 12: Cross-Model Comparison Bar Chart =====

def fig_cross_model_comparison(budget=2.5):
    """Bar chart comparing PPL degradation across models."""
    fig, ax = plt.subplots(figsize=(8, 5))

    models_data = []
    for model_id, cfg in MODELS.items():
        fp16 = cfg.get('fp16_ppl')
        bl = load_ppl(model_id, 'baseline', budget)
        rw = load_ppl(model_id, 'routewtd', budget)
        corouting = load_corouting(model_id)
        rhi = np.mean(corouting['route_heterogeneity_index']) if corouting.get('route_heterogeneity_index') else 0

        if bl and rw and fp16:
            bl_deg = (bl - fp16) / fp16 * 100
            rw_deg = (rw - fp16) / fp16 * 100
            improvement = bl_deg - rw_deg  # positive = route-weighted reduces degradation
            models_data.append((cfg['short'], rhi, bl_deg, rw_deg, improvement))

    if not models_data:
        print("[SKIP] Fig 12: Not enough data")
        return

    x = np.arange(len(models_data))
    w = 0.3

    for i, (name, rhi, bl_deg, rw_deg, imp) in enumerate(models_data):
        ax.bar(i - w/2, bl_deg, w, color='#1f77b4', alpha=0.8, label='Baseline' if i == 0 else '')
        ax.bar(i + w/2, rw_deg, w, color='#ff7f0e', alpha=0.8, label='Route-Weighted' if i == 0 else '')
        ax.annotate(f'RHI={rhi:.2f}\nΔ={imp:+.2f}pp', xy=(i, max(bl_deg, rw_deg)),
                   xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([d[0] for d in models_data])
    ax.set_ylabel('PPL Degradation vs FP16 (%)')
    ax.set_title(f'Cross-Model Comparison ({budget}b Budget)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES / f"fig12_cross_model_{budget}b.pdf")
    plt.savefig(FIGURES / f"fig12_cross_model_{budget}b.png")
    plt.close()
    print(f"[OK] Fig 12: Cross-Model Comparison ({budget}b)")


# ===== Figure 13: Co-Routing Transition Matrix =====

def fig_transition_matrix(model_id='qwen2_moe', layer_idx=10):
    """Visualize a single co-routing transition matrix."""
    corouting = load_corouting(model_id)
    tms = corouting.get('transition_matrices', [])
    if not tms or layer_idx >= len(tms):
        print(f"[SKIP] No transition matrix for {model_id} layer {layer_idx}")
        return

    P = np.array(tms[layer_idx])
    cfg = MODELS[model_id]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(P, cmap='hot', interpolation='nearest', aspect='auto')
    ax.set_xlabel(f'Expert at Layer {layer_idx + 1}')
    ax.set_ylabel(f'Expert at Layer {layer_idx}')
    ax.set_title(f'{cfg["short"]} Expert Transition Matrix (Layer {layer_idx})')
    plt.colorbar(im, ax=ax, label='Transition Probability')

    plt.tight_layout()
    plt.savefig(FIGURES / f"fig13_transition_{model_id}_layer{layer_idx}.pdf")
    plt.savefig(FIGURES / f"fig13_transition_{model_id}_layer{layer_idx}.png")
    plt.close()
    print(f"[OK] Fig 13: Transition Matrix ({model_id}, layer {layer_idx})")


# ===== Figure 14: Baseline Comparison (PPL + Tasks grouped bar) =====

def fig_baseline_comparison():
    """Grouped bar chart comparing MxMoE vs RTN/GPTQ at similar bit widths."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for col, (model_id, cfg) in enumerate([('qwen2_moe', MODELS['qwen2_moe']),
                                            ('deepseek_moe', MODELS['deepseek_moe'])]):
        ax = axes[col]
        fp16 = cfg.get('fp16_ppl', 0)

        methods = []
        ppls = []
        accs = []
        colors_list = []

        # FP16
        fp16_tasks = _load_baseline_tasks(model_id, 'rtn', 16)
        # (no FP16 tasks file, skip)
        methods.append('FP16')
        ppls.append(fp16)
        accs.append(None)
        colors_list.append('#7f7f7f')

        # RTN W3, GPTQ W3
        for mname, mmethod in [('RTN W3', 'rtn'), ('GPTQ W3', 'gptq')]:
            p = _load_baseline_ppl(model_id, mmethod, 3)
            t = _load_baseline_tasks(model_id, mmethod, 3)
            a = t.get('acc_avg', t.get('zero_shot', {}).get('acc_avg', 0)) if t else 0
            methods.append(mname)
            ppls.append(p if p else 0)
            accs.append(a)
            colors_list.append('#bcbd22' if 'RTN' in mname else '#aec7e8')

        # MxMoE 2.5b, 2.75b, 3.0b, 3.25b (baseline)
        for budget in BUDGETS:
            p = load_ppl(model_id, 'baseline', budget)
            t = load_tasks(model_id, 'baseline', budget)
            a = t.get('acc_avg', t.get('zero_shot', {}).get('acc_avg', 0)) if t else 0
            methods.append(f'MxMoE {budget}b')
            ppls.append(p if p else 0)
            accs.append(a)
            colors_list.append(COLORS[f'{model_id}_baseline'])

        # RTN W4, GPTQ W4
        for mname, mmethod in [('RTN W4', 'rtn'), ('GPTQ W4', 'gptq')]:
            p = _load_baseline_ppl(model_id, mmethod, 4)
            t = _load_baseline_tasks(model_id, mmethod, 4)
            a = t.get('acc_avg', t.get('zero_shot', {}).get('acc_avg', 0)) if t else 0
            methods.append(mname)
            ppls.append(p if p else 0)
            accs.append(a)
            colors_list.append('#e377c2' if 'RTN' in mname else '#17becf')

        x = np.arange(len(methods))
        valid_ppls = [p if p and p < 1e6 else 0 for p in ppls]

        bars = ax.bar(x, valid_ppls, color=colors_list, alpha=0.85, edgecolor='black', linewidth=0.5)

        # Add value labels
        for i, (bar, p) in enumerate(zip(bars, valid_ppls)):
            if p > 0 and p < 1e6:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{p:.2f}', ha='center', va='bottom', fontsize=7, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('WikiText2 PPL')
        ax.set_title(cfg['name'])
        ax.grid(True, alpha=0.2, axis='y')
        if fp16:
            ax.axhline(y=fp16, color='#7f7f7f', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIGURES / "fig14_baseline_comparison.pdf")
    plt.savefig(FIGURES / "fig14_baseline_comparison.png")
    plt.close()
    print("[OK] Fig 14: Baseline Comparison")


if __name__ == '__main__':
    print("=" * 60)
    print("Paper Figure Generation")
    print("=" * 60)

    # Generate all figures
    fig_ppl_vs_budget()
    fig_rhi_per_layer()
    fig_access_heatmap()
    fig_allocation_heatmap(budget=2.5)
    fig_allocation_diff(budget=2.5)
    fig_downstream_tasks(budget=2.5)
    fig_downstream_tasks(budget=2.75)
    fig_rhi_vs_improvement()
    fig_per_layer_sensitivity()
    fig_expert_heterogeneity()
    fig_summary_table()
    fig_shared_expert_analysis(budget=2.5)
    fig_cross_model_comparison(budget=2.5)
    fig_transition_matrix('qwen2_moe', layer_idx=10)
    fig_transition_matrix('deepseek_moe', layer_idx=10)
    fig_baseline_comparison()

    print("\n" + "=" * 60)
    print(f"All figures saved to {FIGURES}")
    print("=" * 60)
