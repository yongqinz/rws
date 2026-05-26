#!/usr/bin/env python3
"""Generate EMNLP paper figures with 3-model data."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path("/data/zengyq/paper/baselines/emnlp/figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

BUDGETS = [2.3, 2.5, 2.75, 3.0, 3.25]

# Data from JSON result files
qwen_base = {2.3: 17.813, 2.5: 9.619, 2.75: 8.525, 3.0: 7.953, 3.25: 7.644}
qwen_rws  = {2.3: 17.620, 2.5: 9.351, 2.75: 8.428, 3.0: 7.951, 3.25: 7.652}
qwen_fp16 = 6.797

ds_base = {2.3: 10.228, 2.5: 8.005, 2.75: 7.320, 3.0: 6.934, 3.25: 6.730}
ds_rws  = {2.3: 10.179, 2.5: 7.901, 2.75: 7.311, 3.0: 6.928, 3.25: 6.734}
ds_fp16 = 6.116

mix_base = {2.3: 5.617, 2.5: 4.762, 2.75: 4.492, 3.0: 4.358, 3.25: 4.158}
mix_rws  = {2.3: 5.533, 2.5: 4.715, 2.75: 4.460, 3.0: 4.341, 3.25: 4.150}
mix_fp16 = 3.89

# ===== Figure 1: PPL improvement vs budget (3 models) =====
fig, ax = plt.subplots(figsize=(5.0, 3.2))
b = BUDGETS
qg = [100 * (qwen_base[x] - qwen_rws[x]) / qwen_base[x] for x in b]
mg = [100 * (mix_base[x]  - mix_rws[x])  / mix_base[x]  for x in b]
dg = [100 * (ds_base[x]   - ds_rws[x])   / ds_base[x]   for x in b]
ax.axhline(0, color="gray", lw=0.6)
ax.fill_between([2.2, 2.85], -0.5, 3.5, color="#cc5c33", alpha=0.06, zorder=-1)
ax.plot(b, qg, "o-", color="#cc5c33", lw=2, label=r"Qwen1.5-MoE ($\mathrm{RHI}=1.84$)", markersize=7)
ax.plot(b, mg, "D-", color="#2ca02c", lw=2, label=r"Mixtral-8x7B ($\mathrm{RHI}=1.84$)", markersize=6)
ax.plot(b, dg, "s-", color="#3d8ec9", lw=2, label=r"DeepSeek-MoE ($\mathrm{RHI}=1.56$)", markersize=7)
ax.set_xlabel("Bit budget (bits/param)", fontsize=11)
ax.set_ylabel(r"$\Delta$PPL over baseline (\%) $\uparrow$", fontsize=11)
ax.set_ylim(-0.5, 3.5)
ax.set_xticks(BUDGETS)
ax.legend(frameon=False, loc="upper right", fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT / "fig_ppl_improvement.pdf")
fig.savefig(OUT / "fig_ppl_improvement.png", dpi=200)
plt.close(fig)
print("fig_ppl_improvement done")

# ===== Figure 2: RHI diagnostic (3 data points) =====
fig, ax = plt.subplots(figsize=(3.8, 3.2))
rhi_vals = [1.56, 1.84, 1.84]
delta_ppl = [1.30, 2.81, 0.98]  # positive = improvement
colors = ["#3d8ec9", "#cc5c33", "#2ca02c"]
labels = [
    r"DeepSeek-MoE ($\mathrm{RHI}=1.56$)",
    r"Qwen1.5-MoE ($\mathrm{RHI}=1.84$)",
    r"Mixtral-8x7B ($\mathrm{RHI}=1.84$)",
]
offsets = [(8, -4), (8, 4), (8, -8)]
for r, d, c, l, off in zip(rhi_vals, delta_ppl, colors, labels, offsets):
    ax.scatter(r, d, s=100, color=c, zorder=5, label=l)
    ax.annotate(f"{d:.1f}%", (r, d), textcoords="offset points",
                xytext=off, fontsize=9, color=c)
ax.axhline(0, color="gray", lw=0.5, ls="-")
ax.set_xlabel(r"Route Heterogeneity Index ($\mathrm{RHI}$)", fontsize=10)
ax.set_ylabel(r"$\Delta$PPL at 2.5\,bits (\%) $\uparrow$", fontsize=10)
ax.set_xlim(1.4, 2.1)
ax.set_ylim(-0.5, 3.5)
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "fig_rhi.pdf")
fig.savefig(OUT / "fig_rhi.png", dpi=200)
plt.close(fig)
print("fig_rhi done")

# ===== Figure 3: PPL curves three-panel =====
fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), sharey=False)
for ax, name, base_d, rws_d, fp16, color in [
    (axes[0], r"Qwen1.5-MoE ($\mathrm{RHI}=1.84$)", qwen_base, qwen_rws, qwen_fp16, "#cc5c33"),
    (axes[1], r"Mixtral-8x7B ($\mathrm{RHI}=1.84$)", mix_base, mix_rws, mix_fp16, "#2ca02c"),
    (axes[2], r"DeepSeek-MoE ($\mathrm{RHI}=1.56$)", ds_base, ds_rws, ds_fp16, "#3d8ec9"),
]:
    ax.plot(b, [base_d[x] for x in b], "o-", color="#888", lw=1.8, label="Baseline", markersize=5)
    ax.plot(b, [rws_d[x] for x in b], "o-", color=color, lw=1.8, label="RWS (ours)", markersize=5)
    ax.axhline(fp16, color="#222", lw=0.8, ls=":", label=f"FP16 = {fp16:.2f}")
    ax.set_xlabel("Bits/param", fontsize=10)
    ax.set_ylabel("WikiText-2 PPL", fontsize=10)
    ax.set_title(name, fontsize=9)
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_xticks(BUDGETS)
    ax.tick_params(labelsize=8)
fig.tight_layout(w_pad=1.5)
fig.savefig(OUT / "fig_ppl_curves.pdf")
fig.savefig(OUT / "fig_ppl_curves.png", dpi=200)
plt.close(fig)
print("fig_ppl_curves done")

# ===== Figure 4: Allocation heatmap =====
import json
try:
    base_cfg = json.load(open("/data/zengyq/paper/baselines/MxMoE/qconfigs/w2a16_g128_asym+w4a16_g-1_asym/qwen2_moe_rtn_Slayer_bs512_wbits2.5_r1.0.json"))
    rws_cfg  = json.load(open("/data/zengyq/paper/baselines/MxMoE/qconfigs/w2a16_g128_asym+w4a16_g-1_asym/qwen2_moe_rtn_Slayer_bs512_wbits2.5_r1.0_prop1.json"))

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))
    for ax, cfg, title in [(axes[0], base_cfg, "Baseline"), (axes[1], rws_cfg, "RWS (ours)")]:
        layers = sorted([k for k in cfg if k.isdigit()], key=int)
        if not layers:
            break
        n_layers = len(layers)
        n_experts = len(cfg[layers[0]])
        alloc = np.zeros((n_layers, n_experts))
        for i, l in enumerate(layers):
            for j, (eid, strat) in enumerate(sorted(cfg[l].items(), key=lambda x: int(x[0]))):
                alloc[i, j] = 1.0 if "w4" in str(strat).lower() or "W4" in str(strat) else 0.0
        im = ax.imshow(alloc, aspect="auto", cmap="RdYlBu_r", interpolation="nearest")
        ax.set_xlabel("Expert index", fontsize=9)
        ax.set_ylabel("Layer", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)
    fig.colorbar(im, ax=axes, label="4-bit allocation", shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_allocation.pdf")
    fig.savefig(OUT / "fig_allocation.png", dpi=200)
    plt.close(fig)
    print("fig_allocation done")
except Exception as e:
    print(f"fig_allocation skipped: {e}")

print("All figures generated.")
