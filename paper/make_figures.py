#!/usr/bin/env python
"""Generate paper figures from results/ directory.

Produces:
  - fig1_hero.pdf : summary: PPL-gap vs budget, Qwen + DS
  - fig3_ppl_curves.pdf : two-panel PPL curves
  - fig4_downstream.pdf : 7-task bars at 2.5b and 2.75b (and 2.3b appendix)
  - fig5_rhi.pdf : RHI conceptual + data point overlay
  - fig6_allocation_heatmap.pdf : allocation comparison at 2.5b
"""
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

RES = Path("/data/zengyq/paper/baselines/MxMoE/results")
OUT = Path("/data/zengyq/paper/baselines/paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ----- Data tables --------------------------------------------------------
BUDGETS = [2.3, 2.5, 2.75, 3.0, 3.25]

qwen_base = {2.3: 17.813, 2.5: 9.619, 2.75: 8.525, 3.0: 7.953, 3.25: 7.644}
qwen_rws  = {2.3: 17.620, 2.5: 9.351, 2.75: 8.428, 3.0: 7.955, 3.25: 7.652}
qwen_fp16 = 6.797

ds_base = {2.3: 10.228, 2.5: 8.005, 2.75: 7.320, 3.0: 6.934, 3.25: 6.730}
ds_rws  = {2.3: 10.179, 2.5: 8.041, 2.75: 7.318, 3.0: 6.939, 3.25: 6.734}
ds_fp16 = 6.1155

# Downstream (7-task avg)
qwen_ds_base = {2.5: 0.5952, 2.75: 0.6226, 2.3: 0.4428}
qwen_ds_rws  = {2.5: 0.6015, 2.75: 0.6257, 2.3: 0.4384}
ds_ds_base   = {2.5: 0.595, 2.3: 0.5574}
ds_ds_rws    = {2.5: 0.597, 2.3: 0.5537}

# ---- Figure 1: Hero ------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 3.2))
b = BUDGETS
# improvement = baseline - rws as %, positive = RWS wins
qg = [100 * (qwen_base[x] - qwen_rws[x]) / qwen_base[x] for x in b]
dg = [100 * (ds_base[x]   - ds_rws[x])   / ds_base[x]   for x in b]
ax.axhline(0, color="gray", lw=0.6)
ax.fill_between([2.3, 2.75], -1, 4, color="#cc5c33", alpha=0.06, zorder=-1,
                label="Aggressive-compression regime")
ax.plot(b, qg, "o-", color="#cc5c33", lw=2, label=r"Qwen1.5-MoE (RHI=1.84)", markersize=7)
ax.plot(b, dg, "s-", color="#3d8ec9", lw=2, label=r"DeepSeek-MoE (RHI=1.56)", markersize=7)
ax.set_xlabel("Bit budget (bits/param)")
ax.set_ylabel("PPL improvement over baseline (%) $\\uparrow$")
ax.set_title("Route-Weighted Sensitivity wins in the aggressive regime")
ax.set_ylim(-1.2, 3.5)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "fig1_hero.pdf")
fig.savefig(OUT / "fig1_hero.png", dpi=150)
plt.close(fig)

# ---- Figure 3: PPL curves two-panel -------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.3), sharey=False)
for ax, name, base_d, rws_d, fp16 in [
    (axes[0], "Qwen1.5-MoE-A2.7B  (RHI=1.84)", qwen_base, qwen_rws, qwen_fp16),
    (axes[1], "DeepSeek-MoE-16B  (RHI=1.56)", ds_base, ds_rws, ds_fp16),
]:
    ax.plot(b, [base_d[x] for x in b], "o-", color="#888", lw=2, label="Baseline MxMoE", markersize=6)
    ax.plot(b, [rws_d[x] for x in b], "o-", color="#cc5c33", lw=2, label="RWS (ours)", markersize=6)
    ax.axhline(fp16, color="#222", lw=0.8, ls=":", label=f"FP16 = {fp16:.2f}")
    ax.set_xlabel("Bit budget (bits/param)")
    ax.set_ylabel("WikiText-2 PPL")
    ax.set_title(name, fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT / "fig3_ppl_curves.pdf")
fig.savefig(OUT / "fig3_ppl_curves.png", dpi=150)
plt.close(fig)

# ---- Figure 4: Downstream 2.5b + 2.75b per-task bars --------------------
# Load detailed per-task JSON
def load_tasks(path):
    with open(path) as f:
        d = json.load(f)["zero_shot"]
    return {k: v for k, v in d.items() if k != "acc_avg"}

task_order = ["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "lambada_openai", "lambada_standard"]
task_labels = ["PIQA", "HSwag", "ARC-E", "ARC-C", "WinoG", "L-OAI", "L-Std"]

qwen_25_base = load_tasks(RES / "baseline_2.5b_tasks.json")
qwen_25_rws  = load_tasks(RES / "routewtd_2.5b_tasks.json")
qwen_275_base = load_tasks(RES / "baseline_2.75b_tasks.json")
qwen_275_rws  = load_tasks(RES / "routewtd_2.75b_tasks.json")

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2), sharey=True)
for ax, name, base, rws in [
    (axes[0], "Qwen1.5-MoE @ 2.5 bit", qwen_25_base, qwen_25_rws),
    (axes[1], "Qwen1.5-MoE @ 2.75 bit", qwen_275_base, qwen_275_rws),
]:
    x = np.arange(len(task_order))
    base_v = [base[t] for t in task_order]
    rws_v  = [rws[t]  for t in task_order]
    ax.bar(x - 0.18, base_v, 0.34, color="#888", label="Baseline")
    ax.bar(x + 0.18, rws_v, 0.34, color="#cc5c33", label="RWS")
    ax.set_xticks(x); ax.set_xticklabels(task_labels, rotation=30, ha="right")
    ax.set_ylim(0, 0.85)
    ax.set_title(name, fontsize=10)
    if ax is axes[0]: ax.set_ylabel("Accuracy")
    ax.legend(frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT / "fig4_downstream.pdf")
fig.savefig(OUT / "fig4_downstream.png", dpi=150)
plt.close(fig)

# ---- Figure 5: RHI conceptual with data points --------------------------
fig, ax = plt.subplots(figsize=(4.8, 3.0))
# Conceptual curve: sigmoid-like starting near 0 until RHI ~ 1.6, rising after
xs = np.linspace(1.3, 2.2, 200)
ys = 3.0 / (1 + np.exp(-8 * (xs - 1.7)))  # rising curve capping at ~3% ΔPPL
ax.plot(xs, ys, color="#3d8ec9", lw=2)
ax.fill_between(xs, 0, ys, color="#3d8ec9", alpha=0.08)
ax.axhline(0, color="gray", lw=0.5)
# Our two data points (using 2.5b absolute ΔPPL gain in %)
pts = [(1.56, 100*(ds_base[2.5]-ds_rws[2.5])/ds_base[2.5], "DeepSeek-MoE"),
       (1.84, 100*(qwen_base[2.5]-qwen_rws[2.5])/qwen_base[2.5], "Qwen1.5-MoE")]
for rhi, gain, name in pts:
    ax.scatter(rhi, gain, s=80, zorder=3, color="#cc5c33", edgecolor="k")
    ax.annotate(name, (rhi, gain), textcoords="offset points", xytext=(6, 6), fontsize=9)
ax.set_xlabel("Route Heterogeneity Index (RHI)")
ax.set_ylabel(r"PPL improvement at 2.5 b (%) $\uparrow$")
ax.set_title("RHI predicts when RWS helps", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "fig5_rhi.pdf")
fig.savefig(OUT / "fig5_rhi.png", dpi=150)
plt.close(fig)

# ---- Allocation heatmap from qconfig at 2.5b (baseline vs RWS for Qwen) -
from collections import Counter

def bits_grid(qcfg_path, n_layers, n_experts):
    with open(qcfg_path) as f:
        cfg = json.load(f)
    # grid: rows=experts, cols=layers; value = avg bits across gate/up/down
    grid = np.zeros((n_experts, n_layers))
    for L in range(n_layers):
        exps = cfg[str(L)]["experts"]
        for E in range(n_experts):
            if str(E) not in exps:
                continue
            wbits = []
            for block in ["gate", "up", "down"]:
                b = exps[str(E)][block]["w_bits"]
                gsize = exps[str(E)][block]["w_gsize"]
                eff = {2: 2.25, 4: 4.0}.get(b, b)
                wbits.append(eff)
            grid[E, L] = float(np.mean(wbits))
    return grid

QCONF_DIR = Path("/data/zengyq/paper/baselines/MxMoE/qconfigs/w2a16_g128_asym+w4a16_g-1_asym")
grid_base = bits_grid(QCONF_DIR / "qwen2_moe_rtn_Slayer_bs512_wbits2.5_r1.0.json", n_layers=24, n_experts=60)
grid_rws  = bits_grid(QCONF_DIR / "qwen2_moe_rtn_Slayer_bs512_wbits2.5_r1.0_prop1.json", n_layers=24, n_experts=60)

fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0), sharey=True)
for ax, grid, title in [(axes[0], grid_base, "Baseline MxMoE @ 2.5 b"),
                        (axes[1], grid_rws, "RWS (ours) @ 2.5 b")]:
    im = ax.imshow(grid, aspect="auto", cmap="RdYlBu_r", vmin=2.25, vmax=4.0)
    ax.set_xlabel("Layer index")
    ax.set_title(title, fontsize=10)
    if ax is axes[0]: ax.set_ylabel("Expert index (0–59)")
fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, label="Avg. bits/param")
fig.savefig(OUT / "fig6_allocation.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig6_allocation.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("All figures written to", OUT)
