import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
full = json.load(open(sys.argv[1], encoding="utf-8")); nof = json.load(open(sys.argv[2], encoding="utf-8")); out = sys.argv[3]

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1 and 2 (method = entity)
fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=200, facecolor=SURF)
ax.set_facecolor(SURF)

groups = ["10 play stats\n(without final_* stats)", "All 23 stats\n(plus 13 final_* stats)"]
data = {"kNN": (BLUE, [nof["knn"], full["knn"]]), "Ridge logistic": (ORANGE, [nof["logit"], full["logit"]])}
w, gap = 0.34, 0.04
x = np.array([-0.35, 0.65])
for j, (name, (col, rs)) in enumerate(data.items()):
    pos = x + (j - 0.5) * (w + gap)
    acc = np.array([r["acc"] for r in rs]); lo = np.array([r["boot_lo"] for r in rs]); hi = np.array([r["boot_hi"] for r in rs])
    ax.bar(pos, acc, width=w, color=col, label=name, zorder=3, linewidth=0)
    ax.errorbar(pos, acc, yerr=[acc - lo, hi - acc], fmt="none", ecolor=INK2, elinewidth=1.2, capsize=4, capthick=1.2, zorder=4)
    for p, a, h in zip(pos, acc, hi):
        ax.text(p, h + 0.015, f"{a:.2f}", ha="center", va="bottom", fontsize=10.5, color=INK, fontweight="bold", zorder=5)

# references: shuffled-label band and largest-class guess
sh_lo = min(full["shuffled_knn"]["mean"], nof["shuffled_knn"]["mean"], full["shuffled_logit"]["mean"], nof["shuffled_logit"]["mean"])
sh_hi = max(full["shuffled_knn"]["p95"], nof["shuffled_knn"]["p95"], full["shuffled_logit"]["p95"], nof["shuffled_logit"]["p95"])
floor = full["floor"]["acc"]
ax.axhspan(sh_lo, sh_hi, color="#d9d8d3", alpha=0.55, zorder=1, linewidth=0)
ax.axhline(floor, color=INK2, linestyle=(0, (4, 3)), linewidth=1.3, zorder=2)
ax.text(-1.55, floor + 0.014, f"Largest-class guess\n{floor:.2f}", ha="left", va="bottom", fontsize=9.5, color=INK2, linespacing=1.15)
ax.text(-1.55, sh_lo - 0.014, f"Shuffled Riot labels\n{sh_lo:.2f} to {sh_hi:.2f}", ha="left", va="top", fontsize=9.5, color=INK2, linespacing=1.15)

ax.set_xlim(-1.6, 1.15); ax.set_ylim(0, 1.0)
ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=10.5, color=INK)
ax.set_ylabel("Cross-validated accuracy (fraction of champions)", fontsize=10.5, color=INK2)
ax.set_yticks(np.arange(0, 1.01, 0.2)); ax.tick_params(axis="y", labelsize=10, colors=INK2); ax.tick_params(axis="x", length=0)
ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.legend(loc="upper left", frameon=False, fontsize=10, labelcolor=INK, handlelength=1.2, bbox_to_anchor=(0.0, 1.0))
fig.text(0.07, 0.965, "Champion stats predict Riot's primary class well above chance", fontsize=13, color=INK, fontweight="bold", ha="left", va="top")
fig.text(0.07, 0.915, f"N = {full['N']} champions. Stratified 5-fold CV repeated over 10 seeds, with k (kNN) and C (ridge) tuned inside the folds.\nError bars: 95% bootstrap intervals over champions.",
         fontsize=8.6, color=INK2, ha="left", va="top", linespacing=1.3)
fig.subplots_adjust(left=0.1, right=0.97, top=0.82, bottom=0.14)
fig.savefig(out, facecolor=SURF)
print("saved", out)