"""DEMO ONLY — visual variants for review.

Writes exclusively to ``analysis/figures/demo/``.
It NEVER overwrites the official figures or the official plotting scripts.

Variants produced
-----------------
H0 / H1 / H2 : heat-map cell geometry  (auto / square / near-square)
A            : stage-wise 95% CI band built from REAL bootstrap intervals
B            : stage-to-stage increment arrows with REAL paired CIs
L1 / L2      : two treatments of the ``-0.04`` label
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures" / "demo"
OUT.mkdir(parents=True, exist_ok=True)

C_RED = "#C0645C"
C_BLUE = "#4A7BA7"
C_LCR = C_BLUE
C_SLARD = C_RED
C_LINE = "#DEDEDE"
C_TEXT = "#1D2126"
C_MUTED = "#5F6670"
V_MIN, V_MAX = -16.0, 48.0

STAGES = ["Direct", "Extract+Static", "Extract+Dynamic", "Full"]
SHORT = ["Direct", "Static", "Dynamic", "Full"]
LCR_STAGE = [36.37, 33.53, 39.46, 48.19]
SLARD_STAGE = [59.92, 59.88, 57.43, 66.88]
BACKBONES = ["Qwen2.5-7B", "Qwen3-8B", "DeepSeek-R1",
             "GPT-5.4", "Claude 4.6", "LegalOne-8B"]
LCR_DELTA = [15.25, 23.21, 19.70, 7.33, 4.57, 0.90]
SLARD_DELTA = [6.73, 4.93, 6.06, 9.25, 14.80, -0.04]
CLASS_LABELS = ["Condition", "Sanction", "Responsibility", "No conflict", "Definition"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10.5, "axes.labelsize": 11.0, "xtick.labelsize": 10.5,
    "ytick.labelsize": 11.0, "legend.fontsize": 10.5,
    "text.color": C_TEXT, "axes.labelcolor": C_TEXT,
    "xtick.color": C_TEXT, "ytick.color": C_TEXT,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})

CMAP = LinearSegmentedColormap.from_list("shared", [
    (0.00, C_RED), (0.12, "#CD8279"), (0.26, "#DDA79E"), (0.38, "#EDCAC3"),
    (0.46, "#F7EFEB"), (0.50, "#F8F6F3"),
    (0.54, "#D8E5F2"), (0.62, "#B4CCE5"), (0.70, "#8FB0D3"),
    (0.80, "#6B95C4"), (0.90, "#537CAD"), (1.00, C_BLUE),
])
CMAP.set_bad("white")


def save(fig, name: str):
    for ext in (".pdf", ".png", ".svg"):
        fig.savefig(OUT / (name + ext), bbox_inches="tight",
                    pad_inches=0.05 if ext != ".svg" else 0.25,
                    dpi=170 if ext == ".png" else None)
    plt.close(fig)
    print("written", name)


def bare(ax):
    for nm, s in ax.spines.items():
        s.set_visible(nm in ("left", "bottom"))
        s.set_linewidth(0.9)
        s.set_color(C_TEXT)
    ax.tick_params(axis="y", length=3, width=0.8, color=C_TEXT, pad=6)
    ax.tick_params(axis="x", length=4, width=0.8, color=C_TEXT, pad=4)


def bold_ticks(ax):
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
        t.set_color(C_TEXT)


def legend_top(ax, handles, labels):
    leg = ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.45, 1.005),
                    ncol=1, frameon=False, markerfirst=True, handletextpad=0.5,
                    labelspacing=0.35, borderpad=0.0, borderaxespad=0.0, fontsize=10.5)
    for t in leg.get_texts():
        t.set_color(C_TEXT)
        t.set_fontweight("bold")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
with open(ROOT / "outputs" / "class_effects_LCR-CN_by_backbone.csv", encoding="utf-8") as f:
    long = list(csv.DictReader(f))
with open(ROOT / "outputs" / "class_effects_LCR-CN.csv", encoding="utf-8") as f:
    mean = list(csv.DictReader(f))
order = [int(r["class_id"]) for r in mean]
col = {"Qwen2.5-7B-Instruct": 0, "Qwen3-8B": 1, "DeepSeek-R1-0528-Qwen3-8B": 2,
       "GPT-5.4": 3, "Claude Sonnet 4.6": 4, "LegalOne-8B": 5}
mat = np.full((5, 6), np.nan)
for r in long:
    j = col.get(r["backbone"])
    i = order.index(int(r["class_id"]))
    if j is not None:
        mat[i, j] = float(r["delta"])

with open(ROOT / "outputs" / "stage_ci_real.csv", encoding="utf-8") as f:
    ci_rows = list(csv.DictReader(f))
STAGE_CI = {(r["dataset"], r["contrast"]): (float(r["ci_low"]), float(r["ci_high"]))
            for r in ci_rows if r["kind"] == "stage"}
DELTA_CI = {(r["dataset"], r["contrast"]):
            (float(r["estimate"]), float(r["ci_low"]), float(r["ci_high"]))
            for r in ci_rows if r["kind"] == "contrast"}

# --------------------------------------------------------------------------- #
# H0 / H1 / H2 : cell geometry
# --------------------------------------------------------------------------- #
def heat(aspect, figsize, name, title):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    im = ax.imshow(mat, cmap=CMAP, norm=TwoSlopeNorm(vcenter=0, vmin=V_MIN, vmax=V_MAX),
                   aspect=aspect, extent=(0, 6, 4.5, -0.5), interpolation="nearest")
    ax.set_yticks(np.arange(5))
    ax.set_yticklabels(CLASS_LABELS, weight="bold", color=C_TEXT)
    ax.set_ylim(4.5, -0.5)
    ax.set_xlim(0, 6)
    ax.set_xticks(np.arange(6) + 0.5)
    ax.set_xticklabels(BACKBONES, rotation=42, ha="right", rotation_mode="anchor")
    ax.set_xticks(np.arange(7), minor=True)
    ax.set_yticks(np.arange(6) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=1.6)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0, pad=5)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.035, pad=0.02, aspect=24)
    cb.set_ticks([-16, 0, 16, 32, 48])
    cb.set_label("Δ F1 (pp)", fontsize=10.0)
    cb.ax.tick_params(labelsize=9.5, length=0, pad=3)
    cb.outline.set_visible(False)
    ax.set_title(title, fontsize=9.0, color=C_MUTED, pad=6)
    bold_ticks(ax)
    bold_ticks(cb.ax)
    save(fig, name)


heat("auto", (4.0, 3.7), "demo_H0_auto_rect", "H0  aspect=auto (original stretched cells)")
heat("equal", (6.8, 6.2), "demo_H1_equal_square", "H1  aspect=equal (current square cells)")
heat(0.92, (4.0, 3.7), "demo_H2_near_square", "H2  aspect=0.92 (near-square, layout-balanced)")

# --------------------------------------------------------------------------- #
# A : REAL per-stage 95% CI band
# --------------------------------------------------------------------------- #
xs = np.arange(4)
fig, ax = plt.subplots(figsize=(5.0, 3.7), constrained_layout=True)
lo_l = [STAGE_CI[("LCR-CN", s)][0] for s in SHORT]
hi_l = [STAGE_CI[("LCR-CN", s)][1] for s in SHORT]
lo_s = [STAGE_CI[("SLARD-31", s)][0] for s in SHORT]
hi_s = [STAGE_CI[("SLARD-31", s)][1] for s in SHORT]
ax.fill_between(xs, lo_l, hi_l, color=C_LCR, alpha=0.20, lw=0, zorder=1)
ax.fill_between(xs, lo_s, hi_s, color=C_SLARD, alpha=0.16, lw=0, zorder=1)
ax.plot(xs, LCR_STAGE, "-o", color=C_LCR, lw=1.6, ms=7.0, mfc=C_LCR, mec=C_LCR, zorder=3)
ax.plot(xs, SLARD_STAGE, "--s", color=C_SLARD, lw=1.4, ms=6.2, mfc=C_SLARD, mec=C_SLARD, zorder=2)
for x, v in zip(xs, LCR_STAGE):
    ax.text(x, v - 3.6, f"{v:.2f}", ha="center", va="top", fontsize=9.6,
            weight="bold", color=C_TEXT, zorder=4)
for x, v in zip(xs, SLARD_STAGE):
    ax.text(x, v + 3.4, f"{v:.2f}", ha="center", va="bottom", fontsize=9.6,
            weight="bold", color=C_TEXT, zorder=4)
for vals, colr in ((LCR_STAGE, C_LCR), (SLARD_STAGE, C_SLARD)):
    ax.text(xs[-1] + 0.10, vals[-1], "***", color=colr, fontsize=11.0,
            weight="bold", ha="left", va="center", zorder=5)
ax.set_xticks(xs)
ax.set_xticklabels(STAGES)
ax.set_xlim(-0.4, 3.4)
ax.set_ylim(28.0, 72.0)
ax.set_yticks([30, 40, 50, 60, 70])
ax.set_ylabel("Mean Macro-F1 (%)")
ax.set_axisbelow(True)
bare(ax)
bold_ticks(ax)
legend_top(ax,
           [Line2D([], [], ls="-", marker="o", ms=6.5, color=C_LCR),
            Line2D([], [], ls="--", marker="s", ms=5.8, color=C_SLARD)],
           ["LCR-CN", "SLARD-31"])
save(fig, "demo_A_real_CI_band")

# --------------------------------------------------------------------------- #
# B : stage-to-stage increment arrows with REAL paired CIs
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(5.4, 3.9), constrained_layout=True)
ax.plot(xs, LCR_STAGE, "-o", color=C_LCR, lw=1.6, ms=7.0, mfc=C_LCR, mec=C_LCR, zorder=3)
ax.plot(xs, SLARD_STAGE, "--s", color=C_SLARD, lw=1.4, ms=6.2, mfc=C_SLARD, mec=C_SLARD, zorder=2)
# arrows sit in the middle of each interval; labels are pushed away from the
# curves into the empty corridor between the two series (y ~= 50-56)
for vals, colr, tag_y, ds in ((LCR_STAGE, C_LCR, 52.0, "LCR-CN"),
                              (SLARD_STAGE, C_SLARD, 55.2, "SLARD-31")):
    for i in range(3):
        key = f"{SHORT[i+1]} - {SHORT[i]}"
        d, lo, hi = DELTA_CI[(ds, key)]
        xm = i + 0.5
        y0, y1 = vals[i], vals[i + 1]
        ax.annotate("", xy=(xm, y1), xytext=(xm, y0),
                    arrowprops=dict(arrowstyle="-|>", color=colr, lw=1.0,
                                    shrinkA=3, shrinkB=3, alpha=0.75), zorder=3)
        ax.text(xm, tag_y, f"{d:+.2f}  [{lo:+.2f}, {hi:+.2f}]",
                ha="center", va="center", fontsize=7.6, color=colr,
                weight="bold", zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2))
# point values are omitted here: the arrows already quantify every change,
# and absolute values can be read from the y-axis (scheme A keeps them)
for vals, colr in ((LCR_STAGE, C_LCR), (SLARD_STAGE, C_SLARD)):
    ax.text(xs[-1] + 0.10, vals[-1], "***", color=colr, fontsize=11.0,
            weight="bold", ha="left", va="center", zorder=5)
ax.set_xticks(xs)
ax.set_xticklabels(STAGES)
ax.set_xlim(-0.4, 3.4)
ax.set_ylim(26.0, 74.0)
ax.set_yticks([30, 40, 50, 60, 70])
ax.set_ylabel("Mean Macro-F1 (%)")
ax.set_axisbelow(True)
bare(ax)
bold_ticks(ax)
legend_top(ax,
           [Line2D([], [], ls="-", marker="o", ms=6.5, color=C_LCR),
            Line2D([], [], ls="--", marker="s", ms=5.8, color=C_SLARD)],
           ["LCR-CN", "SLARD-31"])
save(fig, "demo_B_increment_CI")

# --------------------------------------------------------------------------- #
# L1 / L2 : treatment of the -0.04 label
# --------------------------------------------------------------------------- #
def delta_bars(name, mode, title):
    fig, ax = plt.subplots(figsize=(5.0, 3.7), constrained_layout=True)
    ys = np.arange(6)
    ax.barh(ys + 0.17, LCR_DELTA, height=0.34, color=C_LCR, edgecolor="none", zorder=2)
    ax.barh(ys - 0.17, SLARD_DELTA, height=0.34, color=C_SLARD, edgecolor="none", zorder=2)
    for y, v in zip(ys, LCR_DELTA):
        ax.text(v + 0.45, y + 0.17, f"{v:+.2f}", ha="left", va="center",
                fontsize=9.6, weight="bold", color=C_TEXT, zorder=4)
    for y, v in zip(ys, SLARD_DELTA):
        if v >= 0:
            ax.text(v + 0.45, y - 0.17, f"{v:+.2f}", ha="left", va="center",
                    fontsize=9.6, weight="bold", color=C_TEXT, zorder=4)
        else:
            ax.plot(0, y - 0.17, marker="o", ms=4.2, mfc="white", mec=C_SLARD,
                    mew=1.4, zorder=5, clip_on=False)
            if mode == "current":
                ax.text(0.45, y - 0.17, f"{v:+.2f}", ha="left", va="center",
                        fontsize=9.6, weight="bold", color=C_TEXT, zorder=4)
            else:  # leader line, label pushed clear of the axis and tick labels
                ax.annotate(f"{v:+.2f}", xy=(0, y - 0.17), xytext=(3.6, y - 0.17),
                            ha="left", va="center", fontsize=9.6, weight="bold",
                            color=C_TEXT,
                            arrowprops=dict(arrowstyle="-", color=C_SLARD, lw=0.9,
                                            shrinkA=3, shrinkB=4), zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels(BACKBONES, weight="bold", color=C_TEXT)
    ax.set_ylim(5.5, -0.5)
    ax.set_xlim(0, 26.5)
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.set_xlabel("Full − Direct Macro-F1 (points)")
    ax.set_axisbelow(True)
    bare(ax)
    bold_ticks(ax)
    legend_top(ax,
               [Line2D([], [], ls="none", marker="s", ms=7.0, color=C_LCR),
                Line2D([], [], ls="none", marker="s", ms=7.0, color=C_SLARD)],
               ["LCR-CN", "SLARD-31"])
    save(fig, name)


delta_bars("demo_L1_label_current", "current", "L1  label immediately right of zero")
delta_bars("demo_L2_label_leader", "leader", "L2  zero marker + leader line, label moved clear")

# --------------------------------------------------------------------------- #
# H3 : square cells, but the PDF height is locked to panel (a)'s 268 pt.
#      This is the only geometry that keeps square cells AND equal scaling.
#      bbox_inches is disabled so the exported canvas is exactly figsize.
# --------------------------------------------------------------------------- #
def heat_aligned(figsize, name, title):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    im = ax.imshow(mat, cmap=CMAP, norm=TwoSlopeNorm(vcenter=0, vmin=V_MIN, vmax=V_MAX),
                   aspect="equal", extent=(0, 6, 4.5, -0.5), interpolation="nearest")
    ax.set_yticks(np.arange(5))
    ax.set_yticklabels(CLASS_LABELS, weight="bold", color=C_TEXT)
    ax.set_ylim(4.5, -0.5)
    ax.set_xlim(0, 6)
    ax.set_xticks(np.arange(6) + 0.5)
    ax.set_xticklabels(BACKBONES, rotation=42, ha="right", rotation_mode="anchor")
    ax.set_xticks(np.arange(7), minor=True)
    ax.set_yticks(np.arange(6) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=1.6)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0, pad=5)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.035, pad=0.02, aspect=24)
    cb.set_ticks([-16, 0, 16, 32, 48])
    cb.set_label("Δ F1 (pp)", fontsize=10.0)
    cb.ax.tick_params(labelsize=9.5, length=0, pad=3)
    cb.outline.set_visible(False)
    bold_ticks(ax)
    bold_ticks(cb.ax)
    for ext in (".pdf", ".png", ".svg"):
        fig.savefig(OUT / (name + ext), pad_inches=0.0,
                    dpi=170 if ext == ".png" else None)
    plt.close(fig)
    print("written", name)


heat_aligned((4.86, 3.72), "demo_H3_square_aligned",
             "H3  square cells, canvas locked to panel (a) height")

print("\nall demo files in", OUT)
