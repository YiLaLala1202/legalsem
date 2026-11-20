# 跨系统对比：阶段性能分解 + 各骨干 ΔF1
# 与 plot_class_effect_lcr.py 同一套配色、字体、坐标轴与导出设置；IDE 点运行直接出 PDF+SVG
# 数字、名字、颜色都在配置区，改图只改这一段

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无界面后端：只写文件，不弹窗
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ==================== 配置区 ====================
FIG_DIR = (Path(__file__).resolve().parent / "figures")  # 输出目录
PREFIX = "fig_stage_backbone"  # 两张图共用的文件名前缀
NAME_A = "_a_stage_wise"  # 图(a) 阶段性能分解
NAME_B = "_b_backbone_delta"  # 图(b) 各骨干提升幅度

STAGES = ["Direct", "Extract+Static", "Extract+Dynamic", "Full"]  # 图(a) 横轴：四个阶段
LCR_STAGE = [36.37, 33.53, 39.46, 48.19]  # 图(a) LCR-CN 的均值 Macro-F1
SLARD_STAGE = [59.92, 59.88, 57.43, 66.88]  # 图(a) SLARD-31 的均值 Macro-F1

# 图(a) 真实 95% bootstrap 置信区间（5,000 次重采样、seed 2026、测试实例为抽样单位）
# 来源：ci_stage_bootstrap.py -> outputs/stage_ci_real.csv；八个阶段均值已逐位复现论文
CI_LCR_LO = [34.79, 32.02, 37.84, 46.32]
CI_LCR_HI = [37.83, 35.01, 41.00, 49.96]
CI_SLARD_LO = [58.06, 58.40, 55.66, 65.22]
CI_SLARD_HI = [60.92, 60.83, 58.56, 68.14]
BAND_ALPHA_LCR = 0.20  # 区间带透明度
BAND_ALPHA_SLARD = 0.16

BACKBONES = [  # 图(b) 纵轴：六个骨干（短名与热力图那套保持一致）
    "Qwen2.5-7B", "Qwen3-8B", "DeepSeek-R1",
    "GPT-5.4", "Claude 4.6", "LegalOne-8B",
]
LCR_DELTA = [15.25, 23.21, 19.70, 7.33, 4.57, 0.90]  # 图(b) LCR-CN 的 Full − Direct
SLARD_DELTA = [6.73, 4.93, 6.06, 9.25, 14.80, -0.04]  # 图(b) SLARD-31 的 Full − Direct

C_PURPLE = "#8A7BB5"  # 紫：L*=55 C*=34，与 Fig.1 的淡紫同族
C_BLUE = "#5597C7"  # 蓝：L*=60 C*=32
C_LCR = C_BLUE  # LCR-CN = 蓝
C_SLARD = C_PURPLE  # SLARD-31 = 紫
C_LINE = "#DEDEDE"  # 参考线
C_TEXT = "#000000"  # 文字与刻度数字
# 图(b) 实心条与折线同值：同一角色永远用同一个色，才是"一个色系"
C_BAR_LCR = C_BLUE
C_BAR_SLARD = C_PURPLE

Y_MIN, Y_MAX = 24.0, 76.0  # 图(a) 纵轴范围（留出点值与 CI 标注两行）
Y_TICKS_A = [30, 40, 50, 60, 70]  # 图(a) 纵轴刻度
X_MIN_B, X_MAX_B = -2.0, 27.5  # 图(b) 横轴范围（左轴即 0 基准）
X_TICKS_B = [0, 5, 10, 15, 20, 25]  # 图(b) 横轴刻度
BAR_H = 0.34  # 图(b) 每组两根条的高度
LBL_GAP = 0.9  # 图(b) 数值与条末端的间距
PAD = 0.05  # PDF 导出边距
PAD_SVG = 0.25  # SVG 导出边距：四周多留白

plt.rcParams.update({  # 与分类效应图同一套：Arial、可编辑矢量
    "font.family": "sans-serif",
    "font.sans-serif": ["Calibri", "DejaVu Sans"],
    "font.size": 10.5,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.labelsize": 11.0,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 11.0,
    "legend.fontsize": 10.5,
    "text.color": C_TEXT,
    "axes.labelcolor": C_TEXT,
    "xtick.color": C_TEXT,
    "ytick.color": C_TEXT,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})
# ==============================================


def bare(ax):  # 只留左、下两条坐标轴，其余三边去掉
    for name, s in ax.spines.items():
        s.set_visible(name in ("left", "bottom"))
        s.set_linewidth(0.9)
        s.set_color(C_TEXT)
    ax.tick_params(axis="y", length=3, width=0.8, color=C_TEXT, pad=6)
    ax.tick_params(axis="x", length=4, width=0.8, color=C_TEXT, pad=4)


def bold_ticks(ax):  # 所有刻度数字：黑色加粗
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
        t.set_color(C_TEXT)


def auto_legend(ax, handles, labels, ncol=2, fontsize=9.0, edge="#AEB6BF",
                anchor=None, handlelength=2.0):
    """把图例放到绘图区里最空的一块：栅格化所有数据与文字，逐个候选位置打分取最优。
    anchor=(x, y) 时按轴分数直接指定位置（仍会夹到图内，不贴轴）。
    不再用写死的角落锚点，边框也加深到可见。"""
    leg = ax.legend(handles=handles, labels=labels, loc="lower left", ncol=ncol,
                    frameon=True, facecolor="white", edgecolor=edge, framealpha=1.0,
                    markerfirst=True, handletextpad=0.45, columnspacing=1.4,
                    handlelength=handlelength, borderpad=0.45, borderaxespad=0.0,
                    fontsize=fontsize)
    for t in leg.get_texts():
        t.set_color(C_TEXT)
        t.set_fontweight("bold")
    leg.get_frame().set_linewidth(0.8)

    fig = ax.figure
    fig.canvas.draw()
    le, ab = leg.get_window_extent(), ax.get_window_extent()
    lw_, lh_ = le.width / ab.width, le.height / ab.height

    if anchor is not None:  # 指定位置：夹进图内，但不再搜索
        xa = min(max(anchor[0], 0.015), 0.985 - lw_)
        ya = min(max(anchor[1], 0.015), 0.985 - lh_)
        leg.set_bbox_to_anchor((xa, ya), transform=ax.transAxes)
        return leg

    G = 56
    occ = np.zeros((G, G), dtype=bool)

    def to_ax(pts):
        return (pts[:, 0] - ab.x0) / ab.width, (pts[:, 1] - ab.y0) / ab.height

    def mark(xa, ya, r=1):
        j, i = int(xa * G), int(ya * G)
        occ[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1] = True

    for ln in ax.lines:
        xs_, ys_ = ln.get_data()
        if xs_ is None or len(xs_) == 0:
            continue
        pts = ln.get_transform().transform(np.column_stack([np.asarray(xs_, float),
                                                            np.asarray(ys_, float)]))
        xa, ya = to_ax(pts)
        r = 2 if ln.get_marker() not in ("None", None, "no") else 1
        for a, b in zip(xa, ya):
            mark(a, b, r)
    for t in ax.texts:
        tb = t.get_window_extent()
        for a in np.linspace((tb.x0 - ab.x0) / ab.width, (tb.x1 - ab.x0) / ab.width, 6):
            for b in np.linspace((tb.y0 - ab.y0) / ab.height, (tb.y1 - ab.y0) / ab.height, 4):
                mark(a, b, 0)
    for p in ax.patches:
        try:
            pb = p.get_window_extent()
        except Exception:
            continue
        if pb.width * pb.height > 0.02 * ab.width * ab.height:
            for a in np.linspace((pb.x0 - ab.x0) / ab.width, (pb.x1 - ab.x0) / ab.width, 12):
                for b in np.linspace((pb.y0 - ab.y0) / ab.height, (pb.y1 - ab.y0) / ab.height, 12):
                    mark(a, b, 1)

    best = None
    # 最大空矩形（直方图法），把图例放进这块空地的右上角
    heights = np.zeros(G, dtype=int)
    lwr, lhr = max(1, int(np.ceil(lw_ * G))), max(1, int(np.ceil(lh_ * G)))
    for r in range(G):
        heights = np.where(~occ[r], heights + 1, 0)
        stack = []
        for c in range(G + 1):
            cur = heights[c] if c < G else 0
            start = c
            while stack and stack[-1][1] > cur:
                sc, sh = stack.pop()
                if sh * (c - sc) > (best[0] if best else 0):
                    best = (sh * (c - sc), r - sh + 1, r, sc, c - 1)
                start = sc
            stack.append((start, cur))
    area, r0, r1, c0, c1 = best
    if area >= lwr * lhr:  # 放得下就贴这块空地的右上角
        xa = (c1 - lwr + 1) / G
        ya = (r1 - lhr + 1) / G
    else:  # 放不下就居中，宁可与数据挤一点也不贴轴
        xa = max(0.02, c0 / G + ((c1 - c0 + 1) / G - lw_) / 2)
        ya = max(0.02, r0 / G + ((r1 - r0 + 1) / G - lh_) / 2)
    xa = min(max(xa, 0.015), 0.985 - lw_)
    ya = min(max(ya, 0.015), 0.985 - lh_)
    leg.set_bbox_to_anchor((xa, ya), transform=ax.transAxes)
    return leg


def save(fig, base):  # 只导出 PDF / SVG
    fig.savefig(base.with_suffix(".png"), dpi=160, bbox_inches="tight"); fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=PAD, dpi=600)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=PAD_SVG, dpi=600)
    plt.close(fig)
    for ext in (".pdf", ".svg"):
        print("written", base.with_suffix(ext))


FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 图(a) 阶段性能分解：两条折线 + 点旁数值 ----------
xs = np.arange(len(STAGES))
fig, ax = plt.subplots(figsize=(5.0, 4.1), constrained_layout=True)
ax.fill_between(xs, CI_LCR_LO, CI_LCR_HI,  # LCR-CN：真实 95% CI 带
                color=C_LCR, alpha=BAND_ALPHA_LCR, lw=0, zorder=1)
ax.fill_between(xs, CI_SLARD_LO, CI_SLARD_HI,  # SLARD-31：真实 95% CI 带
                color=C_SLARD, alpha=BAND_ALPHA_SLARD, lw=0, zorder=1)
ax.plot(xs, LCR_STAGE, "-o", color=C_LCR, lw=1.6, ms=7.0, mfc=C_LCR,  # LCR-CN：实线 + 圆点
        mec=C_LCR, zorder=3)
ax.plot(xs, SLARD_STAGE, "--s", color=C_SLARD, lw=1.4, ms=6.2, mfc=C_SLARD,  # SLARD-31：虚线 + 方块
        mec=C_SLARD, zorder=2)
# 每个阶段两行：点值（大、黑、加粗）+ 该阶段均值的 95% CI（小、与线同色）
# 两行都落在色带之外，并用极淡白底保证不与带子/折线相互遮挡
for x, v, lo, hi in zip(xs, LCR_STAGE, CI_LCR_LO, CI_LCR_HI):
    ax.text(x, v - 3.4, f"{v:.2f}", ha="center", va="top", fontsize=9.4,
            weight="bold", color=C_TEXT, zorder=4)
    ax.text(x, v - 6.3, f"[{lo:.1f}, {hi:.1f}]", ha="center", va="top", fontsize=8.6, weight="bold",
            color=C_TEXT, zorder=4,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.5))
for x, v, lo, hi in zip(xs, SLARD_STAGE, CI_SLARD_LO, CI_SLARD_HI):
    ax.text(x, v - 3.4, f"{v:.2f}", ha="center", va="top", fontsize=9.4,  # 与蓝线一致：放线下方
            weight="bold", color=C_TEXT, zorder=4)
    ax.text(x, v - 6.3, f"[{lo:.1f}, {hi:.1f}]", ha="center", va="top", fontsize=8.6, weight="bold",
            color=C_TEXT, zorder=4,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.5))
# paired-inference annotation: the Full - Direct gain is significant on both tasks
for vals, col in ((LCR_STAGE, C_LCR), (SLARD_STAGE, C_SLARD)):
    ax.text(xs[-1] + 0.10, vals[-1], "***", color=C_TEXT, fontsize=11.0,
            weight="bold", ha="left", va="center", zorder=5)
ax.set_xticks(xs)
ax.set_xticklabels(STAGES)
ax.set_xlim(-0.4, len(STAGES) - 0.6)
ax.set_ylim(Y_MIN, Y_MAX)
ax.set_yticks(Y_TICKS_A)
ax.set_ylabel("Mean Macro-F1 (%)")
ax.set_axisbelow(True)
bare(ax)
bold_ticks(ax)  # 刻度数字黑色加粗
auto_legend(ax,  # 图例：指定到上方居中偏右的空位
           [Line2D([], [], ls="-", marker="o", ms=6.5, color=C_LCR, mec=C_LCR),
            Line2D([], [], ls=(0, (3.2, 1.6)), marker="s", ms=5.8, color=C_SLARD,
                   mec=C_SLARD, dash_capstyle="butt")],  # 短划：手柄里也看得出是虚线
           ["LCR-CN", "SLARD-31"], anchor=(0.415, 0.895), handlelength=2.3)
save(fig, FIG_DIR / (PREFIX + NAME_A))

# ---------- 图(b) 各骨干提升幅度：点 + 细茎（彩色面积压到 12 个点） ----------
n = len(BACKBONES)
ys = np.arange(n)
# 与图(a) 保持同一画布高度：两图 PDF 尺寸一致，并排时缩放比与显示高度都相同
fig, ax = plt.subplots(figsize=(5.0, 4.1), constrained_layout=True)
ax.axvline(0, color=C_LINE, lw=0.9, zorder=0)  # 0 基准
for y, v in zip(ys, LCR_DELTA):
    ax.plot([0, v], [y + 0.17] * 2, color=C_BAR_LCR, lw=1.2, alpha=0.6,  # 细茎
            solid_capstyle="round", zorder=1)
    ax.plot(v, y + 0.17, "o", ms=6.8, color=C_BAR_LCR, zorder=3)
    ax.text(v + LBL_GAP, y + 0.17, f"{v:+.2f}", ha="left", va="center",
            fontsize=9.6, weight="bold", color=C_TEXT, zorder=4)
for y, v in zip(ys, SLARD_DELTA):
    m = v
    ax.plot([0, m], [y - 0.17] * 2, color=C_BAR_SLARD, lw=1.2, alpha=0.6,
            solid_capstyle="round", zorder=1)
    ax.plot(m, y - 0.17, "o", ms=6.8, mfc="white", mec=C_BAR_SLARD, mew=1.8, zorder=3)
    ax.text(m + LBL_GAP, y - 0.17, f"{v:+.2f}", ha="left", va="center",
            fontsize=9.6, weight="bold", color=C_TEXT, zorder=4)
ax.set_yticks(ys)
ax.set_yticklabels(BACKBONES, weight="bold", color=C_TEXT)
ax.set_ylim(n - 0.5, -0.5)  # 第一个骨干在最上
ax.set_xlim(X_MIN_B, X_MAX_B)
ax.set_xticks(X_TICKS_B)
ax.set_xlabel("Full − Direct Macro-F1 (pp)", labelpad=7)
ax.set_axisbelow(True)
bare(ax)
bold_ticks(ax)  # 刻度数字黑色加粗
auto_legend(ax,
           [Line2D([], [], ls="none", marker="o", ms=6.8, color=C_BAR_LCR, mec=C_BAR_LCR),
            Line2D([], [], ls="none", marker="o", ms=6.8, mfc="white", mec=C_BAR_SLARD, mew=1.8)],
           ["LCR-CN", "SLARD-31"])
save(fig, FIG_DIR / (PREFIX + NAME_B))
