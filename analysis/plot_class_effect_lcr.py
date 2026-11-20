# LCR-CN 类别级效应：三张独立图
# 核心结论：完整框架在 4/5 个类别上优于 Direct，六个骨干方向一致；覆盖率解释了定义类的短板
# IDE 点运行直接出 PDF/SVG/TIFF/PNG，只依赖 numpy + matplotlib；路径/名字/配色都在配置区

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无界面后端：只写文件，不弹窗
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

# ==================== 配置区 ====================
DATA_DIR = (Path(__file__).resolve().parent / "outputs")  # 源 CSV 目录
FIG_DIR = (Path(__file__).resolve().parent / "figures")  # 输出目录
MEAN_CSV = DATA_DIR / "class_effects_LCR-CN.csv"  # 类别均值表
LONG_CSV = DATA_DIR / "class_effects_LCR-CN_by_backbone.csv"  # 逐骨干长表
PREFIX = "fig_class_effect_LCR-CN"  # 三张图共用的文件名前缀
NAME_A = "_a_per_class_f1"  # 图(a) 逐类 F1
NAME_B = "_b_delta_by_backbone"  # 图(b) 各骨干 ΔF1 热力图
NAME_C = "_c_coverage"  # 图(c) 候选覆盖率

CLASS_NAMES = {  # 左侧类别名（样本数放图注）
    0: "No conflict",
    1: "Responsibility",
    2: "Definition",
    3: "Sanction",
    4: "Condition",
}
BACKBONES = [  # 热力图列：(CSV 全称, 图上短名)，顺序即列顺序
    ("Qwen2.5-7B-Instruct", "Qwen2.5-7B"),
    ("Qwen3-8B", "Qwen3-8B"),
    ("DeepSeek-R1-0528-Qwen3-8B", "DeepSeek-R1"),
    ("GPT-5.4", "GPT-5.4"),
    ("Claude Sonnet 4.6", "Claude 4.6"),
    ("LegalOne-8B", "LegalOne-8B"),
]
LBL_SHORT = ["Qwen2.5-7B", "Qwen3-8B", "DeepSeek-R1", "GPT-5.4", "Claude 4.6", "LegalOne-8B"]  # 热力图斜排用

V_MIN, V_MAX = -16.0, 48.0  # 热力图色标：恰好覆盖实测 -15.9…+48.3
NOISE_BAND = 5.0  # |ΔF1| 小于该值算噪声
FADE = 0.45  # 噪声带内的透明度
X_MAX_A = 76.0  # 图(a) x 轴上限
COVERAGE_XMAX = 105.0  # 图(c) x 轴上限
PAD = 0.05  # PDF 导出边距
PAD_SVG = 0.25  # SVG 导出边距：四周多留白
TAG_GAP = 1.7  # 数值与圆点的间距（数据单位）

C_PURPLE = "#8A7BB5"  # 紫：L*=55 C*=34，与 Fig.1 的淡紫面板同族
C_BLUE = "#5597C7"  # 蓝：L*=60 C*=32
C_DIRECT = C_PURPLE  # Direct 基线 = 紫
C_FULL = C_BLUE  # Full 本方法 = 蓝
C_DIFF_BG = "#F2F5F8"  # 森林图栏底色：压浅，不再是一块明显色块
C_LINE = "#DEDEDE"  # 参考线
C_LINK = "#D8DDE2"  # 哑铃连线：中性浅灰
C_TEXT = "#000000"  # 文字与刻度数字
C_MUTED = "#000000"  # 次要文字

# 热力图色表：紫 → 近白 → 蓝，彩度整体压到 L*66/68 一档，向 Fig.1 的粉彩靠拢
CMAP = LinearSegmentedColormap.from_list("lcr_shared", [(0.0, "#665080"), (0.5, "#F7F7FA"), (1.0, "#2A6796")])
CMAP.set_bad("white")
CMAP_BAR = LinearSegmentedColormap.from_list("cov_ramp", ["#EDF3F9", "#8FB2D3", C_BLUE])  # 覆盖率越高越深

plt.rcParams.update({  # 三张图共用：Arial、可编辑矢量
    "font.family": "sans-serif",
    "font.sans-serif": ["Calibri", "DejaVu Sans"],
    "font.size": 10.5,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titlesize": 12.0,
    "axes.titleweight": "bold",
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


def auto_legend(ax, handles, labels, ncol=2, fontsize=9.0, edge="#AEB6BF"):
    """把图例放到绘图区里最空的一块：栅格化所有数据与文字，逐个候选位置打分取最优。
    不再用写死的角落锚点，边框也加深到可见。"""
    leg = ax.legend(handles=handles, labels=labels, loc="lower left", ncol=ncol,
                    frameon=True, facecolor="white", edgecolor=edge, framealpha=1.0,
                    markerfirst=True, handletextpad=0.45, columnspacing=1.4,
                    borderpad=0.45, borderaxespad=0.0, fontsize=fontsize)
    for t in leg.get_texts():
        t.set_color(C_TEXT)
        t.set_fontweight("bold")
    leg.get_frame().set_linewidth(0.8)

    fig = ax.figure
    fig.canvas.draw()
    le, ab = leg.get_window_extent(), ax.get_window_extent()
    lw_, lh_ = le.width / ab.width, le.height / ab.height  # 图例占轴的分数尺寸

    G = 56  # 占用栅格
    occ = np.zeros((G, G), dtype=bool)

    def to_ax(pts):  # 显示坐标 -> 轴分数
        return (pts[:, 0] - ab.x0) / ab.width, (pts[:, 1] - ab.y0) / ab.height

    def mark(xa, ya, r=1):
        j, i = int(xa * G), int(ya * G)
        occ[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1] = True

    for ln in ax.lines:  # 折线 / 散点
        xs_, ys_ = ln.get_data()
        if xs_ is None or len(xs_) == 0:
            continue
        pts = ln.get_transform().transform(np.column_stack([np.asarray(xs_, float),
                                                            np.asarray(ys_, float)]))
        xa, ya = to_ax(pts)
        r = 2 if ln.get_marker() not in ("None", None, "no") else 1
        for a, b in zip(xa, ya):
            mark(a, b, r)
    for t in ax.texts:  # 数值与标注
        tb = t.get_window_extent()
        for a in np.linspace((tb.x0 - ab.x0) / ab.width, (tb.x1 - ab.x0) / ab.width, 6):
            for b in np.linspace((tb.y0 - ab.y0) / ab.height, (tb.y1 - ab.y0) / ab.height, 4):
                mark(a, b, 0)
    for p in ax.patches:  # 背景块等
        try:
            pb = p.get_window_extent()
        except Exception:
            continue
        if pb.width * pb.height > 0.02 * ab.width * ab.height:  # 大块整体标记
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


def save_fixed(fig, base):  # 固定画布、不裁切：让图(b) 的 PDF 高度与图(a) 完全一致
    fig.savefig(base.with_suffix(".png"), dpi=160, bbox_inches="tight"); fig.savefig(base.with_suffix(".pdf"), bbox_inches=None, pad_inches=0.0, dpi=600)
    fig.savefig(base.with_suffix(".svg"), bbox_inches=None, pad_inches=0.0, dpi=600)
    plt.close(fig)
    for ext in (".pdf", ".svg"):
        print("written", base.with_suffix(ext))


# 读数据
with open(MEAN_CSV, encoding="utf-8") as f:
    mean = list(csv.DictReader(f))
with open(LONG_CSV, encoding="utf-8") as f:
    long = list(csv.DictReader(f))
for r in mean:
    r["class_id"] = int(r["class_id"])
    r["support"] = int(r["support"])
    for k in ("f1_direct", "f1_full", "delta", "coverage"):
        r[k] = float(r[k]) if r[k] else float("nan")
for r in long:
    r["class_id"] = int(r["class_id"])
    r["delta"] = float(r["delta"])

n = len(mean)
ys = np.arange(n)
ids = [r["class_id"] for r in mean]
labels = [CLASS_NAMES.get(r["class_id"], r["class_name"]) for r in mean]
means = [r["coverage"] if r["coverage"] == r["coverage"] else 0.0 for r in mean]
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 图(a) 逐类 F1：极简哑铃 + 右侧森林图 ----------
fig, ax = plt.subplots(figsize=(5.8, 3.7), constrained_layout=True)
for y, r in zip(ys, mean):
    a = 1.0 if abs(r["delta"]) >= NOISE_BAND else FADE
    ax.plot([r["f1_direct"], r["f1_full"]], [y, y], color=C_LINK, lw=2.4,  # 中性浅灰连线
            solid_capstyle="round", alpha=a, zorder=1)
    ax.plot(r["f1_direct"], y, marker="o", ls="none", ms=7.6, mfc="white",  # Direct：空心圆
            mec=C_DIRECT, mew=1.8, alpha=a, zorder=2)
    ax.plot(r["f1_full"], y, marker="o", ls="none", ms=7.6, mfc=C_FULL,  # Full：实心圆
            mec=C_FULL, alpha=a, zorder=3)
    if r["f1_full"] >= r["f1_direct"]:  # 数值放两端外侧
        ax.text(r["f1_direct"] - TAG_GAP, y, f"{r['f1_direct']:.1f}", ha="right",
                va="center", fontsize=9.6, weight="bold", color=C_TEXT, alpha=a, zorder=4)
        ax.text(r["f1_full"] + TAG_GAP, y, f"{r['f1_full']:.1f}", ha="left",
                va="center", fontsize=9.6, weight="bold", color=C_TEXT, alpha=a, zorder=4)
    else:  # 下降行换边，避免相撞
        ax.text(r["f1_direct"] + TAG_GAP, y, f"{r['f1_direct']:.1f}", ha="left",
                va="center", fontsize=9.6, weight="bold", color=C_TEXT, alpha=a, zorder=4)
        ax.text(r["f1_full"] - TAG_GAP, y, f"{r['f1_full']:.1f}", ha="right",
                va="center", fontsize=9.6, weight="bold", color=C_TEXT, alpha=a, zorder=4)
ax.set_yticks(ys)
ax.set_yticklabels(labels, weight="bold", color=C_TEXT)
ax.set_ylim(n - 0.5, -0.5)
ax.set_xlim(0, X_MAX_A)
ax.set_xticks([0, 25, 50, 75])
ax.set_xlabel("Class F1 (%)")
ax.set_axisbelow(True)
bare(ax)
bold_ticks(ax)  # 刻度数字黑色加粗
auto_legend(ax,  # 图例：按数据占用自动找最空的一块
            [Line2D([], [], ls="none", marker="o", ms=7.0, mfc="white",
                    mec=C_DIRECT, mew=1.8),
             Line2D([], [], ls="none", marker="o", ms=7.0, mfc=C_FULL,
                    mec=C_FULL)],
            ["Direct", "Full"])

dt = blended_transform_factory(ax.transAxes, ax.transData)  # 右侧：内嵌森林图
CI = {  # class_id -> (delta, 2.5%, 97.5%)；5000 次配对 bootstrap
    0: (7.04, 4.20, 9.87), 1: (14.24, 12.12, 16.42),
    2: (-3.79, -7.98, 0.51), 3: (18.06, 15.00, 21.10),
    4: (23.55, 21.15, 26.04),
}
D_LO, D_HI = -9.0, 27.0
X0, X1 = 1.035, 1.315
def dx(v): return X0 + (v - D_LO) / (D_HI - D_LO) * (X1 - X0)
ax.add_patch(Rectangle((1.02, -0.80), 0.30, n + 0.30, transform=dt,  # 栏底色
                       facecolor=C_DIFF_BG, edgecolor="none", clip_on=False, zorder=0))
ax.text((X0 + X1) / 2, -0.68, "Δ (95% CI)", transform=dt, ha="center",
        va="center", fontsize=9.0, weight="bold", clip_on=False)
ax.plot([dx(0), dx(0)], [-0.5, n - 0.5], transform=dt, color=C_MUTED,
        lw=0.8, ls=(0, (2, 2)), alpha=0.7, clip_on=False, zorder=1)
for y, r in zip(ys, mean):
    v, lo, hi = CI[r["class_id"]]
    c = C_FULL if v > 0 else C_DIRECT
    a = 1.0 if abs(v) >= NOISE_BAND else FADE
    ax.plot([dx(lo), dx(hi)], [y, y], transform=dt, color=c, lw=1.8,  # 95% CI
            alpha=a, solid_capstyle="round", clip_on=False, zorder=2)
    ax.plot(dx(v), y, "o", ms=4.6, color=c, alpha=a, transform=dt,  # 点估计
            mec="white", mew=0.9, clip_on=False, zorder=3)
save(fig, FIG_DIR / (PREFIX + NAME_A))

# ---------- 图(b) 各骨干 ΔF1 热力图：正值(提升)颜色更重 ----------
col = {full: j for j, (full, _) in enumerate(BACKBONES)}
mat = np.full((n, len(BACKBONES)), np.nan)
for r in long:
    j = col.get(r["backbone"])
    if j is not None:
        mat[ids.index(r["class_id"]), j] = r["delta"]
# 正方形格子 + 固定画布 4.86x3.72in：导出高度 268pt，与图(a) 的 267.6pt 齐平
fig, ax = plt.subplots(figsize=(5.5, 3.9), constrained_layout=True)
im = ax.pcolormesh(np.arange(len(BACKBONES) + 1), np.arange(n + 1) - 0.5, mat,  # 矢量网格：放大不糊
                   cmap=CMAP, norm=TwoSlopeNorm(vcenter=0, vmin=V_MIN, vmax=V_MAX),
                   shading="flat", edgecolors="none", antialiased=False)
ax.set_yticks(np.arange(n))
ax.set_yticklabels(labels, weight="bold", color=C_TEXT)
ax.set_ylim(n - 0.5, -0.5)
ax.set_xlim(0, len(BACKBONES))
ax.set_aspect("equal")
ax.set_xticks(np.arange(len(BACKBONES)) + 0.5)
ax.set_xticklabels(LBL_SHORT, rotation=45, ha="right", rotation_mode="anchor",
                   fontsize=8.2, style="normal")
ax.tick_params(which="major", length=0, pad=4)  # 连续色块：不画格间白缝
for name_, s in ax.spines.items():  # 只留一圈极细的外框，界定整片色域
    s.set_visible(True)
    s.set_linewidth(0.6)
    s.set_color("#E4E4E4")
for i in range(n):  # 格内数值：深底白字、浅底黑字
    for j in range(len(BACKBONES)):
        v = mat[i, j]
        if v == v:
            ax.text(j + 0.5, i, f"{v:+.0f}", ha="center", va="center", fontsize=9.2,
                    color="white" if (0.299 * CMAP(TwoSlopeNorm(vcenter=0, vmin=V_MIN, vmax=V_MAX)(v))[0] + 0.587 * CMAP(TwoSlopeNorm(vcenter=0, vmin=V_MIN, vmax=V_MAX)(v))[1] + 0.114 * CMAP(TwoSlopeNorm(vcenter=0, vmin=V_MIN, vmax=V_MAX)(v))[2]) < 0.56 else "#1D2126", zorder=3)
cb = fig.colorbar(im, ax=ax, orientation="horizontal",  # 先借它算位置与刻度
                  fraction=0.05, pad=0.035, aspect=26)
cb.set_ticks([-16, 0, 16, 32, 48])
cb.ax.set_xlabel("Δ F1 (pp)", fontsize=10.0, labelpad=4)
cb.ax.tick_params(labelsize=9.5, length=0, pad=3)
cb.outline.set_visible(False)
fig.canvas.draw()  # 先让 constrained_layout 定完位，再把色标宽度对齐到热力图
_pa, _pc = ax.get_position(), cb.ax.get_position()
fig.set_layout_engine("none")
_pos = [_pa.x0, _pc.y0, _pa.width, _pc.height]
cb.remove()  # 换成矢量渐变：fig.colorbar 会把它压成位图，放大就糊
cax = fig.add_axes(_pos)
_grad = np.linspace(V_MIN, V_MAX, 1024).reshape(1, -1)  # 1024 段，矢量色块，肉眼看不出台阶
_mesh = cax.pcolormesh(np.linspace(V_MIN, V_MAX, 1025), [0, 1], _grad, cmap=CMAP,
                       norm=TwoSlopeNorm(vcenter=0, vmin=V_MIN, vmax=V_MAX),
                       shading="flat", edgecolors="face", linewidth=0.4)
_mesh.set_antialiased(False)  # 关抗锯齿 + 关吸附：否则相邻色块之间会留下竖缝
_mesh.set_snap(False)
cax.set_xlim(V_MIN, V_MAX)
cax.set_ylim(0, 1)
cax.set_yticks([])
cax.set_xticks([-16, 0, 16, 32, 48])
cax.tick_params(labelsize=9.5, length=0, pad=3)
cax.set_xlabel("Δ F1 (pp)", fontsize=10.0, labelpad=4)
for _s in cax.spines.values():
    _s.set_visible(False)
bold_ticks(ax)  # 类别名与骨干名黑色加粗
bold_ticks(cax)  # 色标刻度数字也一样
save_fixed(fig, FIG_DIR / (PREFIX + NAME_B))

# ---------- 图(c) 逐类候选覆盖率：覆盖率越高条色越深 ----------
fig, ax = plt.subplots(figsize=(5.0, 3.7), constrained_layout=True)
colors = [CMAP_BAR(m / 100) for m in means]
ax.barh(ys, means, height=0.54, color=colors, edgecolor="none", zorder=1)  # 深色 = 覆盖率高
for y, m, c in zip(ys, means, colors):  # 数值放条内左端，按条色深浅自动选黑/白字
    dark = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2] < 0.55  # 感知亮度
    ax.text(2.2, y, f"{m:.1f}", va="center", ha="left", fontsize=9.6, weight="bold",
            color="white" if dark else C_TEXT, zorder=4)
ax.axvline(100, color=C_LINE, lw=1.0, ls=(0, (2, 2)), zorder=0)  # 100% 参考线
ax.set_yticks(ys)
ax.set_yticklabels(labels, weight="bold", color=C_TEXT)
ax.set_ylim(n - 0.5, -0.5)
ax.set_xlim(0, COVERAGE_XMAX)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_xlabel("Candidate coverage (%)")
ax.set_axisbelow(True)
bare(ax)
bold_ticks(ax)  # 刻度数字黑色加粗
save(fig, FIG_DIR / (PREFIX + NAME_C))
