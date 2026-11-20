# SLARD-31 类别级效应：三张独立图
# IDE 打开本文件点运行即可出图，只依赖 numpy + matplotlib，不依赖本目录其他 .py
# 输入路径、类别名、骨干名、配色、输出名全在下面配置区，改图只改这一段

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无界面后端：只写文件，不弹窗
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ==================== 配置区 ====================
DATA_DIR = (Path(__file__).resolve().parent / "outputs")  # 两张源 CSV 所在目录
FIG_DIR = (Path(__file__).resolve().parent / "figures")  # 三张图的输出目录
MEAN_CSV = DATA_DIR / "class_effects_SLARD-31.csv"  # 类别均值表：F1 / support / coverage
LONG_CSV = DATA_DIR / "class_effects_SLARD-31_by_backbone.csv"  # 逐骨干长表：图(b) 用
PREFIX = "fig_class_effect_SLARD-31"  # 三张图共用的文件名前缀

NAME_A = "_a_per_class_f1"  # 图(a) 逐类 F1 文件名
NAME_B = "_b_delta_by_backbone"  # 图(b) 各骨干 ΔF1 热力图文件名
NAME_C = "_c_coverage"  # 图(c) 候选覆盖率文件名

CLASS_NAMES = {  # 三张图左侧的类别名：class_id -> 显示文字
    0: "Environmental protection",
    1: "Energy",
    2: "Civil affairs",
    3: "Human rights",
    4: "Education",
    5: "Animal husbandry",
    6: "Industrial administration",
    7: "Legal affairs",
    8: "Land administration",
    9: "Water resources",
    10: "Intellectual property",
    11: "Forestry",
    12: "Sports",
    13: "Agriculture",
    14: "Real estate",
    15: "Tourism",
    16: "Cultural heritage and history",
    17: "Business administration",
    18: "Contracts",
    19: "National security",
    20: "Enterprises",
    21: "Commerce and materials",
    22: "Military affairs",
    23: "Advertising",
    24: "Surveying and mapping",
    25: "Business environment",
    26: "Labor and trade unions",
    27: "Pricing",
    28: "Fisheries",
    29: "Construction",
    30: "Culture",
}
BACKBONES = [  # 图(b) 横轴骨干列：(CSV 全称, 图上短名)，顺序即列顺序
    ("Qwen2.5-7B-Instruct", "Qwen2.5-7B"),
    ("Qwen3-8B", "Qwen3-8B"),
    ("DeepSeek-R1-0528-Qwen3-8B", "DeepSeek-R1"),
    ("GPT-5.4", "GPT-5.4"),
    ("Claude Sonnet 4.6", "Claude 4.6"),
    ("LegalOne-8B", "LegalOne-8B"),
]

COLOR_LIMIT = 50.0  # 图(b) 色标上下限 ±50 pp
NOISE_BAND = 5.0  # 图(a) 中 |ΔF1| 小于该值算噪声
PALE = 0.40  # 图(a) 噪声带内标记的透明度
COVERAGE_XMAX = 105.0  # 图(c) x 轴上限，留白让 100% 的点画完整
ROW_H = 0.235  # 每个类别占的画布高度（inch），31 行用较小值
YTICK_SIZE = 6.5  # 三张图左侧类别名字号
PAD = 0.13  # 三张图四周边距
LEGEND_LOC = "upper left"  # 图(a) 图例位置，31 行时放左上不挡数据

C_POS = "#2C6E8E"  # 图(a) 改善色：墨蓝
C_NEG = "#B4503F"  # 图(a) 下降色：砖红
C_COV = "#3E7C74"  # 图(c) 覆盖率点：墨绿
C_GRID = "#E4E0D8"  # 三张图的网格线 / 轴线
C_MARK = "#5A6570"  # 图(a) 图例标记色

CMAP = LinearSegmentedColormap.from_list(  # 图(b) 发散色表：红=下降 米白=0 蓝=改善
    "delta_soft",
    [(0.00, "#7E2E2A"), (0.18, "#C26A55"), (0.38, "#E8C9BC"),
     (0.50, "#F6F3ED"),
     (0.62, "#C8DAE1"), (0.82, "#6D9EB4"), (1.00, "#1E5572")],
)
CMAP.set_bad("#FFFFFF")  # 图(b) 缺数据的格子留白

plt.rcParams.update({  # 三张图共用的字体设置
    "font.family": "DejaVu Sans",
    "font.size": 9.0,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": YTICK_SIZE,
    "legend.fontsize": 8.5,
    "pdf.fonttype": 42,  # PDF 内嵌 TrueType，投稿友好
})
# ==============================================

# 读两张 CSV
with open(MEAN_CSV, encoding="utf-8") as f:
    mean = list(csv.DictReader(f))
with open(LONG_CSV, encoding="utf-8") as f:
    long = list(csv.DictReader(f))
for r in mean:  # 字符串转数字
    r["class_id"] = int(r["class_id"])
    r["support"] = int(r["support"])
    for k in ("f1_direct", "f1_full", "delta", "coverage"):
        r[k] = float(r[k]) if r[k] else float("nan")
for r in long:
    r["class_id"] = int(r["class_id"])
    r["delta"] = float(r["delta"])

n = len(mean)  # 类别数，决定画布高度
ys = np.arange(n)  # 三张图共用的 y 坐标
ids = [r["class_id"] for r in mean]  # 图(b) 用 class_id 定位行
labels = [  # 三张图左侧统一标签：类别名 + 测试集样本数
    f"{CLASS_NAMES.get(r['class_id'], r['class_name'])} ({r['support']})" for r in mean
]
FIG_DIR.mkdir(parents=True, exist_ok=True)  # 输出目录不存在就建

# ---------- 图(a) 逐类 F1 哑铃图 ----------
fig, ax = plt.subplots(figsize=(6.8, max(3.2, n * ROW_H + 1.0)), constrained_layout=True)
for y, r in zip(ys, mean):  # 一行一个类别
    c = C_POS if r["delta"] > 0 else C_NEG  # 线色：升蓝降红
    a = 1.0 if abs(r["delta"]) >= NOISE_BAND else PALE  # 噪声带内用淡色
    ax.plot([r["f1_direct"], r["f1_full"]], [y, y], color=c, lw=2.6,  # 连接 Direct 与 Full 的横线
            alpha=a, solid_capstyle="round", zorder=2)
    ax.plot(r["f1_direct"], y, "o", ms=6.5, mfc="white", mec=c, mew=1.5,  # 左端空心圆 = Direct
            alpha=a, zorder=3)
    ax.plot(r["f1_full"], y, "D", ms=5.6, color=c, alpha=a, zorder=4)  # 右端实心菱形 = Full
ax.set_yticks(ys)  # 左侧类别名
ax.set_yticklabels(labels)
ax.set_ylim(n - 0.55, -0.55)  # 按 Δ 降序，最大的在最上
ax.set_xlim(0, 100)  # 横轴 F1 百分比
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_xlabel("Class F1 (%)")  # 横轴标题
ax.grid(axis="x", color=C_GRID, lw=0.7)  # 竖向浅灰网格
ax.set_axisbelow(True)
for s in ("top", "right"):  # 去掉上、右边框
    ax.spines[s].set_visible(False)
for s in ax.spines.values():  # 剩余边框改浅灰细线
    s.set_color(C_GRID)
    s.set_linewidth(0.8)
ax.plot([], [], "o", ms=6, mfc="white", mec=C_MARK, label="Direct")  # 图例：Direct 空心圆
ax.plot([], [], "D", ms=5.2, color=C_MARK, label="Full")  # 图例：Full 菱形
ax.legend(frameon=False, loc=LEGEND_LOC, handletextpad=0.35, borderaxespad=0.2)
ax.set_title("(a) Per-class F1: Direct vs Full", loc="left", pad=9)  # 图(a) 标题
base = FIG_DIR / (PREFIX + NAME_A)  # 图(a) 输出路径
fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=PAD)  # 矢量版
fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=PAD)  # 预览版
plt.close(fig)
print("written", base.with_suffix(".pdf"))
print("written", base.with_suffix(".png"))

# ---------- 图(b) 各骨干 ΔF1 热力图 ----------
col = {full: j for j, (full, _) in enumerate(BACKBONES)}  # CSV 骨干全称 -> 列号
mat = np.full((n, len(BACKBONES)), np.nan)  # 行=类别，列=骨干
for r in long:  # 填入 ΔF1
    j = col.get(r["backbone"])
    if j is not None:
        mat[ids.index(r["class_id"]), j] = r["delta"]
fig, ax = plt.subplots(figsize=(7.4, max(3.8, n * ROW_H + 1.2)), constrained_layout=True)
im = ax.imshow(mat, cmap=CMAP, vmin=-COLOR_LIMIT, vmax=COLOR_LIMIT,  # 每格颜色 = ΔF1
               aspect="auto", extent=(0, len(BACKBONES), n - 0.5, -0.5),
               interpolation="nearest")
ax.set_yticks(np.arange(n))  # 左侧类别名，与图(a) 行对齐
ax.set_yticklabels(labels)
ax.set_ylim(n - 0.5, -0.5)
ax.set_xlim(0, len(BACKBONES))
ax.set_xticks(np.arange(len(BACKBONES)) + 0.5)  # 底部骨干短名
ax.set_xticklabels([short for _, short in BACKBONES], rotation=25, ha="right")
ax.set_xticks(np.arange(len(BACKBONES) + 1), minor=True)  # 格子之间的白色分隔线
ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
ax.grid(which="minor", color="white", lw=1.3)
ax.tick_params(which="minor", length=0)
ax.tick_params(which="major", length=0, pad=4)
for s in ax.spines.values():  # 热力图不加边框
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, orientation="horizontal",  # 下方色标：红=下降 蓝=改善
                  shrink=0.70, pad=0.03, aspect=32)
cb.set_label("Δ F1 (pp)")  # 色标标题
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=8)
ax.set_title("(b) Δ F1 by backbone", loc="left", pad=9)  # 图(b) 标题
base = FIG_DIR / (PREFIX + NAME_B)  # 图(b) 输出路径
fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=PAD)
fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=PAD)
plt.close(fig)
print("written", base.with_suffix(".pdf"))
print("written", base.with_suffix(".png"))

# ---------- 图(c) 逐类候选覆盖率 ----------
cov = [r["coverage"] if r["coverage"] == r["coverage"] else 0.0 for r in mean]  # 缺值记 0
fig, ax = plt.subplots(figsize=(6.6, max(3.2, n * ROW_H + 1.0)), constrained_layout=True)
ax.hlines(ys, 0, cov, color=C_GRID, lw=1.4, zorder=1)  # 从 0 到覆盖率值的引导线
ax.scatter(cov, ys, s=52, color=C_COV, edgecolors="white",  # 末端圆点 = 覆盖率数值
           linewidths=1.0, zorder=3)
ax.set_yticks(ys)  # 左侧类别名
ax.set_yticklabels(labels)
ax.set_ylim(n - 0.55, -0.55)
ax.set_xlim(0, COVERAGE_XMAX)  # 右端留白，100% 的点不被切
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_xlabel("Candidate coverage (%)")  # 横轴标题
ax.grid(axis="x", color=C_GRID, lw=0.7)  # 竖向浅灰网格
ax.set_axisbelow(True)
for s in ("top", "right", "left"):  # 去掉上、右、左边框
    ax.spines[s].set_visible(False)
for s in ax.spines.values():  # 底边改浅灰细线
    s.set_color(C_GRID)
    s.set_linewidth(0.8)
ax.tick_params(axis="y", length=0)
ax.set_title("(c) Candidate coverage", loc="left", pad=9)  # 图(c) 标题
base = FIG_DIR / (PREFIX + NAME_C)  # 图(c) 输出路径
fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=PAD)
fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=PAD)
plt.close(fig)
print("written", base.with_suffix(".pdf"))
print("written", base.with_suffix(".png"))
