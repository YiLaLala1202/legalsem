# 论文 motivation 图：表面匹配 vs 条件对齐（左右对照卡片式）
# IDE 点运行直接出 PDF+SVG，只依赖 matplotlib，不依赖本目录其他 .py
# 所有文字、颜色、坐标都在下面配置区，改图只改这一段

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ==================== 配置区 ====================
FIG_DIR = (Path(__file__).resolve().parent / "figures")  # 输出目录
NAME = "fig_motivation"  # 输出文件名（.pdf + .png）
SIZE = (10.4, 7.5)  # 画布尺寸（inch）
DPI = 400  # PNG 分辨率

INK = "#1F2328"  # 主文字
MUTED = "#5C636B"  # 次要文字
RULE = "#D9D5C9"  # 分隔线 / 浅边框
PAPER = "#FAF9F5"  # 通栏面板底色
RED, RED_FILL = "#B4483C", "#FBEAE6"  # 左栏：问题路径
GREEN, GREEN_FILL = "#4A8A5C", "#EAF3EC"  # 右栏：正确路径
BLUE, BLUE_FILL = "#3C6E9E", "#E8F0F7"  # 条件公式 / 对齐项
ORANGE, ORANGE_FILL = "#B5652F", "#FAE7D8"  # 新增条件 R / Condition changed

INTRO_TEXT = "A subordinate provision adds a new precondition (R) to the condition (A) inherited from its superior provision."  # 图首总起句

# ---- 顶部两条法条 ----
SUP_TITLE = "Superior provision"  # 上方法条标签
SUB_TITLE = "Subordinate provision"  # 下方法条标签
SUP_Q1 = "… it fails to implement the measures stated in its reply"  # 上方法条第一行
SUP_Q2 = "without explaining the reasons …"  # 上方法条第二行
SUB_Q1 = "… fails to implement the measures stated in its reply without explanation,"  # 下方法条第一行
SUB_Q2 = "and still fails to take corrective action after two follow-up requests …"  # 新增条件 R 的文本依据

# ---- 左栏：表面匹配 ----
LEFT_TITLE = "Surface matching"
LEFT_S1 = "Both texts describe\nthe same procedural duty"
LEFT_HL = "Wording is highly similar"
LEFT_S3 = "The added precondition R\nis never inspected"
LEFT_CONC = "Predicted: No conflict"

# ---- 右栏：条件对齐 ----
RIGHT_TITLE = "Condition alignment"
RIGHT_S1 = "Recover the conditions\nbehind the wording"
RIGHT_HL = "$C_{\\mathrm{sup}} = A$      $C_{\\mathrm{sub}} = A \\wedge R$"
RIGHT_S3 = "R is isolated as a\nnew applicable condition"
RIGHT_CONC = "Predicted: Addition or modification"

# ---- 底部：语义要素对齐 ----
ALIGN_TITLE = "Semantic element alignment"
ALIGN_NOTE = "A: shared trigger  ·  R: added precondition"
ALIGN_ROWS = [  # 两行三列：属性名 + 判定值（六个要素名固定，不改写）
    [("Subject", "aligned", BLUE, BLUE_FILL), ("Action_Behavior", "aligned", BLUE, BLUE_FILL), ("Responsibility", "aligned", BLUE, BLUE_FILL)],  # 完全对齐
    [("Definition", "not specific", MUTED, "#ECEAE3"), ("Sanction", "not specific", MUTED, "#ECEAE3"), ("Condition", "changed", ORANGE, ORANGE_FILL)],  # 后两者不具体，条件改变
]

# ---- 最底：金标准冲突类型 ----
GOLD_LABEL = "Gold conflict type"
GOLD_TEXT = "Addition or modification of applicable conditions"
# ================================================

plt.rcParams.update({  # 论文衬线字体 + 可编辑 PDF
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
})


def box(x, y, w, h, fc, ec, lw=1.0, r=1.2, z=2):  # 画圆角矩形
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))


def arrow(x, y1, y2):  # 画向下的细箭头
    ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>",
                                 mutation_scale=7, lw=0.8, color=MUTED, zorder=3))


fig, ax = plt.subplots(figsize=SIZE)  # 画布
ax.set_xlim(0, 104)
ax.set_ylim(0, 74.5)
ax.axis("off")
fig.patch.set_facecolor("white")

ax.text(52, 72.4, INTRO_TEXT, fontsize=9.6, color=INK, ha="center", va="center")  # 图首总起句

# ---------- 顶部：两条法条并排 ----------
box(3, 56.5, 98, 12.5, PAPER, RULE, 0.9, 1.4)  # 通栏圆角框
ax.text(5.5, 66.2, SUP_TITLE, fontsize=9.2, weight="bold", color=INK, va="center")  # 上方法条标签
ax.text(5.5, 63.6, SUP_Q1, fontsize=8.0, style="italic", color=MUTED, va="center")  # 上方法条第一行
ax.text(5.5, 61.6, SUP_Q2, fontsize=8.0, style="italic", color=MUTED, va="center")  # 上方法条第二行
ax.plot([51.8, 51.8], [57.3, 68.0], color=RULE, lw=0.8, zorder=3)  # 中间分隔线
ax.text(53.6, 66.2, SUB_TITLE, fontsize=9.2, weight="bold", color=INK, va="center")  # 下方法条标签
ax.text(53.6, 63.6, SUB_Q1, fontsize=8.0, style="italic", color=MUTED, va="center")  # 下方法条第一行
ax.text(53.6, 61.6, SUB_Q2, fontsize=8.0, style="italic", color=ORANGE, weight="bold", va="center")  # 新增条件 R 的文本依据

arrow(26.5, 56.2, 54.2)  # 分叉到左栏
arrow(77.5, 56.2, 54.2)  # 分叉到右栏

# ---------- 左栏：只看文字相似 ----------
box(3, 21.5, 47, 32.5, "white", RED, 1.3, 1.6)  # 左栏红框
ax.text(26.5, 50.8, LEFT_TITLE, fontsize=12, weight="bold", color=INK, ha="center", va="center")  # 左栏标题
ax.plot([6.5, 46.5], [48.9, 48.9], color=RED, lw=0.9, alpha=0.45)  # 标题下细线
ax.text(26.5, 45.0, LEFT_S1, fontsize=9.3, color=INK, ha="center", va="center", linespacing=1.4)  # 第一步
arrow(26.5, 42.4, 40.2)
box(8.5, 35.4, 35, 4.2, RED_FILL, "none", 0, 0.7)  # 高亮块
ax.text(26, 37.5, LEFT_HL, fontsize=9.3, style="italic", color=RED, ha="center", va="center")  # 误判依据
arrow(26.5, 33.8, 31.8)
ax.text(26.5, 29.3, LEFT_S3, fontsize=9.3, color=MUTED, ha="center", va="center", linespacing=1.4)  # 被忽略的部分
box(6.5, 22.4, 40, 3.7, RED_FILL, RED, 0.8, 0.7)  # 左栏结论
ax.text(26.5, 24.3, LEFT_CONC, fontsize=10, weight="bold", color=RED, ha="center", va="center")

# ---------- 右栏：还原条件结构 ----------
box(54, 21.5, 47, 32.5, "white", GREEN, 1.3, 1.6)  # 右栏绿框
ax.text(77.5, 50.8, RIGHT_TITLE, fontsize=12, weight="bold", color=INK, ha="center", va="center")  # 右栏标题
ax.plot([57.5, 97.5], [48.9, 48.9], color=GREEN, lw=0.9, alpha=0.45)  # 标题下细线
ax.text(77.5, 45.0, RIGHT_S1, fontsize=9.3, color=INK, ha="center", va="center", linespacing=1.4)  # 第一步
arrow(77.5, 42.4, 40.2)
box(59.5, 35.4, 36, 4.2, BLUE_FILL, "none", 0, 0.7)  # 高亮块
ax.text(77.5, 37.5, RIGHT_HL, fontsize=10.4, color=BLUE, ha="center", va="center")  # 两条适用条件
arrow(77.5, 33.8, 31.8)
ax.text(77.5, 29.3, RIGHT_S3, fontsize=9.3, color=MUTED, ha="center", va="center", linespacing=1.4)  # 新增条件被识别
box(57.5, 22.4, 40, 3.7, GREEN_FILL, GREEN, 0.8, 0.7)  # 右栏结论
ax.text(77.5, 24.3, RIGHT_CONC, fontsize=10, weight="bold", color=GREEN, ha="center", va="center")

# ---------- 底部：语义要素对齐面板 ----------
box(3, 11, 98, 9.4, PAPER, RULE, 0.9, 1.2)  # 通栏面板
ax.text(5.5, 18.6, ALIGN_TITLE, fontsize=9.2, weight="bold", color=INK, va="center")  # 面板标题
ax.text(98, 18.6, ALIGN_NOTE, fontsize=7.8, color=MUTED, ha="right", va="center")  # A / R 含义
for r, row in enumerate(ALIGN_ROWS):  # 两行
    y = 16.2 - r * 2.9
    for c, (label, value, color, fill) in enumerate(row):  # 每行三列
        x = 5.5 + c * 31
        ax.text(x, y, label, fontsize=8.5, color=MUTED, va="center")  # 属性名
        ax.text(x + 15, y, value, fontsize=8.5, weight="bold", color=color, va="center",
                bbox=dict(boxstyle="round,pad=0.32", fc=fill, ec="none"))  # 判定值小标签
arrow(52, 10.7, 8.8)  # 指向金标准

# ---------- 最底：金标准冲突类型 ----------
box(22, 2.4, 60, 6, ORANGE_FILL, ORANGE, 1.0, 0.9)  # 结论条
ax.text(52, 6.5, GOLD_LABEL, fontsize=8.0, color=MUTED, ha="center", va="center")  # 小标题
ax.text(52, 4.2, GOLD_TEXT, fontsize=10.5, weight="bold", color=INK, ha="center", va="center")  # 金标准

FIG_DIR.mkdir(parents=True, exist_ok=True)  # 输出目录不存在就建
out = FIG_DIR / NAME
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)  # 矢量版
fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.05)  # 矢量可编辑版
plt.close(fig)
print("written", out.with_suffix(".pdf"))
print("written", out.with_suffix(".svg"))
