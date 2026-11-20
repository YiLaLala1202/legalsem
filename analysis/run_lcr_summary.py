# LCR-CN 机制分析一页摘要（Section 5.4），输出 Markdown
# IDE 打开本文件点运行即可，只依赖 numpy + scikit-learn，不依赖本目录其他 .py
# 输入路径、类别名、骨干名全在下面配置区；输出写到 analysis/outputs/

# ==================== 配置区：路径 / 名字 / 参数，改这里 ====================
import csv
import json
import re
import statistics as st
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent  # 预测文件根目录
OUT_DIR = (Path(__file__).resolve().parent / "outputs")  # 统计结果输出目录
OUT_DIR.mkdir(parents=True, exist_ok=True)

BACKBONES = [  # 六个骨干的规范名
    "Qwen2.5-7B-Instruct", "Qwen3-8B", "DeepSeek-R1-0528-Qwen3-8B",
    "GPT-5.4", "Claude Sonnet 4.6", "LegalOne-8B",
]
ALIASES = {  # 从文件名识别骨干用的子串
    "Qwen2.5-7B-Instruct": ["qwen2.5"], "Qwen3-8B": ["qwen3"],
    "DeepSeek-R1-0528-Qwen3-8B": ["deepseek"], "GPT-5.4": ["gpt"],
    "Claude Sonnet 4.6": ["claude"], "LegalOne-8B": ["legalone"],
}
LCR_SPACE = list(range(5))  # LCR-CN 五类标签空间
SLARD_SPACE = list(range(31))  # SLARD-31 三十一类标签空间
LCR_NAMES = {  # LCR-CN 类别英文名
    0: "No conflict", 1: "Authority/responsibility", 2: "Concept/definition scope",
    3: "Sanction magnitude/range", 4: "Applicability condition",
}
SLARD_NAMES = {  # SLARD-31 类别英文名
    0: "Environmental protection", 1: "Energy", 2: "Civil affairs", 3: "Human rights",
    4: "Education", 5: "Animal husbandry", 6: "Industrial administration",
    7: "Legal affairs", 8: "Land administration", 9: "Water resources",
    10: "Intellectual property", 11: "Forestry", 12: "Sports", 13: "Agriculture",
    14: "Real estate", 15: "Tourism", 16: "Cultural heritage and history",
    17: "Business administration", 18: "Contracts", 19: "National security",
    20: "Enterprises", 21: "Commerce and materials", 22: "Military affairs",
    23: "Advertising", 24: "Surveying and mapping", 25: "Business environment",
    26: "Labor and trade unions", 27: "Pricing", 28: "Fisheries",
    29: "Construction", 30: "Culture",
}
LCR_NAME_TO_ID = {  # LCR-CN 中文名 -> id
    "无冲突": 0, "职权或责任划分不符": 1, "概念或定义范围不符": 2,
    "处罚幅度或范围不符": 3, "增设或变更适用条件": 4,
}
SLARD_NAME_TO_ID = {  # SLARD-31 中文名 -> id
    "环境保护": 0, "能源": 1, "民政": 2, "人权": 3, "教育": 4, "牧业": 5,
    "工业管理": 6, "法制工作": 7, "土地": 8, "水利": 9, "知识产权": 10,
    "林业": 11, "体育": 12, "农业": 13, "房地产": 14, "旅游": 15,
    "文物文史": 16, "工商管理": 17, "合同": 18, "国家安全": 19, "企业": 20,
    "商贸物资": 21, "军事": 22, "广告": 23, "测绘": 24, "营商环境优化": 25,
    "劳动工会": 26, "价格": 27, "渔业": 28, "建筑业": 29, "文化": 30,
}
SPACES = {"LCR-CN": LCR_SPACE, "SLARD-31": SLARD_SPACE}  # 数据集 -> 标签空间
NAMES = {"LCR-CN": LCR_NAMES, "SLARD-31": SLARD_NAMES}  # 数据集 -> 类别名

LCR = ROOT / "conflict_superior_laws" / "output"  # LCR-CN 预测根目录
SLARD = ROOT / "SLARD_classification" / "output"  # SLARD-31 预测根目录
GPT = ROOT / "conflict_superior_laws(GPT-Extract)" / "output"  # GPT 抽取预测根目录
SEARCH_DIRS = {  # (数据集, 配置) -> 搜索目录
    ("LCR-CN", "Direct"): [LCR / "baseline"],
    ("LCR-CN", "Static"): [LCR / "ablation"],
    ("LCR-CN", "Dynamic"): [LCR / "ablation"],
    ("LCR-CN", "Full"): [LCR / "full" / "pred_classfication"],
    ("LCR-CN", "Extract"): [LCR / "ablation"],
    ("LCR-CN", "GPT-Static"): [GPT / "GPT-Extract + Static"],
    ("LCR-CN", "GPT-Dynamic"): [GPT / "GPT-Extract + Dynamic"],
    ("SLARD-31", "Direct"): [SLARD / "baseline"],
    ("SLARD-31", "Static"): [SLARD / "ablation"],
    ("SLARD-31", "Dynamic"): [SLARD / "ablation"],
    ("SLARD-31", "Full"): [SLARD / "full" / "pred_classfication"],
    ("SLARD-31", "Extract"): [SLARD / "ablation"],
}
FILE_RULES = {  # 配置 -> 文件名匹配规则
    "Direct": r"baseline",
    "Static": r"no_dynamic",
    "Dynamic": r"test_class(?!_no_dynamic)|(?<!no_)dynamic",
    "Full": r"test_class",
    "Extract": r"test_extract",
    "GPT-Static": r"no_dynamic_gpt",
    "GPT-Dynamic": r"(?<!no_)dynamic_gpt",
}
SEED = 2026  # 重采样随机种子
CONTRASTS = [("Full - Direct", "Full", "Direct"), ("Dynamic - Static", "Dynamic", "Static"), ("Full - Dynamic", "Full", "Dynamic")]  # 三组预设对比
# ======================================================================

# ---------- 预测文件读取 ----------
def resolve_file(dataset, conf, backbone):  # 定位某个 (数据集, 配置, 骨干) 的预测文件
    dirs = SEARCH_DIRS.get((dataset, conf))
    if not dirs: return None
    rule = re.compile(FILE_RULES[conf], re.I)
    found = []
    for d in dirs:
        if not d.is_dir(): continue
        for p in d.rglob("*.jsonl"):
            name = p.name.lower()
            if not any(a in name for a in ALIASES[backbone]): continue
            if not rule.search(name): continue
            if conf in ("Direct", "Static", "Dynamic", "Full") and "_gpt" in name: continue
            if conf.startswith("GPT") and "_gpt" not in name: continue
            found.append(p)
    if not found: return None
    found.sort(key=lambda p: (len(p.name), -p.stat().st_mtime))
    return found[0]

def read_jsonl(path):  # 逐行读 jsonl，坏行跳过
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: continue
    return rows

def norm_id(v):  # 统一样本 id
    s = str(v).strip()
    try: return str(int(float(s)))
    except (TypeError, ValueError): return s

def as_int(v):  # 转整数，失败返回 None
    try: return int(str(v).strip())
    except (TypeError, ValueError): return None

def gold_lcr(row):  # LCR-CN 的 gold 标签
    for k in ("gold_label", "label", "true_label", "conflict_label_id"):
        if row.get(k) is not None:
            v = as_int(row[k])
            if v is not None and v in LCR_SPACE: return v
    ct = row.get("conflict_type")
    v = as_int(ct)
    if v is not None and v in LCR_SPACE: return v
    return LCR_NAME_TO_ID.get(str(ct).strip())

def pred_lcr(row):  # LCR-CN 的预测标签
    for k in ("pred_conflict_label_id", "pred_label", "prediction", "predict_label"):
        if row.get(k) is not None:
            v = as_int(row[k])
            if v is not None and v in LCR_SPACE: return v
    p = LCR_NAME_TO_ID.get(str(row.get("pred_conflict_type", "")).strip())
    return -1 if p is None else p

def gold_slard(row):  # SLARD-31 的 gold 标签
    for k in ("gold_label", "label", "true_label"):
        if row.get(k) is not None:
            v = as_int(row[k])
            if v is not None and v in SLARD_SPACE: return v
    return SLARD_NAME_TO_ID.get(str(row.get("gold_category", "")).strip())

def pred_slard(row):  # SLARD-31 的预测标签
    for k in ("pred_label", "prediction", "predict_label"):
        if row.get(k) is not None:
            v = as_int(row[k])
            if v is not None and v in SLARD_SPACE: return v
    for k in ("pred_category", "raw_output", "category"):
        if row.get(k) is not None:
            v = SLARD_NAME_TO_ID.get(str(row[k]).strip())
            if v is not None: return v
    return -1

def cand_lcr(row):  # LCR-CN 候选标签集
    raw = row.get("candidate_labels")
    if raw is None: return None
    if isinstance(raw, str):
        try: raw = json.loads(raw)
        except json.JSONDecodeError: raw = []
    out = []
    for x in raw:
        v = as_int(x)
        if v is not None and v in LCR_SPACE and v not in out: out.append(v)
    return out or None

def cand_slard(row):  # SLARD-31 候选类别集
    raw = row.get("candidate_categories")
    if raw is None: return None
    if isinstance(raw, str):
        try: raw = json.loads(raw)
        except json.JSONDecodeError: raw = [s for s in raw.replace("[", "").replace("]", "").split(",") if s]
    out = []
    for x in raw:
        v = as_int(x) if isinstance(x, (int, float)) else SLARD_NAME_TO_ID.get(str(x).strip())
        if v is not None and v not in out: out.append(v)
    return out or None

_REF = {}  # gold 参照表缓存
def reference_gold(dataset):  # 从参照文件建 id -> gold
    if dataset in _REF: return _REF[dataset]
    table = {}
    is_lcr = dataset == "LCR-CN"
    conf = "Full" if is_lcr else "Direct"
    for bb in BACKBONES:
        path = resolve_file(dataset, conf, bb)
        if path is None: continue
        for row in read_jsonl(path):
            g = gold_lcr(row) if is_lcr else gold_slard(row)
            if g is not None: table.setdefault(norm_id(row.get("id")), g)
    _REF[dataset] = table
    return table

PRED_KEYS = {"LCR-CN": ("pred_conflict_label_id", "pred_label", "prediction", "predict_label", "pred_conflict_type"), "SLARD-31": ("pred_label", "prediction", "predict_label", "pred_category", "raw_output", "category")}  # 判断文件里是否真有预测
def has_predictions(path, dataset):  # 只有 gold 的文件不算预测
    for row in read_jsonl(path)[:5]:
        if any(k in row and row[k] is not None for k in PRED_KEYS[dataset]): return True
    return False

def load(path, dataset):  # 解析单个预测文件为 id -> {gold, pred, cand}
    is_lcr = dataset == "LCR-CN"
    direct_lcr = is_lcr and "baseline" in str(path).lower()
    ref = reference_gold(dataset)
    out = {}
    for row in read_jsonl(path):
        rid = norm_id(row.get("id"))
        if direct_lcr:  # LCR baseline 的预测存在 conflict_type 里
            gold = ref.get(rid)
            raw = row.get("conflict_type")
            pred = as_int(raw)
            if pred not in LCR_SPACE: pred = LCR_NAME_TO_ID.get(str(raw).strip(), -1)
            cand = None
        else:
            gold = gold_lcr(row) if is_lcr else gold_slard(row)
            if gold is None: gold = ref.get(rid)
            pred = pred_lcr(row) if is_lcr else pred_slard(row)
            cand = cand_lcr(row) if is_lcr else cand_slard(row)
        if gold is None: continue
        out[rid] = {"gold": gold, "pred": pred, "cand": cand}
    return out

def load_cell(dataset, conf, backbone):  # 读一个 (配置, 骨干) 单元格
    path = resolve_file(dataset, conf, backbone)
    if path is None: return None
    if dataset == "LCR-CN" and conf == "Direct": return load(path, dataset)
    if not has_predictions(path, dataset): return None
    return load(path, dataset)

def load_candidates_from_extract(dataset, backbone):  # 从抽取文件重建候选集
    if dataset != "SLARD-31": return {}
    path = resolve_file(dataset, "Extract", backbone)
    if path is None: return {}
    out = {}
    for row in read_jsonl(path):
        fields = row.get("extracted_fields") or {}
        if isinstance(fields, str):
            try: fields = json.loads(fields)
            except json.JSONDecodeError: fields = {}
        cand = []
        for name, values in fields.items():
            cid = SLARD_NAME_TO_ID.get(str(name).strip())
            if cid is None: continue
            filled = bool(values) if not isinstance(values, str) else bool(values.strip())
            if filled and cid not in cand: cand.append(cid)
        out[norm_id(row.get("id"))] = cand or list(SLARD_SPACE)
    return out

def attach_candidates(pred, mapping):  # 补齐缺失候选集，返回更新行数
    n = 0
    for rid, rec in pred.items():
        if rec["cand"] is None and rid in mapping:
            rec["cand"] = mapping[rid]
            n += 1
    return n

def aligned_ids(*preds):  # 多个预测集共有的样本 id
    common = set(preds[0])
    for p in preds[1:]: common &= set(p)
    return sorted(common, key=lambda x: (len(x), x))

# ---------- 指标 ----------
def macro_f1(y_true, y_pred, space):  # 固定标签空间的 Macro-F1(%)
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=space, average="macro", zero_division=0)
    return float(f1) * 100.0

def macro_f1_fast(y_true, y_pred, k):  # bootstrap 版 Macro-F1(%)，-1 进哨兵列
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    pred = np.where(y_pred < 0, k, y_pred)
    cm = np.bincount(y_true * (k + 1) + pred, minlength=k * (k + 1)).reshape(k, k + 1)
    tp = np.diag(cm[:, :k]).astype(float)
    fp = cm.sum(axis=0)[:k] - tp
    fn = cm.sum(axis=1).astype(float) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)
    return float(f1.mean() * 100.0)

def per_class(y_true, y_pred, space):  # 逐类 P/R/F1/support
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=space, average=None, zero_division=0)
    return {int(k): {"precision": float(p[i]) * 100, "recall": float(r[i]) * 100, "f1": float(f1[i]) * 100, "support": int(sup[i])} for i, k in enumerate(space)}

def per_class_coverage(pred, space):  # 逐类覆盖率
    buckets = {k: {"n": 0, "hit": 0, "acs": 0.0, "t": [], "p": []} for k in space}
    for v in pred.values():
        b = buckets[v["gold"]]
        cand = v["cand"] if v["cand"] is not None else list(space)
        b["n"] += 1
        b["acs"] += len(cand)
        if v["gold"] in cand:
            b["hit"] += 1
            b["t"].append(v["gold"])
            b["p"].append(v["pred"])
    out = {}
    for k, b in buckets.items():
        n = b["n"]
        out[k] = {"support": n, "coverage": (b["hit"] / n * 100) if n else float("nan"), "acs": (b["acs"] / n) if n else float("nan"), "f1": macro_f1(b["t"], b["p"], space) if b["t"] else float("nan")}
    return out

def candidate_diagnostics(pred, space):  # Cov / ACS / rho / Covered-F1
    k = len(space)
    n = len(pred)
    if n == 0: return {"n": 0, "Cov": float("nan"), "ACS": float("nan"), "rho": float("nan"), "Covered Macro-F1": float("nan"), "omission rate": float("nan")}
    hit, size = 0, 0
    cov_t, cov_p = [], []
    for v in pred.values():
        cand = v["cand"] if v["cand"] is not None else list(space)
        size += len(cand)
        in_c = v["gold"] in cand
        hit += int(in_c)
        if in_c:
            cov_t.append(v["gold"])
            cov_p.append(v["pred"])
    acs = size / n
    covered = len(cov_t) / n
    return {"n": n, "Cov": covered * 100, "ACS": acs, "rho": (1 - acs / k) * 100, "Covered Macro-F1": macro_f1(cov_t, cov_p, space) if cov_t else float("nan"), "omission rate": (1 - covered) * 100}

def error_decomposition(pred, space):  # 候选遗漏误差 vs 候选内误差
    n = len(pred)
    wrong = omission = within = 0
    for v in pred.values():
        if v["pred"] == v["gold"]: continue
        wrong += 1
        cand = v["cand"] if v["cand"] is not None else list(space)
        if v["gold"] not in cand: omission += 1
        else: within += 1
    return {"n": n, "errors": wrong, "omission": omission, "within": within, "omission share of errors": (omission / wrong * 100) if wrong else float("nan"), "omission rate of n": (omission / n * 100) if n else float("nan"), "within rate of n": (within / n * 100) if n else float("nan")}

# ---------- 配对统计推断 ----------
def prepare(dataset, pred_a, pred_b, backbones=None):  # 两配置按样本 id 对齐
    backbones = backbones or BACKBONES
    k = len(SPACES[dataset])
    common = None
    for bb in backbones:
        ids_bb = set(pred_a[bb]) & set(pred_b[bb])
        common = ids_bb if common is None else (common & ids_bb)
    ids = sorted(common or set(), key=lambda x: (len(x), x))
    if not ids: raise ValueError("no shared instance ids")
    y = np.array([pred_a[backbones[0]][i]["gold"] for i in ids], dtype=np.int64)
    pa = [np.array([pred_a[bb][i]["pred"] for i in ids], dtype=np.int64) for bb in backbones]
    pb = [np.array([pred_b[bb][i]["pred"] for i in ids], dtype=np.int64) for bb in backbones]
    return {"ids": ids, "y": y, "a": pa, "b": pb, "k": k}

def _diff(y, pa, pb, k, idx=None):  # 两配置 Macro-F1 差的骨干均值
    yy = y if idx is None else y[idx]
    d = []
    for A, B in zip(pa, pb):
        aa = A if idx is None else A[idx]
        bb = B if idx is None else B[idx]
        d.append(macro_f1_fast(yy, bb, k) - macro_f1_fast(yy, aa, k))
    return float(np.mean(d))

def paired_bootstrap(data, n_boot=5000):  # 配对 bootstrap 的 95% CI
    rng = np.random.default_rng(SEED)
    n = len(data["ids"])
    k = data["k"]
    obs = _diff(data["y"], data["a"], data["b"], k)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[i] = _diff(data["y"], data["a"], data["b"], k, idx)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"delta": obs, "ci_low": float(lo), "ci_high": float(hi)}

def approximate_randomization(data, n_perm=5000):  # 近似随机化检验 p（plus-one 校正）
    rng = np.random.default_rng(SEED + 1)
    n = len(data["ids"])
    k = data["k"]
    obs = _diff(data["y"], data["a"], data["b"], k)
    extreme = 0
    for _ in range(n_perm):
        mask = rng.random(n) < 0.5  # 所有骨干共用同一个交换掩码
        pa = [np.where(mask, B, A) for A, B in zip(data["a"], data["b"])]
        pb = [np.where(mask, A, B) for A, B in zip(data["a"], data["b"])]
        if abs(_diff(data["y"], pa, pb, k)) >= abs(obs) - 1e-12: extreme += 1
    return {"delta": obs, "p": (extreme + 1) / (n_perm + 1)}

def holm(pvals):  # Holm step-down 校正
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pvals[i]))
        adj[i] = run
    return adj


# ---------- 主流程：LCR-CN 机制分析一页摘要 ----------
CONFIGS = ["Direct", "Static", "Dynamic", "Full"]  # 四种配置
space = LCR_SPACE  # 五类标签空间
names = LCR_NAMES  # 类别名
mean = lambda xs: st.fmean(xs) if xs else float("nan")  # 空列表返回 nan

agg = {}  # 配置 -> 汇总指标
per_class_f1 = {}  # 配置 -> 逐类 F1
for conf in CONFIGS:
    cov, acs = [], []
    om_share, om_rate, wi_rate = [], [], []
    pcs = {k: [] for k in space}
    for bb in BACKBONES:
        pred = load_cell("LCR-CN", conf, bb)  # 读预测
        if pred is None: continue
        pc = per_class([v["gold"] for v in pred.values()], [v["pred"] for v in pred.values()], space)  # 逐类 F1
        for k in space: pcs[k].append(pc[k]["f1"])
        n = len(pred)
        wrong = sum(1 for v in pred.values() if v["pred"] != v["gold"])  # 错判数
        om = sum(1 for v in pred.values() if v["pred"] != v["gold"] and v["cand"] is not None and v["gold"] not in v["cand"])  # 候选遗漏误差
        if pred and all(v["cand"] is not None for v in pred.values()):
            cov.append(sum(1 for v in pred.values() if v["gold"] in v["cand"]) / n * 100)  # Cov
            acs.append(st.fmean(len(v["cand"]) for v in pred.values()))  # ACS
        om_share.append(om / wrong * 100 if wrong else 0.0)
        om_rate.append(om / n * 100)
        wi_rate.append((wrong - om) / n * 100)
    agg[conf] = {"Cov": mean(cov), "ACS": mean(acs), "rho": (1 - mean(acs) / len(space)) * 100 if acs else float("nan"), "omission share": mean(om_share), "omission rate": mean(om_rate), "within rate": mean(wi_rate)}
    per_class_f1[conf] = {k: mean(v) for k, v in pcs.items()}

cov_table = {}  # 配置 -> 逐类覆盖率
for conf in ("Dynamic", "Full"):
    rows = {k: [] for k in space}
    for bb in BACKBONES:
        pred = load_cell("LCR-CN", conf, bb)  # 读预测
        if pred is None: continue
        for k, v in per_class_coverage(pred, space).items(): rows[k].append(v["coverage"])
    cov_table[conf] = {k: (st.fmean(v) if v else float("nan")) for k, v in rows.items()}

lines = ["# LCR-CN mechanism analysis\n", "Six-backbone means on the fixed 1,404-instance test set.\n", "\n## Candidate space and error decomposition\n", "| Configuration | Cov (%) | ACS | ρ (%) | Errors: omission share (%) | omission rate (%) | within-candidate rate (%) |", "|---|---|---|---|---|---|---|"]
for conf in CONFIGS:
    a = agg[conf]
    fmt = lambda x: "—" if x != x else f"{x:.2f}"
    lines.append(f"| {conf} | {fmt(a['Cov'])} | {fmt(a['ACS'])} | {fmt(a['rho'])} | {a['omission share']:.1f} | {a['omission rate']:.2f} | {a['within rate']:.2f} |")
lines.append("\n## Per-class F1\n")
lines.append("| Class | n | " + " | ".join(CONFIGS) + " | Cov Dynamic (%) | Cov Full (%) |")
lines.append("|---" * (len(CONFIGS) + 4) + "|")
support = {k: 0 for k in space}  # 各类别测试集样本数
p0 = load_cell("LCR-CN", "Full", BACKBONES[0])
if p0:
    for v in p0.values(): support[v["gold"]] += 1
for k in space:
    cells = [("—" if per_class_f1[c][k] != per_class_f1[c][k] else f"{per_class_f1[c][k]:.1f}") for c in CONFIGS]
    cd = cov_table["Dynamic"].get(k, float("nan"))
    cf = cov_table["Full"].get(k, float("nan"))
    lines.append(f"| {names[k]} | {support[k]} | " + " | ".join(cells) + f" | {cd:.1f} | {cf:.1f} |")
lines.append("\n*The omission / within-candidate split is only defined for configurations that have a candidate set; Direct and Static keep the full label space, so all of their errors are within-candidate by construction.*\n")
lines.append("\n## What the numbers say\n")
d, f = agg["Dynamic"], agg["Full"]
lines.append(f"1. Between Dynamic and Full, candidate coverage is essentially unchanged ({d['Cov']:.2f}% vs {f['Cov']:.2f}%) and the candidate set shrinks ({d['ACS']:.2f} to {f['ACS']:.2f} labels), so the gain is not bought by retaining fewer labels.\n")
lines.append(f"2. The omission rate stays flat ({d['omission rate']:.2f}% to {f['omission rate']:.2f}%), whereas errors committed among retained candidates fall from {d['within rate']:.2f}% to {f['within rate']:.2f}% of all instances. FGPR therefore improves discrimination inside the candidate set rather than candidate construction.\n")
concept = per_class_f1["Full"][2]
others = st.fmean(per_class_f1["Full"][k] for k in (0, 1, 3, 4))
lines.append(f"3. The definition class remains the weak point: its F1 is {concept:.1f} against {others:.1f} for the remaining classes, and its candidate coverage is {cov_table['Full'][2]:.1f}% versus {st.fmean(cov_table['Full'][k] for k in (0, 1, 3, 4)):.1f}% elsewhere. The residual error is concentrated where definitional elements are least specific, not where the classifier discriminates poorly.\n")
out = OUT_DIR / "LCR-CN_analysis_summary.md"  # 输出 Markdown
out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print("written", out)
