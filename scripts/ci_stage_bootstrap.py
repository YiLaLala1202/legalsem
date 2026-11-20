"""Compute REAL per-stage bootstrap CIs and paired stage-contrast CIs.

New standalone script (demo phase). It does NOT modify any existing figure or
script. Predictions are read directly from the stored held-out files; no model
calls and no invented intervals.

Protocol: 5,000 bootstrap resamples of the held-out instances, seed 2026, one
shared index vector for all six backbones within a resample (paired over
backbones), Macro-F1 recomputed over the fixed label space of each task. This
matches the inferential protocol described in Section 4.7 of the manuscript.

Outputs: scripts/outputs/stage_ci_real.csv
"""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "outputs" / "stage_ci_real.csv"
SEED = 2026
N_BOOT = 5000

L2I = {
    "无冲突": 0,
    "职权或责任划分不符": 1,
    "概念或定义范围不符": 2,
    "处罚幅度或范围不符": 3,
    "增设或变更适用条件": 4,
}
BACKBONES = ["Qwen2.5-7B", "Qwen3-8B", "DeepSeek-R1", "GPT-5.4",
             "Claude", "LegalOne-8B"]


def rows(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8-sig") if l.strip()]


def as_int(x, name2id: dict | None = None) -> int:
    if x is None:
        return -1
    s = str(x).strip()
    try:
        return int(s)
    except ValueError:
        pass
    if name2id and s in name2id:
        return name2id[s]
    return L2I.get(s, -1)


def canon(stem: str) -> str | None:
    s = stem.lower()
    if "deepseek" in s:
        return "DeepSeek-R1"
    if "qwen2.5" in s:
        return "Qwen2.5-7B"
    if "qwen3" in s:
        return "Qwen3-8B"
    if "gpt" in s:
        return "GPT-5.4"
    if "claude" in s:
        return "Claude"
    if "legalone" in s or "legalaone" in s:
        return "LegalOne-8B"
    return None


def macro_f1_fast(y: np.ndarray, p: np.ndarray, k: int) -> float:
    """Macro-F1 (%) with invalid predictions mapped to an extra column."""
    pred = np.where(p < 0, k, p)
    cm = np.bincount(y * (k + 1) + pred, minlength=k * (k + 1)).reshape(k, k + 1)
    tp = np.diag(cm[:, :k]).astype(float)
    fp = cm[:, :k].sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    den = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, den, out=np.zeros(k), where=den > 0)
    return float(f1.mean() * 100.0)


# --------------------------------------------------------------------------- #
# 1. Build the gold reference and the 31-class name -> id map
# --------------------------------------------------------------------------- #
LCR_GOLD: dict[str, int] = {}
for path in glob.glob(str(ROOT / "outputs/lcr-cn/full/pred_classification/*_test_class.jsonl")):
    if "_v2" in path:
        continue
    for r in rows(path):
        v = r.get("gold_label")
        LCR_GOLD.setdefault(str(r["id"]).strip(), as_int(v if v is not None else r.get("conflict_type")))

SLARD_GOLD: dict[str, int] = {}
NAME2ID: dict[str, int] = {}
for path in glob.glob(str(ROOT / "outputs/slard-31/baseline/*_baseline_output.jsonl")):
    for r in rows(path):
        rid = str(r["id"]).strip()
        SLARD_GOLD.setdefault(rid, int(r["gold_label"]))
        NAME2ID.setdefault(str(r["gold_category"]).strip(), int(r["gold_label"]))

# --------------------------------------------------------------------------- #
# 2. Collect per-(dataset, stage, backbone) prediction vectors
# --------------------------------------------------------------------------- #
PATTERNS = {
    "LCR-CN": {
        "Direct": "outputs/lcr-cn/baseline/*_baseline_output.jsonl",
        "Static": "outputs/lcr-cn/ablation/*/*_test_class_no_dynamic.jsonl",
        "Dynamic": "outputs/lcr-cn/ablation/*/*_test_class*dynamic*.jsonl",
        "Dynamic2": "outputs/lcr-cn/ablation/*/*_test_class.jsonl",
        "Full": "outputs/lcr-cn/full/pred_classification/*_test_class.jsonl",
    },
    "SLARD-31": {
        "Direct": "outputs/slard-31/baseline/*_baseline_output.jsonl",
        "Static": "outputs/slard-31/ablation/*/*_test_class_no_dynamic.jsonl",
        "Dynamic": "outputs/slard-31/ablation/*/*_test_class*dynamic*.jsonl",
        "Full": "outputs/slard-31/full/pred_classification/*_test_class.jsonl",
    },
}

raw: dict[tuple[str, str], dict[str, dict[str, int]]] = {}
for ds, stages in PATTERNS.items():
    gold = LCR_GOLD if ds == "LCR-CN" else SLARD_GOLD
    k = 5 if ds == "LCR-CN" else 31
    for stage, pat in stages.items():
        key_stage = "Dynamic" if stage == "Dynamic2" else stage
        for path in sorted(glob.glob(str(ROOT / pat))):
            stem = Path(path).stem
            if "_v2" in stem or "no_dynamic" in stem and stage != "Static":
                continue
            if stage == "Dynamic2" and "dynamic" in stem:
                continue
            bb = canon(stem)
            if bb is None:                       # e.g. LawLLM-7B is not one of the six
                continue
            cell = raw.setdefault((ds, key_stage), {}).setdefault(bb, {})
            for r in rows(path):
                rid = str(r["id"]).strip()
                if rid not in gold:
                    continue
                if stage == "Direct":
                    pred = as_int(r.get("pred_label") if ds == "SLARD-31"
                                  else r.get("conflict_type"))
                elif ds == "SLARD-31":
                    pred = as_int(r.get("category"), NAME2ID)
                else:
                    pred = as_int(r.get("pred_conflict_label_id",
                                        r.get("pred_conflict_type")))
                cell[rid] = pred
        # sanity
        if key_stage in raw.get((ds, key_stage), {}):
            pass

# --------------------------------------------------------------------------- #
# 3. Align on shared ids and verify against the published stage means
# --------------------------------------------------------------------------- #
PUBLISHED = {
    "LCR-CN":   {"Direct": 36.37, "Static": 33.53, "Dynamic": 39.46, "Full": 48.19},
    "SLARD-31": {"Direct": 59.92, "Static": 59.88, "Dynamic": 57.43, "Full": 66.88},
}
STAGES = ["Direct", "Static", "Dynamic", "Full"]
aligned: dict[tuple[str, str], dict] = {}
print("=== reproduction check (six-backbone mean Macro-F1) ===")
for ds in PUBLISHED:
    k = 5 if ds == "LCR-CN" else 31
    for stage in STAGES:
        cell = raw[(ds, stage)]
        missing = [b for b in BACKBONES if b not in cell]
        if missing:
            raise SystemExit(f"{ds}/{stage}: missing backbones {missing}")
        ids = sorted(set.intersection(*(set(v) for v in cell.values())))
        y = np.array([LCR_GOLD[i] if ds == "LCR-CN" else SLARD_GOLD[i] for i in ids],
                     dtype=np.int64)
        ps = [np.array([cell[b][i] for i in ids], dtype=np.int64) for b in BACKBONES]
        aligned[(ds, stage)] = {"y": y, "p": ps, "k": k, "n": len(ids)}
        obs = float(np.mean([macro_f1_fast(y, p, k) for p in ps]))
        pub = PUBLISHED[ds][stage]
        flag = "OK" if abs(obs - pub) < 0.6 else "CHECK"
        print(f"  {ds:<9} {stage:<8} n={len(ids):>4}  recomputed={obs:6.2f}  "
              f"paper={pub:6.2f}  diff={obs-pub:+6.2f}  {flag}")

# --------------------------------------------------------------------------- #
# 4. Bootstrap: per-stage CI and paired adjacent-contrast CI
# --------------------------------------------------------------------------- #
rng = np.random.default_rng(SEED)
records: list[dict] = []

for ds in PUBLISHED:
    for stage in STAGES:
        d = aligned[(ds, stage)]
        y, ps, k, n = d["y"], d["p"], d["k"], d["n"]
        draws = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = rng.integers(0, n, size=n)
            yy = y[idx]
            draws[b] = np.mean([macro_f1_fast(yy, p[idx], k) for p in ps])
        obs = float(np.mean([macro_f1_fast(y, p, k) for p in ps]))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        records.append({"dataset": ds, "kind": "stage", "contrast": stage,
                        "estimate": round(obs, 3), "ci_low": round(float(lo), 3),
                        "ci_high": round(float(hi), 3), "n": n,
                        "published": PUBLISHED[ds][stage]})

    for a, b in zip(STAGES[:-1], STAGES[1:]):
        da, db = aligned[(ds, a)], aligned[(ds, b)]
        n = min(da["n"], db["n"])
        # align on the same ordered id set (both derived from the same gold map)
        ids_a = sorted(set.intersection(*(set(v) for v in raw[(ds, a)].values())))
        ids_b = sorted(set.intersection(*(set(v) for v in raw[(ds, b)].values())))
        common = sorted(set(ids_a) & set(ids_b))
        goldmap = LCR_GOLD if ds == "LCR-CN" else SLARD_GOLD
        k = da["k"]
        yy_all = np.array([goldmap[i] for i in common], dtype=np.int64)
        pa = [np.array([raw[(ds, a)][bb][i] for i in common], dtype=np.int64)
              for bb in BACKBONES]
        pb = [np.array([raw[(ds, b)][bb][i] for i in common], dtype=np.int64)
              for bb in BACKBONES]
        n = len(common)
        obs = float(np.mean([macro_f1_fast(yy_all, q, k) - macro_f1_fast(yy_all, p, k)
                             for p, q in zip(pa, pb)]))
        draws = np.empty(N_BOOT)
        for it in range(N_BOOT):
            idx = rng.integers(0, n, size=n)
            yb = yy_all[idx]
            draws[it] = np.mean([macro_f1_fast(yb, q[idx], k) - macro_f1_fast(yb, p[idx], k)
                                 for p, q in zip(pa, pb)])
        lo, hi = np.percentile(draws, [2.5, 97.5])
        records.append({"dataset": ds, "kind": "contrast", "contrast": f"{b} - {a}",
                        "estimate": round(obs, 3), "ci_low": round(float(lo), 3),
                        "ci_high": round(float(hi), 3), "n": n, "published": ""})

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["dataset", "kind", "contrast", "estimate",
                                      "ci_low", "ci_high", "n", "published"])
    w.writeheader()
    w.writerows(records)

print("\n=== REAL per-stage 95% CI (for scheme A) ===")
for r in records:
    if r["kind"] == "stage":
        print(f"  {r['dataset']:<9} {r['contrast']:<8} {r['estimate']:6.2f} "
              f"[{r['ci_low']:6.2f}, {r['ci_high']:6.2f}]")
print("\n=== REAL paired adjacent-contrast 95% CI (for scheme B) ===")
for r in records:
    if r["kind"] == "contrast":
        print(f"  {r['dataset']:<9} {r['contrast']:<22} {r['estimate']:+6.2f} "
              f"[{r['ci_low']:+6.2f}, {r['ci_high']:+6.2f}]")
print("\nwritten", OUT)
