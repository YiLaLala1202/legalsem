"""Per-class paired bootstrap CI for Full - Direct on LCR-CN.

Self-contained: reads the stored held-out predictions directly, so it does not
depend on the analysis package modules. LCR-CN baseline files store the
*prediction* under `conflict_type`; gold labels come from the Full files.
"""
import json, glob, os
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "outputs" / "lcr-cn"
FULL = sorted(glob.glob(str(ROOT / "full/pred_classification/*_test_class.jsonl")))
BASE = sorted(glob.glob(str(ROOT / "baseline/*_baseline_output.jsonl")))
NAMES = {0:"No conflict",1:"Authority/responsibility",2:"Concept/definition scope",
         3:"Sanction magnitude/range",4:"Applicability condition"}
L2I = {"无冲突":0,"职权或责任划分不符":1,"概念或定义范围不符":2,
       "处罚幅度或范围不符":3,"增设或变更适用条件":4}
K = 5
N_BOOT = 5000
SEED = 2026

def norm(x):
    s = str(x).strip()
    try: return str(int(float(s)))
    except: return s

def as_int(x):
    if x is None: return -1
    v = str(x).strip()
    try: return int(v)
    except: return L2I.get(v, -1)

def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()]

# gold reference by id (Full files carry gold in conflict_type)
gold = {}
for p in FULL:
    for r in read(p):
        g = as_int(r.get("gold_label", r.get("conflict_type")))
        if g in range(K): gold.setdefault(norm(r["id"]), g)

# paired prediction arrays per backbone, aligned on shared ids
pairs = []
for p in FULL:
    tag = os.path.basename(p).split("_test_class")[0]
    b = next((q for q in BASE if os.path.basename(q).startswith(tag)), None)
    if b is None: continue
    full = {norm(r["id"]): as_int(r.get("pred_conflict_label_id", r.get("pred_conflict_type"))) for r in read(p)}
    direct = {norm(r["id"]): as_int(r.get("conflict_type")) for r in read(b)}
    ids = sorted(set(full) & set(direct) & set(gold))
    pairs.append((tag, ids,
                  np.array([gold[i] for i in ids]),
                  np.array([direct[i] for i in ids]),
                  np.array([full[i] for i in ids])))
    print(f"  {tag:<28} n={len(ids)}")

def f1_class(y, p, c):
    tp = int(((y==c)&(p==c)).sum()); fp = int(((y!=c)&(p==c)).sum()); fn = int(((y==c)&(p!=c)).sum())
    d = 2*tp+fp+fn
    return 2*tp/d*100 if d else 0.0

rng = np.random.default_rng(SEED)
print(f"\n{'class':<28} {'n':>5} {'Direct':>7} {'Full':>7} {'delta':>8} {'95% CI':>20}")
print("-"*80)
for c in range(K):
    n = min(len(ids) for _,ids,_,_,_ in pairs)
    d_vals = [f1_class(y,d,c) for _,_,y,d,f in pairs]
    f_vals = [f1_class(y,f,c) for _,_,y,d,f in pairs]
    delta = float(np.mean(f_vals)-np.mean(d_vals))
    draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        acc = 0.0
        for _,_,y,d,f in pairs:
            yy,dd,ff = y[idx],d[idx],f[idx]
            acc += f1_class(yy,ff,c)-f1_class(yy,dd,c)
        draws[b] = acc/len(pairs)
    lo,hi = np.percentile(draws,[2.5,97.5])
    print(f"{NAMES[c]:<28} {int(sum((y==c).sum() for _,_,y,_,_ in pairs)/len(pairs)):>5} "
          f"{np.mean(d_vals):7.2f} {np.mean(f_vals):7.2f} {delta:+8.2f}  [{lo:+.2f}, {hi:+.2f}]"
          f"{'  CI includes 0' if lo<=0<=hi else ''}")
