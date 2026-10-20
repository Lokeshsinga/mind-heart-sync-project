"""Aggregate leave-one-patient-out (LOPO) runs of ablation_suite_v3.py.
Usage: python aggregate_lopo.py <root_dir> [--scaling global] [--min-pos 2]
root_dir contains one sub-folder per held-out patient, named after the patient (e.g. sub-001), each holding preds/*.npz
(written by ablation_suite_v3.py --out-dir <root_dir>/<patient> --test-patients <patient>).
Per fold, per config: AUC averaged over seeds. Then, across patients with both classes (>= min-pos positives and negatives):
mean AUC with a percentile bootstrap CI over patients, and paired differences between configs (bootstrap CI over patients,
exact sign-flip p). Patient-level resampling is the unit of inference; the seeds are averaged out first."""
import os, sys, glob, re, itertools, argparse
import numpy as np
from sklearn.metrics import roc_auc_score
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--scaling", default="global")
ap.add_argument("--min-pos", type=int, default=2); ap.add_argument("--boot", type=int, default=20000)
ap.add_argument("--pairs", default="adaptive:uniform_cal,adaptive:ceiling_only,adaptive:hybrid,size_only:adaptive,size_fix:adaptive")
a = ap.parse_args(); rng = np.random.default_rng(0)
auc = {}   # config -> {patient: mean AUC over seeds}
npos = {}
for d in sorted(glob.glob(os.path.join(a.root, "*"))):
    pat = os.path.basename(d)
    for f in glob.glob(os.path.join(d, "preds", f"*_{a.scaling}_s*.npz")):
        cfg = re.match(rf"(.+)_{a.scaling}_s\d+\.npz", os.path.basename(f)).group(1)
        z = np.load(f); y = z["y"].astype(int); p = z["prob"]
        pos, neg = int(y.sum()), int((1 - y).sum()); npos[pat] = (pos, neg)
        if pos < a.min_pos or neg < a.min_pos: continue
        auc.setdefault(cfg, {}).setdefault(pat, []).append(roc_auc_score(y, p))
auc = {c: {p: float(np.mean(v)) for p, v in d.items()} for c, d in auc.items()}
pats = sorted(set.intersection(*[set(d) for d in auc.values()])) if auc else []
print("held-out patients used:", pats, "| skipped:", sorted(set(npos) - set(pats)))
print("held-out (pos, neg):", {p: npos[p] for p in sorted(npos)})
def ci(v):
    v = np.asarray(v); b = [rng.choice(v, len(v)).mean() for _ in range(a.boot)]
    return np.percentile(b, [2.5, 97.5])
print("\n%-14s %8s  %-18s  per-patient AUC" % ("config", "mean", "95% CI (patients)"))
for c in sorted(auc):
    v = [auc[c][p] for p in pats]; lo, hi = ci(v)
    print("%-14s %8.4f  [%.4f, %.4f]  %s" % (c, np.mean(v), lo, hi, " ".join("%.3f" % x for x in v)))
print("\npaired differences (A - B), over patients")
for pr in a.pairs.split(","):
    A, B = pr.split(":")
    if A not in auc or B not in auc: continue
    dlt = np.array([auc[A][p] - auc[B][p] for p in pats]); lo, hi = ci(dlt)
    signs = np.array(list(itertools.product([-1, 1], repeat=len(dlt)))) if len(dlt) <= 16 else None
    pv = float(np.mean(np.abs((signs * dlt).mean(1)) >= abs(dlt.mean()) - 1e-12)) if signs is not None else float("nan")
    print("%-26s %+.4f  CI [%+.4f, %+.4f]  sign-flip p=%.3f  (patients better: %d/%d)" % (pr, dlt.mean(), lo, hi, pv, int((dlt > 0).sum()), len(dlt)))