"""Aggregate leave-one-patient-out (LOPO) runs of ablation_suite_v3.py.
Usage: python aggregate_lopo.py <root_dir> [--scaling global] [--min-pos 2] [--from-csv]
--from-csv regenerates the statistics from the published results.csv files (no preds/*.npz needed).
root_dir contains one sub-folder per held-out patient, named after the patient (e.g. sub-001), each holding preds/*.npz
(written by ablation_suite_v3.py --out-dir <root_dir>/<patient> --test-patients <patient>).
Per fold, per config: AUC averaged over seeds. Then, across patients with both classes (>= min-pos positives and negatives):
mean AUC with a percentile bootstrap CI over patients, and paired differences between configs (bootstrap CI over patients,
exact sign-flip p). Patient-level resampling is the unit of inference; the seeds are averaged out first.
Statistical details:
 - Resampling unit: the held-out patient. Each patient contributes ONE AUC per config (mean over seeds), with equal weight,
   regardless of how many windows or seeds the fold has.
 - CI: percentile bootstrap, --boot resamples of the patient list (default 20000).
 - Paired test: exact sign-flip (randomisation) test on the per-patient differences, enumerating all 2^n sign patterns
   (n = number of patients, here 8). Two-sided: p = share of patterns whose |mean difference| >= the observed one.
 - Zero differences stay in the sign-flip enumeration (they contribute 0 under every flip) and are counted as neither
   'better' nor 'worse' in the 'patients better' count (strictly > 0).
 - Patients with fewer than --min-pos positives or negatives are excluded (default 2).
"""
import os, sys, glob, re, itertools, argparse
import numpy as np
from sklearn.metrics import roc_auc_score
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--scaling", default="global")
ap.add_argument("--min-pos", type=int, default=2); ap.add_argument("--boot", type=int, default=20000)
ap.add_argument("--max-seeds", type=int, default=0, help="if >0, use only seeds 0..N-1 in every fold (equal seeds for all configs)")
ap.add_argument("--from-csv", action="store_true", help="read <root>/<patient>/results.csv (columns config, scaling, seed, auc, TN, FP, FN, TP) instead of preds/*.npz; no prediction files needed")
ap.add_argument("--pairs", default="adaptive:uniform_cal,adaptive:ceiling_only,adaptive:hybrid,size_only:adaptive,size_fix:adaptive")
a = ap.parse_args(); rng = np.random.default_rng(0)
auc = {}   # config -> {patient: mean AUC over seeds}
npos = {}
for d in sorted(glob.glob(os.path.join(a.root, "*"))):
    pat = os.path.basename(d)
    if a.from_csv:
        import csv
        fcsv = os.path.join(d, "results.csv")
        if not os.path.isfile(fcsv): continue
        with open(fcsv, newline="") as fh:
            for row in csv.DictReader(fh):
                if row["scaling"] != a.scaling: continue
                if a.max_seeds > 0 and int(row["seed"]) >= a.max_seeds: continue
                try: v = float(row["auc"])
                except ValueError: continue
                if not np.isfinite(v): continue
                pos = int(float(row["TP"]) + float(row["FN"])); neg = int(float(row["TN"]) + float(row["FP"])); npos[pat] = (pos, neg)
                if pos < a.min_pos or neg < a.min_pos: continue
                auc.setdefault(row["config"], {}).setdefault(pat, []).append(v)
        continue
    for f in glob.glob(os.path.join(d, "preds", f"*_{a.scaling}_s*.npz")):
        mm = re.match(rf"(.+)_{a.scaling}_s(\d+)\.npz", os.path.basename(f)); cfg = mm.group(1)
        if a.max_seeds > 0 and int(mm.group(2)) >= a.max_seeds: continue
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
