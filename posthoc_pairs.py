import csv, argparse
import numpy as np
from scipy import stats
ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--scaling", default="global")
ap.add_argument("--pairs", default="adaptive:uniform_cal,adaptive:ceiling_only,hybrid:uniform_cal,adaptive:hybrid,size_only:adaptive")
a = ap.parse_args(); d = {}
with open(a.csv, newline="") as f:
    for r in csv.DictReader(f):
        if r["scaling"] == a.scaling: d.setdefault(r["config"], {})[int(r["seed"])] = float(r["auc"])
for pr in a.pairs.split(","):
    A, B = pr.split(":")
    if A not in d or B not in d: print(pr, "missing"); continue
    s = sorted(set(d[A]) & set(d[B])); x = np.array([d[A][i] - d[B][i] for i in s]); n = len(x)
    se = x.std(ddof=1) / np.sqrt(n); tc = stats.t.ppf(0.975, n - 1); t = x.mean() / se
    p = 2 * stats.t.sf(abs(t), n - 1); w = stats.wilcoxon(x).pvalue
    print("%-24s n=%d diff=%+.4f CI[%+.4f,%+.4f] t=%.2f p=%.4f bonf2=%.4f wilc=%.4f" % (pr, n, x.mean(), x.mean() - tc * se, x.mean() + tc * se, t, p, min(1, 2 * p), w))
