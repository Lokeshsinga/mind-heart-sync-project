"""
ablation_suite.py -- Federated Learning Ablation Suite for SeizeIT2.

Addresses evaluation across differential privacy baselines and feature scaling:
    python ablation_suite.py --seeds 5
    python ablation_suite.py --seeds 1 --rounds 2 --quick      # smoke test
    python ablation_suite.py --configs adaptive                # sensitivity-adaptive noise only
    python ablation_suite.py --audit-rdp                       # compare accountant against Opacus

Configurations:
  1. DEFAULT EXPERIMENT:
     nonprivate   = Standard FedProx (sigma=0.0)
     fixed        = Fixed Gaussian noise (sigma=1.0, unconstrained ceiling)
     uniform_cal  = Uniform Gaussian noise pre-calibrated to satisfy epsilon ceiling
     ceiling_only = Starts at sigma=1.0, increases sigma dynamically to prevent breaching ceiling
  2. EXPLORATORY (Explicit `--configs adaptive`):
     adaptive     = Mahalanobis sensitivity-adaptive noise with dynamic ceiling enforcement

Scaling protocols:
  local  = Original behaviour (each training client uses its own empirical mean/std;
           held-out set uses pooled statistics built from client means and variances)
  global = Single pooled mean/std from federated sufficient statistics (n, sum, sum of squares),
           applied identically to every training client and to held-out patients

Privacy Accounting Scope & Assumptions:
  - Mechanism: Subsampled Gaussian Mechanism (SGM) with Poisson sampling (probability q = B/n).
  - Guarantee Level: Sample-level (10s EEG-ECG window) differential privacy.
    DOES NOT provide Patient-Level DP (group privacy over a patient's recording does not bound epsilon).
  - Data-dependent operations outside accountant: class weighting (n_neg / n_pos), empirical scaling,
    and sensitivity scoring via central Mahalanobis distance.
"""
import os, sys, json, copy, argparse, time, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import gammaln, logsumexp
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, average_precision_score)

# Auto-detect project directory with fallback
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT = SCRIPT_DIR if os.path.exists(os.path.join(SCRIPT_DIR, "seizeit2_multiclient_preprocessed")) \
    else os.path.dirname(SCRIPT_DIR) if os.path.exists(os.path.join(os.path.dirname(SCRIPT_DIR), "seizeit2_multiclient_preprocessed")) \
    else r"C:\Users\lokes\Downloads\project1"

TEST_PATIENTS = ["sub-001", "sub-009"]
TRAIN_CLIENTS = ["sub-002", "sub-003", "sub-004", "sub-005", "sub-006", "sub-007", "sub-008"]
SFREQ = 256
BATCH = 16
LR = 0.003
WD = 1e-4
MU = 0.01
CLIP = 1.0
DELTA = 1e-5
CEILING = 50.0
CEILING_TOL = 1e-4
CONVERSION = "mironov"   # or "balle" (Balle et al. 2020 RDP->DP conversion)
HYBRID_SIGMA = {}
SIZE_SIGMA = {}
SIZEFIX_SIGMA = {}  # Numerical tolerance for ceiling compliance verification
NOISE_MEAN, NOISE_SPREAD, NOISE_FLOOR, NOISE_CEIL = 1.0, 0.40, 0.70, 1.50
ARCH = "mlp"   # "mlp" (59-32-16-1) or "linear" (logistic regression, 59 inputs)
device = torch.device("cpu")  # CPU avoids GPU vmap non-determinism


# ----------------------------------------------------------------------------- accountant
class RDPAccountant:
    """
    Rényi Differential Privacy (RDP) Accountant for Subsampled Gaussian Mechanism (SGM).
    Tracks privacy loss under Poisson subsampling with conversion to (epsilon, delta)-DP
    via Mironov (2017) Proposition 3.
    """
    def __init__(self):
        self.orders = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64,
                                80, 96, 128, 256], dtype=np.float64)
        self.rdp = np.zeros_like(self.orders)

    @staticmethod
    def _rdp(q, sigma, alpha):
        if q <= 0: return 0.0
        if q >= 1: return alpha / (2 * (sigma ** 2))
        a = int(round(alpha)); terms = []
        for j in range(a + 1):
            lc = gammaln(a + 1) - gammaln(j + 1) - gammaln(a - j + 1)
            terms.append(lc + (a - j) * np.log(max(1e-15, 1 - q)) + j * np.log(max(1e-15, q))
                         + (j * (j - 1)) / (2 * (sigma ** 2)))
        return float(max(0.0, logsumexp(terms) / (alpha - 1)))

    def step(self, q, sigma, steps=1):
        for i, a in enumerate(self.orders):
            self.rdp[i] += steps * self._rdp(q, sigma, a)

    def eps(self, vec=None, delta=DELTA):
        v = self.rdp if vec is None else vec
        if CONVERSION == "balle":
            e = v + np.log((self.orders - 1) / self.orders) - (np.log(delta) + np.log(self.orders)) / (self.orders - 1)
        else:
            e = v + np.log(1 / delta) / (self.orders - 1)
        return float(e.min())

    def prospective(self, q, sigma, steps):
        v = self.rdp.copy()
        for i, a in enumerate(self.orders):
            v[i] += steps * self._rdp(q, sigma, a)
        return self.eps(v)


def verify_accountant_against_opacus(tol=1e-10, accountant=None):
    """Independent audit function comparing custom RDP accountant against Opacus."""
    try:
        from opacus.accountants.analysis.rdp import compute_rdp as opacus_rdp, get_privacy_spent
    except ImportError:
        print("[AUDIT] Opacus is unavailable; independent validation was SKIPPED. (Not verified)")
        return "skipped"

    acc = RDPAccountant() if accountant is None else accountant
    orders = list(acc.orders)
    test_cases = [(0.01, 1.0, 100), (0.05, 1.5, 200), (0.10, 0.8, 50)]
    print("[AUDIT] Auditing custom RDP accountant against Opacus:")
    all_passed = True
    for q, sigma, steps in test_cases:
        custom_rdp = np.array([acc._rdp(q, sigma, a) * steps for a in orders])
        op_rdp = opacus_rdp(q=q, noise_multiplier=sigma, steps=steps, orders=orders)
        diff = np.abs(custom_rdp - np.array(op_rdp))
        max_diff = float(np.max(diff))
        custom_eps = acc.eps(custom_rdp, delta=DELTA)
        op_eps, _ = get_privacy_spent(orders=orders, rdp=op_rdp, delta=DELTA)
        print(f"  q={q:.3f}, sigma={sigma:.2f}, steps={steps:3d} -> Max RDP diff: {max_diff:.2e} | "
              f"Eps Custom(Mironov): {custom_eps:.4f}, Opacus(Balle): {op_eps:.4f}")
        if max_diff > tol:
            all_passed = False

    if all_passed:
        print(f"[AUDIT SUCCESS] All RDP accountant verification cases passed within tolerance (max diff <= {tol:.1e}).")
        return "passed"
    else:
        print(f"[AUDIT FAILED] RDP accountant comparison exceeded tolerance ({tol:.1e})!", file=sys.stderr)
        return "failed"



def calibrate_client_min_feasible_sigma(n, rounds, epochs, ceiling=CEILING, max_sigma=100.0):
    """Smallest sigma for a single client such that it meets the ceiling over the full horizon."""
    q = min(1.0, BATCH / n)
    T = math.ceil(n / BATCH) * epochs * rounds
    acc = RDPAccountant()
    if acc.prospective(q, max_sigma, T) > ceiling + CEILING_TOL:
        return None
    lo, hi = 0.1, min(max_sigma, 20.0)
    while acc.prospective(q, hi, T) > ceiling and hi < max_sigma:
        hi = min(max_sigma, hi * 2.0)
        if hi == max_sigma:
            break
    if acc.prospective(q, hi, T) > ceiling + CEILING_TOL:
        return None
    best = hi
    for _ in range(40):
        mid = (lo + hi) / 2
        if acc.prospective(q, mid, T) <= ceiling:
            best = mid
            hi = mid
        else:
            lo = mid
    return float(best)


def calibrate_uniform_sigma(ns, rounds, epochs, ceiling=CEILING, max_sigma=100.0):
    """Smallest uniform sigma such that every client meets the ceiling at its nominal rate."""
    def worst(sig):
        w = 0.0
        for n in ns:
            acc = RDPAccountant()
            T = math.ceil(n / BATCH) * epochs * rounds
            acc.step(min(1.0, BATCH / n), sig, T)
            w = max(w, acc.eps())
        return w
    lo, hi = 0.1, 10.0
    while worst(hi) > ceiling and hi < max_sigma:
        hi = min(max_sigma, hi * 2.0)
        if hi == max_sigma:
            break
    if worst(hi) > ceiling + CEILING_TOL:
        return None
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if worst(mid) > ceiling else (lo, mid)
    return float(hi)


def calibrate_adaptive_sigma(client_ns, relative_sigmas, rounds, epochs, ceiling=CEILING, max_sigma=100.0):
    """Calibrates sensitivity-adaptive noise multipliers across the full training horizon.
    
    Preserves exact relative sensitivity ratios between clients while scaling overall noise
    by global multiplier k such that all clients satisfy cumulative epsilon <= ceiling.
    """
    def worst_eps(k):
        w = 0.0
        for cid, n in client_ns.items():
            acc = RDPAccountant()
            T = math.ceil(n / BATCH) * epochs * rounds
            sig = k * relative_sigmas[cid]
            acc.step(min(1.0, BATCH / n), sig, T)
            w = max(w, acc.eps())
        return w

    lo, hi = 0.1, 10.0
    while worst_eps(hi) > ceiling and hi < max_sigma:
        hi = min(max_sigma, hi * 2.0)
        if hi == max_sigma:
            break
    if worst_eps(hi) > ceiling + CEILING_TOL:
        return None, None

    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if worst_eps(mid) > ceiling else (lo, mid)
    k = float(hi)
    calibrated = {cid: float(k * relative_sigmas[cid]) for cid in client_ns}
    return k, calibrated


# ----------------------------------------------------------------------------- model
class Net(nn.Module):
    def __init__(self, d=59):
        super().__init__()
        if ARCH == "linear":
            self.network = nn.Sequential(nn.Linear(d, 1))
        else:
            self.network = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Dropout(0.2),
                                         nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 1))

    def forward(self, x):
        return self.network(x).squeeze(-1)


# ----------------------------------------------------------------------------- data
def load_data(data_dir, cache):
    labels = np.load(os.path.join(data_dir, "labels.npy"))
    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    if os.path.exists(cache):
        X = np.load(cache)
    else:
        # Robust multi-path resolution for base feature extractor
        for cand in [SCRIPT_DIR, os.path.join(SCRIPT_DIR, "research", "federated_learning"),
                     os.path.join(PROJECT, "research", "federated_learning")]:
            if os.path.exists(os.path.join(cand, "federated_seizeit2_adaptive_dp_fedprox.py")):
                if cand not in sys.path:
                    sys.path.insert(0, cand)
                break
        import federated_seizeit2_adaptive_dp_fedprox as base
        eeg = np.load(os.path.join(data_dir, "eeg.npy")); ecg = np.load(os.path.join(data_dir, "ecg.npy"))
        fe, _ = base.extract_eeg_features(eeg, sfreq=SFREQ)
        fc, _ = base.extract_ecg_features(ecg, sfreq=SFREQ)
        X = np.hstack([fe, fc])
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, X)
    return X, labels, meta


def sensitivity_scores(X, meta):
    """Mahalanobis scores from per-client sufficient statistics (n_i, sum x, sum x x^T) only.
    A windows-based reference computation is run as a cross-check (max abs difference is printed)."""
    stats = {}
    for c in TRAIN_CLIENTS:
        Xc = X[(meta["patient_id"] == c).values]
        stats[c] = (len(Xc), Xc.sum(0), Xc.T @ Xc)
    N = sum(v[0] for v in stats.values())
    mu = sum(v[1] for v in stats.values()) / N
    cov = (sum(v[2] for v in stats.values()) - N * np.outer(mu, mu)) / (N - 1) + 1e-4 * np.eye(X.shape[1])
    s = {}
    for c in TRAIN_CLIENTS:
        d = stats[c][1] / stats[c][0] - mu
        s[c] = float(np.sqrt(max(0.0, d @ np.linalg.solve(cov, d))))
    # cross-check against the windows-based computation
    train = meta["patient_id"].isin(TRAIN_CLIENTS).values
    mu2 = X[train].mean(0); cov2 = np.cov(X[train], rowvar=False) + 1e-4 * np.eye(X.shape[1])
    s2 = {}
    for c in TRAIN_CLIENTS:
        d2 = X[(meta["patient_id"] == c).values].mean(0) - mu2
        s2[c] = float(np.sqrt(max(0.0, d2 @ np.linalg.solve(cov2, d2))))
    print("sensitivity: sufficient-statistics vs windows-based, max abs diff = %.2e" % max(abs(s[c] - s2[c]) for c in TRAIN_CLIENTS))
    arr = np.array([s[c] for c in TRAIN_CLIENTS]); m, sd = arr.mean(), arr.std()
    sig = {c: float(np.clip(NOISE_MEAN + NOISE_SPREAD * (s[c] - m) / (sd + 1e-8), NOISE_FLOOR, NOISE_CEIL))
           for c in TRAIN_CLIENTS}
    return s, sig


# ----------------------------------------------------------------------------- client
class Client:
    def __init__(self, cid, X, y, scaler):
        self.cid, self.n, self.y_np = cid, len(y), y
        self.n_pos = int(y.sum()); self.n_neg = self.n - self.n_pos
        self.single = self.n_pos == 0 or self.n_neg == 0
        self.pos_weight = 1.0 if self.single else self.n_neg / max(1.0, self.n_pos)
        self.mean_raw, self.std_raw = X.mean(0), X.std(0)
        mean, std = scaler(X)
        self.X = torch.tensor((X - mean) / std, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.acc = RDPAccountant(); self.sigma = 0.0; self.frozen = False; self.freeze_reason = None
        self.q = min(1.0, BATCH / self.n); self.steps_epoch = math.ceil(self.n / BATCH)

    def prepare(self, epochs, dp, ceiling, max_sigma=100.0):
        if self.frozen: return False
        if not (dp and ceiling): return True
        T = self.steps_epoch * epochs
        if self.acc.prospective(self.q, self.sigma, T) <= CEILING: return True
        lo = min(self.sigma, max_sigma)
        hi = min(max_sigma, max(self.sigma, 20.0))
        while self.acc.prospective(self.q, hi, T) > CEILING and hi < max_sigma:
            hi = min(max_sigma, hi * 2.0)
            if hi == max_sigma:
                break
        if self.acc.prospective(self.q, hi, T) > CEILING:
            self.frozen = True
            self.freeze_reason = f"Noise search upper bound {hi:.1f} insufficient for ceiling {CEILING}"
            return False
        best = None
        for _ in range(35):
            mid = (lo + hi) / 2
            if self.acc.prospective(self.q, mid, T) <= CEILING:
                best, hi = mid, mid
            else:
                lo = mid
        if best is None:
            self.frozen = True
            self.freeze_reason = "No feasible sigma found in search interval"
            return False
        self.sigma = float(best)
        return True

    def train(self, gstate, epochs, dp, rng, ceiling=False):
        model = Net(self.X.shape[1]); model.load_state_dict(copy.deepcopy(gstate)); model.train()
        gref = {k: v.detach().clone() for k, v in model.named_parameters()}
        lr = LR * 0.5 if self.single else LR
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WD)
        pw = torch.tensor([self.pos_weight])
        names = [k for k, _ in model.named_parameters()]
        from torch.func import functional_call, vmap, grad

        def loss_fn(params, x, y):
            logit = functional_call(model, params, (x.unsqueeze(0),))
            bce = F.binary_cross_entropy_with_logits(logit, y.unsqueeze(0), pos_weight=pw)
            prox = sum(((params[k] - gref[k]) ** 2).sum() for k in params)
            return bce + MU / 2 * prox
        pgrad = vmap(grad(loss_fn), in_dims=(None, 0, 0), randomness="different")

        for _ in range(epochs):
            if dp:
                batches = []
                for _s in range(self.steps_epoch):          # Poisson sampling
                    batches.append(np.flatnonzero(rng.random(self.n) < self.q))
            else:
                perm = rng.permutation(self.n)
                batches = [perm[i:i + BATCH] for i in range(0, self.n, BATCH)]
            for idx in batches:
                if dp:
                    if ceiling:
                        proj_step_eps = self.acc.prospective(self.q, self.sigma, 1)
                        if proj_step_eps > CEILING + CEILING_TOL:
                            self.frozen = True
                            self.freeze_reason = f"Projected step epsilon {proj_step_eps:.4f} exceeds ceiling {CEILING}"
                            raise RuntimeError(f"Step breach: Client {self.cid} step eps {proj_step_eps:.4f} > {CEILING}")
                    denom = max(1.0, self.q * self.n)       # expected batch size
                    if len(idx):
                        xb, yb = self.X[idx], self.y[idx]
                        params = {k: p.detach() for k, p in model.named_parameters()}
                        g = pgrad(params, xb, yb)
                        flat = torch.cat([g[k].reshape(len(idx), -1) for k in names], 1)
                        factor = torch.clamp(CLIP / (flat.norm(dim=1) + 1e-8), max=1.0)
                        summed = {k: (g[k] * factor.view(-1, *[1] * (g[k].dim() - 1))).sum(0) for k in names}
                    else:
                        summed = {k: torch.zeros_like(p) for k, p in model.named_parameters()}
                    for k, p in model.named_parameters():
                        noise = torch.randn_like(p) * self.sigma * CLIP
                        p.grad = (summed[k] + noise) / denom
                    self.acc.step(self.q, self.sigma, 1)
                else:
                    if len(idx) == 0: continue
                    xb, yb = self.X[idx], self.y[idx]
                    params = {k: p.detach() for k, p in model.named_parameters()}
                    g = pgrad(params, xb, yb)
                    for k, p in model.named_parameters(): p.grad = g[k].mean(0)
                opt.step()
        return copy.deepcopy(model.state_dict())


def fed_avg(updates):
    tot = sum(n for _, n in updates); out = copy.deepcopy(updates[0][0])
    for k in out: out[k] = sum((n / tot) * s[k].float() for s, n in updates)
    return out


# ----------------------------------------------------------------------------- one run
def run(config, scaling, seed, X, y, meta, sens_sigma, sigma_uniform, rounds, epochs):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    test = meta["patient_id"].isin(TEST_PATIENTS).values
    tr_ids = TRAIN_CLIENTS
    # scaling protocol
    stats = {c: (X[(meta["patient_id"] == c).values].sum(0), (X[(meta["patient_id"] == c).values] ** 2).sum(0),
                 int((meta["patient_id"] == c).sum())) for c in tr_ids}
    N = sum(v[2] for v in stats.values())
    g_mean = sum(v[0] for v in stats.values()) / N
    g_var = sum(v[1] for v in stats.values()) / N - g_mean ** 2
    g_std = np.where(np.sqrt(np.maximum(g_var, 0)) < 1e-6, 1.0, np.sqrt(np.maximum(g_var, 0)))
    clients = []
    for c in tr_ids:
        m = (meta["patient_id"] == c).values
        if scaling == "global":
            scaler = lambda Xc: (g_mean, g_std)
        else:
            scaler = lambda Xc: (Xc.mean(0), np.where(Xc.std(0) < 1e-6, 1.0, Xc.std(0)))
        cl = Client(c, X[m], y[m], scaler); clients.append(cl)
    if scaling == "local":   # held-out scaling: weighted mean, weighted average of variances
        w = [cl.n / N for cl in clients]
        t_mean = sum(wi * cl.mean_raw for wi, cl in zip(w, clients))
        t_std = np.sqrt(sum(wi * cl.std_raw ** 2 for wi, cl in zip(w, clients)))
        t_std = np.where(t_std < 1e-6, 1.0, t_std)
    else:
        t_mean, t_std = g_mean, g_std

    dp = config != "nonprivate"
    ceiling = config in ("ceiling_only", "adaptive", "uniform_cal", "hybrid", "size_only", "size_fix")
    for cl in clients:
        if config == "nonprivate":
            cl.sigma = 0.0
        elif config == "fixed":
            cl.sigma = 1.0
        elif config == "uniform_cal":
            cl.sigma = float(sigma_uniform) if sigma_uniform is not None else float("nan")
        elif config == "ceiling_only":
            min_feas = calibrate_client_min_feasible_sigma(cl.n, rounds, epochs, CEILING)
            cl.sigma = max(1.0, min_feas) if min_feas is not None else float("nan")
        elif config == "adaptive":
            cl.sigma = float(sens_sigma[cl.cid]) if (sens_sigma and cl.cid in sens_sigma) else float("nan")
        elif config == "hybrid":   # largest client at its ceiling-only sigma, all others at the uniform sigma
            cl.sigma = float(HYBRID_SIGMA.get(cl.cid, float("nan")))
        elif config == "size_fix":  # size-only multipliers, dominant client set to the adaptive rule's sigma
            cl.sigma = float(SIZEFIX_SIGMA.get(cl.cid, float("nan")))
        elif config == "size_only": # size-rank-matched multipliers (no sensitivity score), same k-calibration as adaptive
            cl.sigma = float(SIZE_SIGMA.get(cl.cid, float("nan")))

    # Full-horizon pre-training feasibility check
    if dp and ceiling:
        for cl in clients:
            T_total = cl.steps_epoch * epochs * rounds
            if math.isnan(cl.sigma):
                print(f"[PRE-TRAIN FEASIBILITY FAILURE] Client {cl.cid} noise is NaN/infeasible for ceiling {CEILING}.")
                return dict(config=config, scaling=scaling, seed=seed,
                            acc=float("nan"), prec=float("nan"), recall=float("nan"),
                            spec=float("nan"), f1=float("nan"), auc=float("nan"),
                            ap=float("nan"), TN=0, FP=0, FN=0, TP=0,
                            eps_max=float("inf"), eps_mean=float("inf"),
                            frozen=len(clients),
                            sigma_final=";".join(f"{c.cid}:{c.sigma:.3f}" for c in clients),
                            ceiling_compliant=False,
                            status="FAILED_INFEASIBLE_NOISE"), None, None
            proj_total = cl.acc.prospective(cl.q, cl.sigma, T_total)
            if proj_total > CEILING + CEILING_TOL:
                print(f"[PRE-TRAIN FEASIBILITY FAILURE] Client {cl.cid} sigma={cl.sigma:.4f} projected eps {proj_total:.4f} > {CEILING}.")
                return dict(config=config, scaling=scaling, seed=seed,
                            acc=float("nan"), prec=float("nan"), recall=float("nan"),
                            spec=float("nan"), f1=float("nan"), auc=float("nan"),
                            ap=float("nan"), TN=0, FP=0, FN=0, TP=0,
                            eps_max=proj_total, eps_mean=proj_total,
                            frozen=1,
                            sigma_final=";".join(f"{c.cid}:{c.sigma:.3f}" for c in clients),
                            ceiling_compliant=False,
                            status="FAILED_INFEASIBLE_NOISE"), None, None
    model = Net(X.shape[1]); gstate = copy.deepcopy(model.state_dict())
    for r in range(rounds):
        ups = []
        for cl in clients:
            if not cl.prepare(epochs, dp, ceiling):
                if dp and ceiling:
                    print(f"[CEILING HALT] Client {cl.cid} cannot satisfy ceiling ({CEILING}) in round {r+1}: {getattr(cl, 'freeze_reason', 'infeasible noise search')}. Stopping training.")
                    eps = [c.acc.eps() for c in clients]
                    return dict(config=config, scaling=scaling, seed=seed,
                                acc=float("nan"), prec=float("nan"), recall=float("nan"),
                                spec=float("nan"), f1=float("nan"), auc=float("nan"),
                                ap=float("nan"), TN=0, FP=0, FN=0, TP=0,
                                eps_max=max(eps), eps_mean=float(np.mean(eps)),
                                frozen=sum(c.frozen for c in clients),
                                sigma_final=";".join(f"{c.cid}:{c.sigma:.3f}" for c in clients),
                                ceiling_compliant=False,
                                status="FAILED_INFEASIBLE_NOISE"), None, None
            ups.append((cl.train(gstate, epochs, dp, rng, ceiling=ceiling), cl.n))
        gstate = fed_avg(ups)
    model.load_state_dict(gstate); model.eval()
    Xt = torch.tensor((X[test] - t_mean) / t_std, dtype=torch.float32)
    with torch.no_grad(): prob = torch.sigmoid(model(Xt)).numpy()
    yt = y[test]; pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    spec_val = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    auc_val = float(roc_auc_score(yt, prob)) if len(np.unique(yt)) > 1 else float("nan")
    ap_val = float(average_precision_score(yt, prob)) if len(np.unique(yt)) > 1 else float("nan")
    eps = [cl.acc.eps() for cl in clients] if dp else [float("inf")]
    if dp and ceiling:
        is_compliant = bool(all(e <= CEILING + CEILING_TOL for e in eps) and all(not cl.frozen for cl in clients))
        run_status = "COMPLETED" if is_compliant else "FAILED_CEILING_BREACH"
    else:
        is_compliant = None
        run_status = "COMPLETED"
    return dict(config=config, scaling=scaling, seed=seed,
                acc=accuracy_score(yt, pred), prec=precision_score(yt, pred, zero_division=0),
                recall=recall_score(yt, pred, zero_division=0), spec=spec_val,
                f1=f1_score(yt, pred, zero_division=0), auc=auc_val,
                ap=ap_val, TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
                eps_max=max(eps), eps_mean=float(np.mean(eps)), frozen=sum(cl.frozen for cl in clients),
                sigma_final=";".join(f"{cl.cid}:{cl.sigma:.3f}" for cl in clients),
                ceiling_compliant=is_compliant, status=run_status), prob, yt


def build_summary_df(df, perf_metrics=None):
    """
    Builds structured summary dataframe reporting run counts, ceiling compliance,
    eps_max for valid completed private runs, and performance means/stds calculated
    strictly from completed runs with valid metrics.
    """
    if perf_metrics is None:
        perf_metrics = ["acc", "prec", "recall", "spec", "f1", "auc", "ap"]
    records = []

    for (scaling, cfg), grp in df.groupby(["scaling", "config"]):
        total = len(grp)
        completed = int((grp["status"] == "COMPLETED").sum())
        failed = total - completed
        is_private = (cfg != "nonprivate")

        if is_private:
            compliant = int(grp["ceiling_compliant"].apply(
                lambda v: v is True or (isinstance(v, str) and v.strip().lower() == "true")
            ).sum())
            ceiling_fails = int(grp["ceiling_compliant"].apply(
                lambda v: v is False or (isinstance(v, str) and v.strip().lower() == "false")
            ).sum())
        else:
            compliant = "N/A"
            ceiling_fails = "N/A"

        completed_grp = grp[grp["status"] == "COMPLETED"]

        row_dict = {
            ("total_runs", ""): total,
            ("completed_runs", ""): completed,
            ("failed_runs", ""): failed,
            ("ceiling_compliant_runs", ""): compliant,
            ("ceiling_failure_runs", ""): ceiling_fails,
        }

        # Epsilon max summary for valid completed private runs
        if is_private and len(completed_grp) > 0:
            row_dict[("eps_max", "mean")] = round(float(completed_grp["eps_max"].mean()), 3)
            row_dict[("eps_max", "std")] = round(float(completed_grp["eps_max"].std()), 3) if len(completed_grp) > 1 else np.nan
        else:
            row_dict[("eps_max", "mean")] = np.nan if is_private else float("inf")
            row_dict[("eps_max", "std")] = np.nan

        for m in perf_metrics:
            if len(completed_grp) > 0 and m in completed_grp.columns:
                s = completed_grp[m].dropna()
                row_dict[(m, "mean")] = round(float(s.mean()), 3) if len(s) > 0 else np.nan
                row_dict[(m, "std")] = round(float(s.std()), 3) if len(s) > 1 else np.nan
            else:
                row_dict[(m, "mean")] = np.nan
                row_dict[(m, "std")] = np.nan

        records.append((scaling, cfg, row_dict))

    cols = list(records[0][2].keys())
    idx = pd.MultiIndex.from_tuples([(r[0], r[1]) for r in records], names=["scaling", "config"])
    data = [[r[2][c] for c in cols] for r in records]
    summ_df = pd.DataFrame(data, index=idx, columns=pd.MultiIndex.from_tuples(cols))
    return summ_df


def main():
    global TEST_PATIENTS, TRAIN_CLIENTS, CEILING, CONVERSION, HYBRID_SIGMA, SIZE_SIGMA, SIZEFIX_SIGMA
    ap = argparse.ArgumentParser(description="Privacy-budgeted federated learning ablation suite on SeizeIT2")
    ap.add_argument("--data-dir", default=os.path.join(PROJECT, "seizeit2_multiclient_preprocessed"))
    ap.add_argument("--out-dir", default=os.path.join(PROJECT, "seizeit2_ablation"))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=5)
    # Default configs per required specification: nonprivate,fixed,uniform_cal,ceiling_only
    # Adaptive remains supported via --configs adaptive
    ap.add_argument("--configs", default="nonprivate,fixed,uniform_cal,ceiling_only",
                    help="Comma-separated configurations: nonprivate,fixed,uniform_cal,ceiling_only,adaptive")
    ap.add_argument("--scalings", default="local,global")
    ap.add_argument("--arch", default="mlp", choices=["mlp", "linear"],
                    help="linear = logistic regression diagnostic (60 parameters, far less DP noise)")
    ap.add_argument("--cache", default=os.path.join(PROJECT, "seizeit2_ablation", "features_cache.npy"))
    ap.add_argument("--test-patients", default="sub-001,sub-009", help="comma-separated held-out patients")
    ap.add_argument("--exclude-clients", default="", help="comma-separated patients dropped from training (e.g. sub-004)")
    ap.add_argument("--features", default="all", choices=["all", "eeg", "ecg"],
                    help="feature subset: all 59, EEG only (first 41) or ECG only (last 18)")
    ap.add_argument("--list-patients", action="store_true", help="print per-patient window/seizure counts and exit")
    ap.add_argument("--ceiling", type=float, default=50.0, help="per-client epsilon ceiling (finite)")
    ap.add_argument("--conversion", default="mironov", choices=["mironov", "balle"])
    ap.add_argument("--quick", action="store_true",
                    help="Fast smoke test flag: overrides defaults to 1 seed, 2 rounds, 1 epoch if not set")
    ap.add_argument("--audit-rdp", action="store_true",
                    help="Run an analytical RDP verification against Opacus on startup")
    a = ap.parse_args()

    if a.audit_rdp:
        status = verify_accountant_against_opacus()
        if status == "failed":
            print("[FATAL] RDP audit failed. Exiting before starting experiments.", file=sys.stderr)
            sys.exit(1)

    if a.quick:
        if a.seeds == 5: a.seeds = 1
        if a.rounds == 10: a.rounds = 2
        if a.epochs == 5: a.epochs = 1
        print(f"[QUICK MODE] Running fast smoke test: seeds={a.seeds}, rounds={a.rounds}, epochs={a.epochs}")

    global ARCH
    ARCH = a.arch
    CEILING = float(a.ceiling); CONVERSION = a.conversion
    TEST_PATIENTS = [p.strip() for p in a.test_patients.split(",") if p.strip()]
    excl = [p.strip() for p in a.exclude_clients.split(",") if p.strip()]
    if a.arch == "linear" and a.out_dir == os.path.join(PROJECT, "seizeit2_ablation"):
        a.out_dir = os.path.join(PROJECT, "seizeit2_ablation_linear")

    os.makedirs(os.path.join(a.out_dir, "preds"), exist_ok=True)
    X, y, meta = load_data(a.data_dir, a.cache)
    if a.list_patients:
        mm = meta.assign(y=y); print(mm.groupby("patient_id")["y"].agg(["size", "sum"])); return
    if a.features == "eeg": X = X[:, :41]
    elif a.features == "ecg": X = X[:, 41:]
    print("features:", a.features, "-> input dim", X.shape[1])
    TRAIN_CLIENTS = [p for p in sorted(meta["patient_id"].unique()) if p not in TEST_PATIENTS and p not in excl]
    print("train clients:", TRAIN_CLIENTS, "| held-out:", TEST_PATIENTS, "| ceiling:", CEILING, "| conversion:", CONVERSION)
    scores, sens_sigma_rel = sensitivity_scores(X, meta)
    client_ns = {c: int((meta["patient_id"] == c).sum()) for c in TRAIN_CLIENTS}
    ns_list = [client_ns[c] for c in TRAIN_CLIENTS]
    sig_u = calibrate_uniform_sigma(ns_list, a.rounds, a.epochs, ceiling=CEILING)
    k_adapt, sens_sigma_cal = calibrate_adaptive_sigma(client_ns, sens_sigma_rel, a.rounds, a.epochs, ceiling=CEILING)
    # size-only control: rank-match multipliers to client size (smallest client -> 1.5, largest -> 0.7)
    order = sorted(TRAIN_CLIENTS, key=lambda c: client_ns[c])
    nrk = max(1, len(order) - 1)
    size_rel = {c: NOISE_CEIL - (NOISE_CEIL - NOISE_FLOOR) * (i / nrk) for i, c in enumerate(order)}
    k_size, SIZE_SIGMA = calibrate_adaptive_sigma(client_ns, size_rel, a.rounds, a.epochs, ceiling=CEILING)
    SIZE_SIGMA = SIZE_SIGMA or {}
    # hybrid control: largest client at its own ceiling-only sigma, all others at the uniform sigma
    dom = max(TRAIN_CLIENTS, key=lambda c: client_ns[c])
    mf = calibrate_client_min_feasible_sigma(client_ns[dom], a.rounds, a.epochs, CEILING)
    HYBRID_SIGMA = {c: (sig_u if sig_u is not None else float("nan")) for c in TRAIN_CLIENTS}
    if mf is not None: HYBRID_SIGMA[dom] = max(1.0, mf)
    SIZEFIX_SIGMA = dict(SIZE_SIGMA)
    if sens_sigma_cal and dom in sens_sigma_cal and dom in SIZEFIX_SIGMA:
        SIZEFIX_SIGMA[dom] = float(sens_sigma_cal[dom])
    print("size-fix sigmas (dominant client %s at adaptive value):" % dom, {c: round(v, 3) for c, v in SIZEFIX_SIGMA.items()})
    print("size-only k:", k_size, "| size-only sigmas:", {c: round(v, 3) for c, v in SIZE_SIGMA.items()})
    print("hybrid sigmas (dominant client %s):" % dom, {c: round(v, 3) for c, v in HYBRID_SIGMA.items()})
    print(f"uniform sigma calibrated to ceiling {CEILING}: {sig_u:.4f}" if sig_u is not None else "uniform calibration: INFEASIBLE")
    print(f"adaptive scale factor k: {k_adapt:.4f}" if k_adapt is not None else "adaptive calibration: INFEASIBLE")
    print("sensitivity scores:", {k: round(v, 3) for k, v in scores.items()})
    print("relative sensitivity sigmas:", {k: round(v, 3) for k, v in sens_sigma_rel.items()})
    if sens_sigma_cal:
        print("full-horizon adaptive sigmas:", {k: round(v, 4) for k, v in sens_sigma_cal.items()})

    requested_configs = [c.strip() for c in a.configs.split(",") if c.strip()]
    valid_configs = {"nonprivate", "fixed", "uniform_cal", "ceiling_only", "adaptive", "hybrid", "size_only", "size_fix"}
    for cfg in requested_configs:
        if cfg not in valid_configs:
            raise ValueError(f"Unknown config '{cfg}'. Valid options are: {sorted(list(valid_configs))}")

    rows = []
    for scaling in [s.strip() for s in a.scalings.split(",") if s.strip()]:
        for cfg in requested_configs:
            for seed in range(a.seeds):
                t0 = time.time()
                row, prob, yt = run(cfg, scaling, seed, X, y, meta, sens_sigma_cal, sig_u, a.rounds, a.epochs)
                rows.append(row)
                if prob is not None and yt is not None:
                    np.savez(os.path.join(a.out_dir, "preds", f"{cfg}_{scaling}_s{seed}.npz"), prob=prob, y=yt)
                auc_str = f"{row['auc']:.3f}" if not math.isnan(row['auc']) else "NaN"
                rec_str = f"{row['recall']:.3f}" if not math.isnan(row['recall']) else "NaN"
                print(f"{scaling:6s} {cfg:13s} seed {seed} auc={auc_str} rec={rec_str} "
                      f"eps_max={row['eps_max']:.2f} status={row['status']} ({time.time() - t0:.0f}s)", flush=True)
                pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, "results.csv"), index=False)

    df = pd.DataFrame(rows)
    summ = build_summary_df(df)
    summ.to_csv(os.path.join(a.out_dir, "summary.csv"))
    print("\n", summ.to_string())

    setup_meta = {
        "uniform_sigma": sig_u,
        "sensitivity": scores,
        "adaptive_k": k_adapt,
        "adaptive_sigma_rel": sens_sigma_rel,
        "adaptive_sigma_cal": sens_sigma_cal,
        "size_only_sigma": SIZE_SIGMA, "size_fix_sigma": SIZEFIX_SIGMA, "features": a.features, "hybrid_sigma": HYBRID_SIGMA,
        "test_patients": TEST_PATIENTS, "train_clients": TRAIN_CLIENTS, "conversion": CONVERSION,
        "arch": a.arch,
        "seeds": a.seeds,
        "rounds": a.rounds,
        "epochs": a.epochs,
        "configs": requested_configs,
        "scalings": a.scalings.split(","),
        "privacy_ceiling": CEILING,
        "target_delta": DELTA,
        "dp_guarantee_level": "sample_level_only",
        "data_dependent_unaccounted_ops": [
            "class_weighting_n_neg_div_n_pos",
            "feature_standardization_mean_std",
            "mahalanobis_sensitivity_scoring_and_noise_mapping"
        ]
    }
    json.dump(setup_meta, open(os.path.join(a.out_dir, "setup.json"), "w"), indent=1)
    print(f"\nSaved results.csv, summary.csv, setup.json and preds/ in {a.out_dir}")


if __name__ == "__main__":
    main()