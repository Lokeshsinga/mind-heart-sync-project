"""Cross-check of the ledger's epsilon against dp_accounting (pip install dp-accounting).

Poisson-sampled Gaussian, q = min(1, 16/n), T = ceil(n/16) * 50 steps (10 rounds x 5 epochs),
delta = 1e-5, for the four constant per-client multipliers of the full-horizon diagnostic
(fixed, uniform, ceiling-only, adaptive). Ledger epsilons are those of Table 7 of the paper."""
import math, logging
from dp_accounting import dp_event
from dp_accounting.rdp import rdp_privacy_accountant as r
logging.getLogger("absl").setLevel(logging.ERROR)   # hides 'failed to converge' for orders < 2

def eps(q, sigma, steps, delta=1e-5):
    acc = r.RdpAccountant()
    acc.compose(dp_event.PoissonSampledDpEvent(q, dp_event.GaussianDpEvent(sigma)), steps)
    return acc.get_epsilon(delta)

CLIENTS = ["sub-002", "sub-003", "sub-004", "sub-005", "sub-006", "sub-007", "sub-008"]
N = [588, 18, 37, 99, 6, 36, 15]
SIGMA = {   # per-client noise multipliers
    "fixed":        [1.0] * 7,
    "uniform":      [1.4644] * 7,
    "ceiling_only": [1.0, 1.4644, 1.0310, 1.0, 1.1400, 1.0490, 1.1400],
    "adaptive":     [1.0166, 1.4644, 1.2382, 1.1306, 2.1785, 1.5103, 1.7140],
}
LEDGER = {  # epsilon reported by the project's own accountant (Table 7)
    "fixed":        [9.33, 97.28, 53.31, 26.88, 61.51, 55.35, 61.51],
    "uniform":      [4.95, 50.00, 27.32, 14.61, 34.83, 28.16, 34.83],
    "ceiling_only": [9.33, 50.00, 50.00, 26.88, 50.00, 50.00, 50.00],
    "adaptive":     [8.99, 50.00, 35.32, 22.20, 21.56, 26.99, 28.53],
}
lower = 0
for cfg, sig in SIGMA.items():
    print(cfg)
    for c, n, s, led in zip(CLIENTS, N, sig, LEDGER[cfg]):
        e = eps(min(1.0, 16 / n), s, math.ceil(n / 16) * 50)
        lower += e <= led + 0.05
        print(f"  {c}  n={n:4d}  sigma={s:.4f}  ledger={led:6.2f}  dp_accounting={e:6.2f}")
print(f"dp_accounting <= ledger in {lower} of 28 client-settings")
