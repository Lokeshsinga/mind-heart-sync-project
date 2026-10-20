# Privacy-Budgeted Federated Learning for EEG–ECG Seizure Analytics

Code for the paper *Privacy-Budgeted Federated Learning for EEG–ECG Seizure Analytics: An Audit of Privacy Accounting and Client-Specific Noise*.

> **Scope.** Research code on a nine-patient subset of the public SeizeIT2 dataset. It is **not** a clinical tool and not a validated seizure detector. The reported ε values are accountant outputs for the DP-SGD gradient perturbations only, at **sample level** (not patient level) and under add/remove adjacency. They do not cover the un-noised per-client statistics used for standardisation and sensitivity scoring, or class counts used for loss weighting.

## Files

| File | Purpose |
|---|---|
| `generate_seizeit2_multiclient.py` | Builds the windowed dataset (10 s, 256 Hz, 2 EEG + 1 ECG channel) from the SeizeIT2 BIDS files for sub-001…sub-009. Seizure window = any overlap with an `sz_` event; background = window entirely inside a `bckg` event; windows overlapping `impd`/other events are excluded; background sampled 2:1 per patient (seed 42). |
| `preprocess_seizeit2_multiclient.py` | Per-window demeaning, 4th-order zero-phase Butterworth band-pass (0.5–40 Hz), per-channel z-scoring. |
| `ablation_suite.py` | Main experiment. Own RDP accountant (integer orders, subsampled Gaussian, Poisson sampling), noise configurations (`nonprivate`, `fixed`, `uniform_cal`, `ceiling_only`, `adaptive`, `hybrid`, `size_only`, `size_fix`), logistic-regression and MLP models, leave-patients-out options, ε-ceiling calibration. Writes `results.csv`, `summary.csv`, `setup.json` and per-seed predictions. |
| `aggregate_lopo.py` | Aggregates leave-one-patient-out folds: per-patient AUC, bootstrap CI over patients, paired differences with exact sign-flip tests. `--from-csv` reads the published `results.csv` files. |
| `posthoc_pairs.py` | Paired per-seed comparisons (t-test, 95% CI, Wilcoxon) from full-precision AUCs; produces Table 4. |
| `check_dp_accounting.py` | Cross-checks the accountant against closed-form arithmetic, `dp_accounting` and Opacus. |
| `federated_seizeit2_adaptive_dp_fedprox.py` | Earlier prototype (shuffled batches, greedy ceiling rule). Its ε values use shuffled-batch accounting and are **not** those reported in the Poisson tables of the paper. Kept for transparency. |

## Data

SeizeIT2 is **not** included. Download it from its public source (see the paper's reference [19]) and set the path constants at the top of `generate_seizeit2_multiclient.py` (`DATASET_ROOT`, `OUTPUT_DIR`). The scripts currently contain Windows paths from the authors' machine; edit them before running.

## Results in this repository

| Folder / file | Paper |
|---|---|
| `results/mlp_5seeds/` | Table 1 (MLP, five seeds) |
| `results/lr_full_horizon/` | Table 2 and Table 3 rows for non-private, fixed, uniform, ceiling-only and adaptive noise (logistic regression, five seeds, full-horizon calibration) |
| `results/lr_5seeds/` | Table 2 hybrid and size-only controls; Table 4 and Sect. 6.2 (logistic regression, five seeds; same adaptive, ceiling-only, uniform and non-private values as above) |
| `results/lopo/sub-XXX/` | Table 6 (one folder per held-out patient; `results.csv`, `summary.csv`, `setup.json`) |
| `results/lopo_aggregate.txt` | Table 6 and the LOPO statistics, generated with `aggregate_lopo.py --from-csv` |

Per-seed prediction files (`preds/*.npz`) are **not** published (they are in `.gitignore`). They were used only in the authors' local audit. Every table can be regenerated from the published `results.csv` files without them.

## Reproducing

### A. Without the dataset (from the published CSV files)

```bash
python check_dp_accounting.py                                       # accountant cross-check (needs dp_accounting, opacus)
python aggregate_lopo.py results/lopo --from-csv                    # Table 6 and LOPO statistics
python posthoc_pairs.py results/lr_5seeds/results.csv               # Table 4
python posthoc_pairs.py results/lr_full_horizon/results.csv --pairs adaptive:uniform_cal,adaptive:ceiling_only   # cross-check (full-precision paired comparisons)
```

### B. With SeizeIT2 (rerunning training)

SeizeIT2 is not included. Download it from its public source (paper reference [19]) and set the path constants at the top of `generate_seizeit2_multiclient.py` (`DATASET_ROOT`, `OUTPUT_DIR`); the scripts contain Windows paths from the authors' machine.

```bash
python generate_seizeit2_multiclient.py       # raw windows (937 labelled windows from sub-001..sub-009)
python preprocess_seizeit2_multiclient.py     # filtered, z-scored windows
python ablation_suite.py --help               # all options
```

Experiment commands. Each run's exact arguments are stored in its `setup.json`; check that file if anything differs.

```bash
# Table 2 (non-private, fixed, uniform, ceiling-only, adaptive; logistic regression, five seeds, held-out sub-001 and sub-009)
python ablation_suite.py --arch linear --scalings global --seeds 5 \
  --configs nonprivate,fixed,uniform_cal,ceiling_only,adaptive \
  --out-dir results/lr_full_horizon

# Tables 2 (controls) and 4 (hybrid and size-only controls plus the comparison configurations)
python ablation_suite.py --arch linear --scalings global --seeds 5 \
  --configs nonprivate,uniform_cal,ceiling_only,adaptive,hybrid,size_only \
  --out-dir results/lr_5seeds

# Table 1 (MLP, five seeds)
python ablation_suite.py --arch mlp --scalings local,global --seeds 5 --rounds 10 --epochs 5 \
  --configs nonprivate,fixed,uniform_cal --test-patients sub-001,sub-009 --out-dir results/mlp_5seeds

# Table 6 (one run per held-out patient; two to five seeds per fold, see the paper)
for p in sub-001 sub-002 sub-003 sub-005 sub-006 sub-007 sub-008 sub-009; do
  python ablation_suite.py --arch linear --scalings global --seeds 5 \
    --configs nonprivate,uniform_cal,ceiling_only,adaptive,hybrid,size_only,size_fix \
    --test-patients $p --out-dir results/lopo/$p
done
python aggregate_lopo.py results/lopo --from-csv
```

(`--quick` runs a reduced smoke test. It checks that the code runs; it does **not** reproduce the full ten-round, five-seed experiments.)

Full runs use 10 rounds x 5 local epochs. Training details: Adam, learning rate 0.003 for both the MLP and logistic regression (halved for the single-class client sub-004), weight decay 1e-4, FedProx mu = 0.01, clipping C = 1.0, Poisson sampling with q = min(1, 16/n), delta = 1e-5, epsilon ceiling 50.

Of the 59 features (41 EEG + 18 ECG), 9 are constant after per-window z-scoring (mean, standard deviation and RMS of EEG channels 0 and 1 and the ECG channel), leaving 50 informative features.

**LOPO statistics** (`aggregate_lopo.py`): the resampling unit is the held-out patient; each patient contributes one AUC per configuration (seeds averaged first) with equal weight. Confidence intervals are percentile bootstraps over patients (20 000 resamples); paired differences use an exact two-sided sign-flip test over all 2^n sign patterns. Zero differences stay in the enumeration and are counted as neither better nor worse. Patients with fewer than two positives or two negatives are excluded. In each fold all configurations use the same seeds (five for sub-001 and sub-002, two for the others).

## Environment

Python 3.14.5, dp-accounting 0.4.4, opacus 1.4.1 (accountant cross-check only). Main packages: `numpy`, `pandas`, `scipy`, `scikit-learn`, `torch`, `mne`.

## Citation

Paper under review; citation to be added.

## Licence

To be added.
