# Privacy-Budgeted Federated Learning for EEG–ECG Seizure Analytics

Code for the paper *Privacy-Budgeted Federated Learning for EEG–ECG Seizure Analytics: An Audit of Privacy Accounting and Client-Specific Noise*.

> **Scope.** Research code on a nine-patient subset of the public SeizeIT2 dataset. It is **not** a clinical tool and not a validated seizure detector. The reported ε values are accountant outputs for the DP-SGD gradient perturbations only, at **sample level** (not patient level) and under add/remove adjacency. They do not cover the un-noised per-client statistics used for standardisation and sensitivity scoring, or class counts used for loss weighting.

## Files

| File | Purpose |
|---|---|
| `generate_seizeit2_multiclient.py` | Builds the windowed dataset (10 s, 256 Hz, 2 EEG + 1 ECG channel) from the SeizeIT2 BIDS files for sub-001…sub-009. Seizure window = any overlap with an `sz_` event; background = window entirely inside a `bckg` event; windows overlapping `impd`/other events are excluded; background sampled 2:1 per patient (seed 42). |
| `preprocess_seizeit2_multiclient.py` | Per-window demeaning, 4th-order zero-phase Butterworth band-pass (0.5–40 Hz), per-channel z-scoring. |
| `ablation_suite.py` | Main experiment. Own RDP accountant (integer orders, subsampled Gaussian, Poisson sampling), noise configurations (`nonprivate`, `fixed`, `uniform_cal`, `ceiling_only`, `adaptive`, `hybrid`, `size_only`, `size_fix`), logistic-regression and MLP models, leave-patients-out options, ε-ceiling calibration. Writes `results.csv`, `summary.csv`, `setup.json` and per-seed predictions. |
| `aggregate_lopo.py` | Aggregates leave-one-patient-out folds: per-patient AUC, bootstrap CI over patients, paired differences with exact sign-flip tests. |
| `check_dp_accounting.py` | Cross-checks the accountant against closed-form arithmetic, `dp_accounting` and Opacus. |
| `federated_seizeit2_adaptive_dp_fedprox.py` | Earlier prototype (shuffled batches, greedy ceiling rule). Its ε values use shuffled-batch accounting and are **not** those reported in the Poisson tables of the paper. Kept for transparency. |

## Data

SeizeIT2 is **not** included. Download it from its public source (see the paper's reference [19]) and set the path constants at the top of `generate_seizeit2_multiclient.py` (`DATASET_ROOT`, `OUTPUT_DIR`). The scripts currently contain Windows paths from the authors' machine; edit them before running.

## Reproducing

```bash
python generate_seizeit2_multiclient.py       # raw windows
python preprocess_seizeit2_multiclient.py     # filtered, z-scored windows
python ablation_suite.py --help               # experiment options
python aggregate_lopo.py <root_dir>           # root_dir holds one sub-folder per held-out patient with preds/*.npz
python check_dp_accounting.py                 # accountant cross-checks
```

**LOPO statistics** (`aggregate_lopo.py`): the resampling unit is the held-out patient; each patient contributes one AUC per configuration (seeds averaged first) with equal weight. Confidence intervals are percentile bootstraps over patients (20 000 resamples); paired differences use an exact two-sided sign-flip test over all 2^n sign patterns. Zero differences stay in the enumeration and are counted as neither better nor worse. Patients with fewer than two positives or two negatives are excluded.

The leave-one-patient-out results in the paper were produced by running `ablation_suite.py` once per held-out patient (via `--test-patients`) with logistic regression, global scaling, and two to five seeds per fold, then aggregating with `aggregate_lopo.py`.

## Environment

Python 3.14. Main packages: `numpy`, `pandas`, `scipy`, `scikit-learn`, `torch`, `mne`. The accountant cross-check additionally uses `dp_accounting` and `opacus`.

## Citation

Paper under review; citation to be added.

## Licence

To be added.
