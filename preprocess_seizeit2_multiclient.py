import os
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

# -------------------------------------------------------------------------
# Configuration & Constants
# -------------------------------------------------------------------------
INPUT_DIR = r"C:\Users\lokes\Downloads\project1\seizeit2_multiclient_dataset"
OUTPUT_DIR = r"C:\Users\lokes\Downloads\project1\seizeit2_multiclient_preprocessed"

SFREQ = 256
LOWCUT_HZ = 0.5
HIGHCUT_HZ = 40.0
FILTER_ORDER = 4
EPSILON = 1e-8


def create_bandpass_filter(lowcut, highcut, sfreq, order=4):
    """
    Designs a digital Butterworth bandpass filter in Second-Order Sections (SOS) format.
    """
    sos = butter(
        N=order,
        Wn=[lowcut, highcut],
        btype="bandpass",
        output="sos",
        fs=sfreq
    )
    return sos


def preprocess_signal(X, sos):
    """
    Preprocesses multi-channel windowed signals (shape: (N, C, T)):
      1. Demeans each channel and window independently along the time axis.
      2. Applies zero-phase Butterworth bandpass filtering (0.5 - 40 Hz) using sosfiltfilt.
      3. Z-score normalizes each channel and window independently.
      4. Handles zero / near-zero standard deviation safely.
    """
    # Step 1: Remove mean along time axis (axis=-1)
    mean_raw = np.mean(X, axis=-1, keepdims=True)
    X_demeaned = X - mean_raw

    # Step 2: Zero-phase bandpass filtering along time axis
    X_filtered = sosfiltfilt(sos, X_demeaned, axis=-1)

    # Step 3: Z-score normalization per channel per window
    mean_filt = np.mean(X_filtered, axis=-1, keepdims=True)
    std_filt = np.std(X_filtered, axis=-1, keepdims=True)

    # Step 4: Safe standard deviation to prevent division by zero
    std_safe = np.where(std_filt < EPSILON, 1.0, std_filt)
    X_norm = (X_filtered - mean_filt) / std_safe

    return X_norm.astype(np.float32)


def main():
    print("=" * 70)
    print("SEIZEIT2 MULTI-CLIENT DATASET PREPROCESSING PIPELINE")
    print("=" * 70)
    print(f"Input Directory : {INPUT_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}\n")

    # ---------------------------------------------------------------------
    # 1. Load Raw Multi-Client Dataset
    # ---------------------------------------------------------------------
    # Handle both X_eeg.npy / eeg.npy and y.npy / labels.npy naming conventions
    eeg_in_path = os.path.join(INPUT_DIR, "X_eeg.npy")
    if not os.path.exists(eeg_in_path):
        eeg_in_path = os.path.join(INPUT_DIR, "eeg.npy")

    ecg_in_path = os.path.join(INPUT_DIR, "X_ecg.npy")
    if not os.path.exists(ecg_in_path):
        ecg_in_path = os.path.join(INPUT_DIR, "ecg.npy")

    y_in_path = os.path.join(INPUT_DIR, "y.npy")
    if not os.path.exists(y_in_path):
        y_in_path = os.path.join(INPUT_DIR, "labels.npy")

    meta_in_path = os.path.join(INPUT_DIR, "metadata.csv")

    for path in [eeg_in_path, ecg_in_path, y_in_path, meta_in_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required input dataset file: {path}")

    X_eeg_raw = np.load(eeg_in_path)
    X_ecg_raw = np.load(ecg_in_path)
    labels = np.load(y_in_path)
    metadata = pd.read_csv(meta_in_path)

    N = len(labels)

    print("RAW INPUT DATASET SUMMARY:")
    print(f"  EEG Input Shape   : {X_eeg_raw.shape}, dtype = {X_eeg_raw.dtype}")
    print(f"  ECG Input Shape   : {X_ecg_raw.shape}, dtype = {X_ecg_raw.dtype}")
    print(f"  Labels Shape      : {labels.shape}, dtype = {labels.dtype}")
    print(f"  Metadata Table    : {metadata.shape}")
    print(f"  Patients in Meta  : {sorted(metadata['patient_id'].unique().tolist())}")

    # ---------------------------------------------------------------------
    # 2. Filter Design & Preprocessing
    # ---------------------------------------------------------------------
    print("\n" + "-" * 70)
    print(f"Applying Butterworth Bandpass ({LOWCUT_HZ}-{HIGHCUT_HZ} Hz) & Z-score Normalization...")
    print("-" * 70)

    sos = create_bandpass_filter(LOWCUT_HZ, HIGHCUT_HZ, SFREQ, order=FILTER_ORDER)

    eeg_proc = preprocess_signal(X_eeg_raw, sos)
    ecg_proc = preprocess_signal(X_ecg_raw, sos)

    print("PROCESSED SIGNAL STATISTICS:")
    print("  --- EEG ---")
    print(f"    Shape : {eeg_proc.shape}, dtype = {eeg_proc.dtype}")
    print(f"    Min   : {eeg_proc.min():.4f}, Max : {eeg_proc.max():.4f}")
    print(f"    Mean  : {eeg_proc.mean():.6e} (approx 0)")
    print(f"    Std   : {eeg_proc.std():.4f} (approx 1)")

    print("  --- ECG ---")
    print(f"    Shape : {ecg_proc.shape}, dtype = {ecg_proc.dtype}")
    print(f"    Min   : {ecg_proc.min():.4f}, Max : {ecg_proc.max():.4f}")
    print(f"    Mean  : {ecg_proc.mean():.6e} (approx 0)")
    print(f"    Std   : {ecg_proc.std():.4f} (approx 1)")

    # ---------------------------------------------------------------------
    # 3. Validation Checks
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VALIDATION REPORT")
    print("=" * 70)

    checks = {}

    # Shapes preservation
    checks["EEG shape preserved (N, 2, 2560)"] = (eeg_proc.shape == (N, 2, 2560))
    checks["ECG shape preserved (N, 1, 2560)"] = (ecg_proc.shape == (N, 1, 2560))
    checks["Labels shape preserved (N,)"] = (labels.shape == (N,))
    checks["Metadata length aligned (N rows)"] = (len(metadata) == N)

    # Dtypes
    checks["EEG dtype is float32"] = (eeg_proc.dtype == np.float32)
    checks["ECG dtype is float32"] = (ecg_proc.dtype == np.float32)
    checks["Labels dtype is int64"] = (labels.dtype == np.int64)

    # NaN / Inf checks
    eeg_nan = int(np.isnan(eeg_proc).sum())
    eeg_inf = int(np.isinf(eeg_proc).sum())
    ecg_nan = int(np.isnan(ecg_proc).sum())
    ecg_inf = int(np.isinf(ecg_proc).sum())

    checks["EEG has zero NaN/Inf"] = (eeg_nan == 0 and eeg_inf == 0)
    checks["ECG has zero NaN/Inf"] = (ecg_nan == 0 and ecg_inf == 0)

    # Label alignment & counts
    unique_lbls, counts = np.unique(labels, return_counts=True)
    label_dist = dict(zip([int(k) for k in unique_lbls], [int(v) for v in counts]))
    print(f"Label Counts Across Cohort : {label_dist} (0: Background, 1: Seizure)")
    checks["Metadata labels match labels.npy"] = np.array_equal(metadata["label"].values, labels)
    checks["Metadata sample_id matches array index"] = np.array_equal(metadata["sample_id"].values, np.arange(N))

    # Patient counts
    patient_summary = metadata.groupby(["patient_id", "label"]).size().unstack(fill_value=0)
    if 1 not in patient_summary.columns:
        patient_summary[1] = 0
    if 0 not in patient_summary.columns:
        patient_summary[0] = 0
    patient_summary = patient_summary.rename(columns={0: "background", 1: "seizure"})
    patient_summary["total"] = patient_summary["background"] + patient_summary["seizure"]

    print("\nPATIENT BREAKDOWN:")
    print(patient_summary.to_string())
    checks["Multi-client patient count > 0"] = len(patient_summary) > 0

    # Z-score normalization validation per window
    eeg_means = np.mean(eeg_proc, axis=-1)
    eeg_stds = np.std(eeg_proc, axis=-1)
    checks["EEG windows mean ~ 0 (max error < 1e-4)"] = bool(np.max(np.abs(eeg_means)) < 1e-4)
    checks["EEG windows std ~ 1 (max error < 1e-4)"] = bool(np.max(np.abs(eeg_stds - 1.0)) < 1e-4)

    ecg_means = np.mean(ecg_proc, axis=-1)
    ecg_stds = np.std(ecg_proc, axis=-1)
    checks["ECG windows mean ~ 0 (max error < 1e-4)"] = bool(np.max(np.abs(ecg_means)) < 1e-4)
    checks["ECG windows std ~ 1 (max error < 1e-4)"] = bool(np.max(np.abs(ecg_stds - 1.0)) < 1e-4)

    all_passed = True
    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status:4s}] {name}")

    print("=" * 70)
    if not all_passed:
        raise ValueError("Validation failed. Aborting dataset save.")
    print(">>> ALL VALIDATION CHECKS PASSED <<<\n")

    # ---------------------------------------------------------------------
    # 4. Save Preprocessed Multi-Client Dataset
    # ---------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    eeg_out_path = os.path.join(OUTPUT_DIR, "eeg.npy")
    ecg_out_path = os.path.join(OUTPUT_DIR, "ecg.npy")
    labels_out_path = os.path.join(OUTPUT_DIR, "labels.npy")
    meta_out_path = os.path.join(OUTPUT_DIR, "metadata.csv")

    # Save primary required outputs
    np.save(eeg_out_path, eeg_proc)
    np.save(ecg_out_path, ecg_proc)
    np.save(labels_out_path, labels)
    metadata.to_csv(meta_out_path, index=False)

    # Also save client summary report
    summary_out_path = os.path.join(OUTPUT_DIR, "client_summary.csv")
    patient_summary.to_csv(summary_out_path)

    print(f"Saved preprocessed dataset files to: {OUTPUT_DIR}")
    print(f"  - eeg.npy      : {eeg_proc.shape} ({eeg_proc.dtype})")
    print(f"  - ecg.npy      : {ecg_proc.shape} ({ecg_proc.dtype})")
    print(f"  - labels.npy   : {labels.shape} ({labels.dtype})")
    print(f"  - metadata.csv : {len(metadata)} rows")
    print(f"  - client_summary.csv")
    print("=" * 70)
    print("MULTI-CLIENT PREPROCESSING COMPLETED")


if __name__ == "__main__":
    main()
