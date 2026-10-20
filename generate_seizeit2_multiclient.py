import os
import glob
import re
import mne
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------
# Dataset Configuration & Constants
# -------------------------------------------------------------------------
DATASET_ROOT = r"C:\Users\lokes\Downloads\project1\seizeIT2\on005873"
OUTPUT_DIR = r"C:\Users\lokes\Downloads\project1\seizeit2_multiclient_dataset"

TARGET_SUBJECTS = [f"sub-{i:03d}" for i in range(1, 10)]  # sub-001 through sub-009

SFREQ = 256
WINDOW_SEC = 10
SAMPLES_PER_WINDOW = SFREQ * WINDOW_SEC  # 2560 samples
RANDOM_SEED = 42


def parse_bids_filename(filename):
    """
    Parses subject, session, task, and run from a BIDS filename.
    """
    sub_match = re.search(r"sub-([a-zA-Z0-9]+)", filename)
    ses_match = re.search(r"ses-([a-zA-Z0-9]+)", filename)
    run_match = re.search(r"run-(\d+)", filename)
    task_match = re.search(r"task-([a-zA-Z0-9]+)", filename)

    sub = f"sub-{sub_match.group(1)}" if sub_match else None
    ses = f"ses-{ses_match.group(1)}" if ses_match else "ses-01"
    run = int(run_match.group(1)) if run_match else None
    task = task_match.group(1) if task_match else None

    return sub, ses, task, run


def load_run_events(events_path):
    """
    Loads and categorizes events from a BIDS events.tsv file.
    Returns:
      sz_events: list of dicts with onset, end, type
      bckg_events: list of dicts with onset, end, type
      impd_events: list of dicts with onset, end, type
      other_events: list of dicts with onset, end, type
    """
    sz_events = []
    bckg_events = []
    impd_events = []
    other_events = []

    if not os.path.exists(events_path):
        return sz_events, bckg_events, impd_events, other_events

    try:
        df = pd.read_csv(events_path, sep="\t")
        if "eventType" not in df.columns:
            return sz_events, bckg_events, impd_events, other_events

        for _, row in df.iterrows():
            ev_type = str(row["eventType"]).strip()
            onset = float(row.get("onset", 0.0))
            duration = float(row.get("duration", 0.0))
            ev_end = onset + duration

            ev_info = {
                "type": ev_type,
                "onset": onset,
                "end": ev_end,
                "duration": duration
            }

            if ev_type == "impd":
                impd_events.append(ev_info)
            elif ev_type.startswith("sz_"):
                sz_events.append(ev_info)
            elif ev_type == "bckg":
                bckg_events.append(ev_info)
            else:
                other_events.append(ev_info)
    except Exception as e:
        print(f"Warning: Could not read events file {events_path}: {e}")

    return sz_events, bckg_events, impd_events, other_events


def scan_and_extract_candidate_windows(subject_id, dataset_root):
    """
    Scans all runs for a single subject and extracts candidate seizure and background windows.
    Adheres to:
      1. Seizure (1): Complete 10s window overlaps a seizure event starting with sz_.
      2. Background (0): Complete 10s window lies strictly inside a bckg event.
      3. Exclusion: Windows overlapping impd or any other non-seizure/non-bckg event are excluded.
      4. Unlabeled time is never treated as background.
    """
    sub_dir = os.path.join(dataset_root, subject_id)
    if not os.path.isdir(sub_dir):
        print(f"Warning: {subject_id} directory not found.")
        return [], []

    session_dirs = sorted(glob.glob(os.path.join(sub_dir, "ses-*")))
    if not session_dirs:
        session_dirs = [sub_dir]

    patient_seizure_windows = []
    patient_background_windows = []

    for ses_path in session_dirs:
        ses_id = os.path.basename(ses_path) if "ses-" in os.path.basename(ses_path) else "ses-01"
        eeg_dir = os.path.join(ses_path, "eeg")
        ecg_dir = os.path.join(ses_path, "ecg")

        if not os.path.isdir(eeg_dir) or not os.path.isdir(ecg_dir):
            continue

        eeg_files = sorted(glob.glob(os.path.join(eeg_dir, "*_eeg.edf")))

        for eeg_file in eeg_files:
            _, _, _, run_num = parse_bids_filename(eeg_file)
            if run_num is None:
                continue

            ecg_file = os.path.join(
                ecg_dir, f"{subject_id}_{ses_id}_task-szMonitoring_run-{run_num:02d}_ecg.edf"
            )
            events_file = os.path.join(
                eeg_dir, f"{subject_id}_{ses_id}_task-szMonitoring_run-{run_num:02d}_events.tsv"
            )

            if not os.path.exists(ecg_file) or not os.path.exists(events_file):
                continue

            # Load events
            sz_events, bckg_events, impd_events, other_events = load_run_events(events_file)
            exclusion_events = impd_events + other_events

            # Inspect EDF headers with preload=False
            raw_eeg = mne.io.read_raw_edf(eeg_file, preload=False, verbose=False)
            raw_ecg = mne.io.read_raw_edf(ecg_file, preload=False, verbose=False)

            # Validate sampling frequency & channels
            if raw_eeg.info["sfreq"] != SFREQ or raw_ecg.info["sfreq"] != SFREQ:
                print(f"Skipping {subject_id} run-{run_num:02d}: sfreq != {SFREQ}")
                continue
            if len(raw_eeg.ch_names) != 2 or len(raw_ecg.ch_names) != 1:
                print(f"Skipping {subject_id} run-{run_num:02d}: unexpected channel counts")
                continue

            run_duration = min(
                raw_eeg.n_times / raw_eeg.info["sfreq"],
                raw_ecg.n_times / raw_ecg.info["sfreq"]
            )

            # Generate candidate non-overlapping 10-second windows
            start_times = np.arange(0, run_duration - WINDOW_SEC + 1e-5, WINDOW_SEC)

            for w_start in start_times:
                w_end = w_start + WINDOW_SEC

                # Rule 3: Exclude if overlaps impd or any other unwanted event
                overlaps_exclusion = any(
                    max(w_start, ev["onset"]) < min(w_end, ev["end"])
                    for ev in exclusion_events
                )
                if overlaps_exclusion:
                    continue

                # Rule 1: Check Seizure (overlaps sz_ event)
                sz_overlap = [
                    ev for ev in sz_events
                    if max(w_start, ev["onset"]) < min(w_end, ev["end"])
                ]

                if sz_overlap:
                    patient_seizure_windows.append({
                        "patient_id": subject_id,
                        "session_id": ses_id,
                        "run_id": run_num,
                        "start_sec": float(w_start),
                        "end_sec": float(w_end),
                        "duration_sec": WINDOW_SEC,
                        "sampling_rate_hz": SFREQ,
                        "label": 1,
                        "event_type": sz_overlap[0]["type"],
                        "eeg_file": eeg_file,
                        "ecg_file": ecg_file
                    })
                    continue

                # Rule 2: Check Background (completely inside a bckg event)
                inside_bckg = any(
                    (w_start >= ev["onset"] and w_end <= ev["end"])
                    for ev in bckg_events
                )

                if inside_bckg:
                    patient_background_windows.append({
                        "patient_id": subject_id,
                        "session_id": ses_id,
                        "run_id": run_num,
                        "start_sec": float(w_start),
                        "end_sec": float(w_end),
                        "duration_sec": WINDOW_SEC,
                        "sampling_rate_hz": SFREQ,
                        "label": 0,
                        "event_type": "bckg",
                        "eeg_file": eeg_file,
                        "ecg_file": ecg_file
                    })

    return patient_seizure_windows, patient_background_windows


def extract_window_signals(selected_windows):
    """
    Extracts paired EEG (2ch) and ECG (1ch) signals with preload=False.
    Sorts windows by (eeg_file, start_sec) to avoid redundant file openings.
    """
    sorted_windows = sorted(
        selected_windows,
        key=lambda w: (w["patient_id"], w["session_id"], w["run_id"], w["start_sec"])
    )

    all_eeg = []
    all_ecg = []
    metadata_list = []

    current_eeg_file = None
    raw_eeg = None
    raw_ecg = None

    print(f"\nExtracting signal data for {len(sorted_windows)} selected windows...")

    for idx, win in enumerate(sorted_windows):
        eeg_file = win["eeg_file"]
        ecg_file = win["ecg_file"]

        if eeg_file != current_eeg_file:
            raw_eeg = mne.io.read_raw_edf(eeg_file, preload=False, verbose=False)
            raw_ecg = mne.io.read_raw_edf(ecg_file, preload=False, verbose=False)
            current_eeg_file = eeg_file

        start_samp = int(round(win["start_sec"] * SFREQ))
        end_samp = start_samp + SAMPLES_PER_WINDOW

        # Slice 10-second segments
        eeg_seg = raw_eeg[:, start_samp:end_samp][0].astype(np.float32)
        ecg_seg = raw_ecg[:, start_samp:end_samp][0].astype(np.float32)

        if eeg_seg.shape != (2, SAMPLES_PER_WINDOW) or ecg_seg.shape != (1, SAMPLES_PER_WINDOW):
            print(f"Warning: unexpected shape for {win['patient_id']} run-{win['run_id']} at {win['start_sec']}s")
            continue

        all_eeg.append(eeg_seg)
        all_ecg.append(ecg_seg)

        metadata_list.append({
            "sample_id": len(metadata_list),
            "patient_id": win["patient_id"],
            "session_id": win["session_id"],
            "run_id": win["run_id"],
            "start_sec": win["start_sec"],
            "end_sec": win["end_sec"],
            "duration_sec": win["duration_sec"],
            "sampling_rate_hz": win["sampling_rate_hz"],
            "label": win["label"],
            "event_type": win["event_type"]
        })

        if (len(metadata_list)) % 50 == 0:
            print(f"  Extracted {len(metadata_list)}/{len(sorted_windows)} windows...")

    X_eeg = np.stack(all_eeg, axis=0)
    X_ecg = np.stack(all_ecg, axis=0)
    y = np.array([m["label"] for m in metadata_list], dtype=np.int64)
    meta_df = pd.DataFrame(metadata_list)

    return X_eeg, X_ecg, y, meta_df


def validate_dataset(X_eeg, X_ecg, y, meta_df):
    """
    Validates shapes, dtypes, NaN/Inf, duplicates, and metadata alignment before saving.
    """
    print("\n" + "=" * 65)
    print("PRE-SAVING DATASET VALIDATION")
    print("=" * 65)

    checks = {}
    N = len(y)

    # 1. Shapes
    checks["EEG shape is (N, 2, 2560)"] = (X_eeg.shape == (N, 2, 2560))
    checks["ECG shape is (N, 1, 2560)"] = (X_ecg.shape == (N, 1, 2560))
    checks["labels shape is (N,)"] = (y.shape == (N,))
    checks["metadata rows == N"] = (len(meta_df) == N)

    # 2. Dtypes
    checks["EEG dtype is float32"] = (X_eeg.dtype == np.float32)
    checks["ECG dtype is float32"] = (X_ecg.dtype == np.float32)
    checks["y dtype is int64"] = (y.dtype == np.int64)

    # 3. NaN and Inf
    checks["No NaN/Inf in EEG"] = bool(np.isnan(X_eeg).sum() == 0 and np.isinf(X_eeg).sum() == 0)
    checks["No NaN/Inf in ECG"] = bool(np.isnan(X_ecg).sum() == 0 and np.isinf(X_ecg).sum() == 0)

    # 4. Duplicate windows
    dup_count = int(
        meta_df.duplicated(subset=["patient_id", "session_id", "run_id", "start_sec", "end_sec"]).sum()
    )
    checks["No duplicate windows"] = (dup_count == 0)

    # 5. Metadata and Label alignment
    checks["Metadata sample_id matches array indices"] = np.array_equal(
        meta_df["sample_id"].values, np.arange(N)
    )
    checks["Metadata labels match y array"] = np.array_equal(meta_df["label"].values, y)

    # 6. Patient IDs preserved
    unique_patients = sorted(meta_df["patient_id"].unique().tolist())
    checks["Patient IDs preserved"] = len(unique_patients) > 0

    # 7. No impd events
    checks["No impd events in dataset"] = bool("impd" not in meta_df["event_type"].values)

    all_passed = True
    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status:4s}] {name}")

    print("=" * 65)
    if not all_passed:
        raise ValueError("Dataset validation failed. Saving aborted.")
    print(">>> ALL VALIDATION CHECKS PASSED <<<\n")


def main():
    print("=" * 65)
    print("SEIZEIT2 MULTI-CLIENT DATASET GENERATOR (sub-001 to sub-009)")
    print("=" * 65)
    print(f"Source Directory: {DATASET_ROOT}")
    print(f"Target Directory: {OUTPUT_DIR}\n")

    rng = np.random.default_rng(RANDOM_SEED)

    all_selected_windows = []
    patient_summary_records = []

    for sub_id in TARGET_SUBJECTS:
        print(f"Scanning {sub_id}...")
        sz_wins, bg_wins = scan_and_extract_candidate_windows(sub_id, DATASET_ROOT)

        num_sz = len(sz_wins)
        num_bg_avail = len(bg_wins)

        # 2:1 background-to-seizure sampling per patient
        if num_sz > 0:
            num_bg_target = min(num_bg_avail, num_sz * 2)
            if num_bg_target > 0:
                sampled_indices = rng.choice(num_bg_avail, size=num_bg_target, replace=False)
                sampled_bg = [bg_wins[i] for i in sampled_indices]
            else:
                sampled_bg = []
        else:
            sampled_bg = []

        patient_selected = sz_wins + sampled_bg
        all_selected_windows.extend(patient_selected)

        patient_summary_records.append({
            "patient_id": sub_id,
            "seizure_windows": num_sz,
            "available_background": num_bg_avail,
            "selected_background": len(sampled_bg),
            "total_windows": len(patient_selected)
        })

        print(
            f"  {sub_id}: {num_sz} Seizures, {len(sampled_bg)} Selected BG (from {num_bg_avail} candidates), Total = {len(patient_selected)}"
        )

    # ---------------------------------------------------------------------
    # Cohort Extraction
    # ---------------------------------------------------------------------
    if not all_selected_windows:
        raise RuntimeError("No candidate windows found across cohort.")

    X_eeg, X_ecg, y, meta_df = extract_window_signals(all_selected_windows)

    # Validation
    validate_dataset(X_eeg, X_ecg, y, meta_df)

    # ---------------------------------------------------------------------
    # Save Dataset
    # ---------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    eeg_out = os.path.join(OUTPUT_DIR, "X_eeg.npy")
    ecg_out = os.path.join(OUTPUT_DIR, "X_ecg.npy")
    y_out = os.path.join(OUTPUT_DIR, "y.npy")
    meta_out = os.path.join(OUTPUT_DIR, "metadata.csv")
    summary_out = os.path.join(OUTPUT_DIR, "client_summary.csv")

    np.save(eeg_out, X_eeg)
    np.save(ecg_out, X_ecg)
    np.save(y_out, y)
    meta_df.to_csv(meta_out, index=False)

    summary_df = pd.DataFrame(patient_summary_records)
    summary_df.to_csv(summary_out, index=False)

    print("=" * 65)
    print("PATIENT-BY-PATIENT CLIENT SUMMARY")
    print("=" * 65)
    print(summary_df.to_string(index=False))
    print("-" * 65)
    print(f"Total Cohort Windows : {len(y)}")
    print(f"Total Seizure Windows: {int(np.sum(y == 1))}")
    print(f"Total Background Wins: {int(np.sum(y == 0))}")
    print(f"Saved dataset to     : {OUTPUT_DIR}")
    print("=" * 65)
    print("MULTI-CLIENT DATASET GENERATION COMPLETED")


if __name__ == "__main__":
    main()
