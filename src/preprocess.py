"""
MFCC extraction pipeline for Revio vehicle health monitoring.

Folder → fixed-length audio → MFCC (40 coefficients) → per-clip normalised feature array.
Call the three load_*_dataset() helpers from notebooks or training scripts.
"""

import sys
from pathlib import Path

import librosa
import numpy as np

# ---------------------------------------------------------------------------
# Global constants — shared with augment.py and training scripts
# ---------------------------------------------------------------------------
SAMPLE_RATE = 22050
DURATION = 1.8                              # seconds; all clips padded/trimmed to this
N_SAMPLES = int(SAMPLE_RATE * DURATION)     # 39 690
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512

# ---------------------------------------------------------------------------
# Label maps — folder name → integer class index (sorted by index in label_names)
# ---------------------------------------------------------------------------
BRAKING_LABELS: dict[str, int] = {
    "normal_brakes": 0,
    "worn_out_brakes": 1,
}

STARTUP_LABELS: dict[str, int] = {
    "normal_engine_startup": 0,
    "bad_ignition": 1,
    "dead_battery": 2,
}

IDLE_LABELS: dict[str, int] = {
    "normal_engine_idle": 0,
    "low_oil": 1,
    "power_steering": 2,
    "serpentine_belt": 3,
}

# Compound-fault sub-folders inside idle state — excluded from demo scope
IDLE_SKIP: set[str] = {"combined"}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_audio(file_path: str, sr: int = SAMPLE_RATE, duration: float = DURATION) -> np.ndarray:
    """Load a WAV file as a fixed-length mono signal.

    Shorter clips are zero-padded on the right; longer clips are trimmed.
    Returns float32 array of shape (N_SAMPLES,).
    """
    n_samples = int(sr * duration)
    # librosa.load already resamples and converts to float32 in [-1, 1]
    audio, _ = librosa.load(file_path, sr=sr, mono=True, duration=duration)
    if len(audio) < n_samples:
        audio = np.pad(audio, (0, n_samples - len(audio)), mode="constant")
    else:
        audio = audio[:n_samples]
    return audio.astype(np.float32)


def extract_mfcc(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Extract and per-clip normalise MFCCs from a fixed-length audio array.

    Returns float32 array of shape (n_mfcc, time_frames).
    With DURATION=1.8s, sr=22050, hop_length=512 → time_frames = 78.

    Per-coefficient mean/variance normalisation is applied so the CNN sees
    zero-centred inputs regardless of recording level.
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )  # shape: (n_mfcc, time_frames)

    # Normalise each coefficient independently across its time axis
    mean = mfcc.mean(axis=1, keepdims=True)
    std = mfcc.std(axis=1, keepdims=True)
    mfcc = (mfcc - mean) / (std + 1e-8)

    return mfcc.astype(np.float32)


def process_dataset(
    state_dir: str | Path,
    label_map: dict[str, int],
    sr: int = SAMPLE_RATE,
    duration: float = DURATION,
    n_mfcc: int = N_MFCC,
    skip_dirs: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract MFCCs for every class sub-folder defined in label_map.

    Parameters
    ----------
    state_dir  : root folder for one vehicle state (e.g. 'data/raw/braking state')
    label_map  : mapping of sub-folder name → integer class index
    skip_dirs  : sub-folder names to ignore (e.g. compound-fault folders)

    Returns
    -------
    X           : float32 (N, n_mfcc, time_frames)
    y           : int32   (N,)
    label_names : list[str] — label_names[i] is the class name for label i
    """
    skip_dirs = skip_dirs or set()
    state_dir = Path(state_dir)
    label_names: list[str] = [
        name for name, _ in sorted(label_map.items(), key=lambda kv: kv[1])
    ]

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for class_name, label in label_map.items():
        class_dir = state_dir / class_name
        if not class_dir.exists():
            print(f"  [WARN] Not found, skipping: {class_dir}")
            continue

        wav_files = sorted(class_dir.glob("*.wav"))
        if not wav_files:
            print(f"  [WARN] No .wav files in {class_dir}")
            continue

        print(f"  [{label}] {class_name:<30s} {len(wav_files):>4d} files")
        for wav_path in wav_files:
            try:
                audio = load_audio(str(wav_path), sr=sr, duration=duration)
                mfcc = extract_mfcc(audio, sr=sr, n_mfcc=n_mfcc)
                X_list.append(mfcc)
                y_list.append(label)
            except Exception as exc:
                print(f"    [ERROR] {wav_path.name}: {exc}")

    X = np.array(X_list, dtype=np.float32)   # (N, n_mfcc, time_frames)
    y = np.array(y_list, dtype=np.int32)      # (N,)
    return X, y, label_names


# ---------------------------------------------------------------------------
# Dataset loaders — one per vehicle state
# ---------------------------------------------------------------------------

def load_braking_dataset(
    data_dir: str = "data/raw",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load Braking state — 2 classes: normal_brakes, worn_out_brakes."""
    state_dir = Path(data_dir) / "braking state"
    print(f"Loading Braking dataset from: {state_dir}")
    return process_dataset(state_dir, BRAKING_LABELS)


def load_startup_dataset(
    data_dir: str = "data/raw",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load Start-Up state — 3 classes: normal_engine_startup, bad_ignition, dead_battery."""
    state_dir = Path(data_dir) / "startup state"
    print(f"Loading Start-Up dataset from: {state_dir}")
    return process_dataset(state_dir, STARTUP_LABELS)


def load_idle_dataset(
    data_dir: str = "data/raw",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load Idle state — 4 classes (single-fault only; combined/ folder is skipped)."""
    state_dir = Path(data_dir) / "idle state"
    print(f"Loading Idle dataset from: {state_dir}")
    return process_dataset(state_dir, IDLE_LABELS, skip_dirs=IDLE_SKIP)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"

    loaders = [
        (load_braking_dataset, "Braking"),
        (load_startup_dataset, "Start-Up"),
        (load_idle_dataset, "Idle"),
    ]

    for loader, name in loaders:
        X, y, labels = loader(data_dir)
        print(f"\n{name}: X={X.shape}  y={y.shape}  dtype=({X.dtype}, {y.dtype})")
        for i, lbl in enumerate(labels):
            count = int((y == i).sum())
            print(f"  {i}: {lbl:<30s} {count:>4d} samples")
        print()
