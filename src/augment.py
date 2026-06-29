"""
Data augmentation for Revio vehicle audio classification.

TWO levels of augmentation are provided:

  Audio-level  (before MFCC extraction)
    add_noise, time_stretch, pitch_shift, augment_audio
    → use in process_and_augment_dataset() for maximum realism

  MFCC-level   (after MFCC extraction, SpecAugment style)
    freq_mask, time_mask, spec_augment
    → use in augment_dataset() to cheaply expand an already-extracted split

RULE: call these functions only on the TRAINING split.
      Validation and test splits must stay unmodified.
"""

import random
from pathlib import Path

import librosa
import numpy as np

from src.preprocess import (
    DURATION,
    HOP_LENGTH,
    N_FFT,
    N_MFCC,
    N_SAMPLES,
    SAMPLE_RATE,
    extract_mfcc,
    load_audio,
)

# ---------------------------------------------------------------------------
# Reproducibility helper
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Audio-level augmentations
# Each function takes a float32 audio array and returns a float32 array of
# the SAME length (N_SAMPLES).  They are safe to compose.
# ---------------------------------------------------------------------------

def add_noise(audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """Add Gaussian white noise scaled by RMS amplitude."""
    rms = np.sqrt(np.mean(audio ** 2)) + 1e-8
    noise = np.random.randn(len(audio)).astype(np.float32)
    return audio + noise_factor * rms * noise


def time_stretch(
    audio: np.ndarray,
    rate: float | None = None,
    rate_range: tuple[float, float] = (0.85, 1.15),
) -> np.ndarray:
    """Speed up or slow down audio without changing pitch.

    rate > 1 → faster (shorter), rate < 1 → slower (longer).
    Output is padded/trimmed back to the original length.
    """
    if rate is None:
        rate = random.uniform(*rate_range)
    stretched = librosa.effects.time_stretch(y=audio, rate=rate)
    n = len(audio)
    if len(stretched) < n:
        stretched = np.pad(stretched, (0, n - len(stretched)), mode="constant")
    else:
        stretched = stretched[:n]
    return stretched.astype(np.float32)


def pitch_shift(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_steps: float | None = None,
    steps_range: tuple[float, float] = (-3.0, 3.0),
) -> np.ndarray:
    """Shift pitch by n semitones without changing duration."""
    if n_steps is None:
        n_steps = random.uniform(*steps_range)
    shifted = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)
    return shifted.astype(np.float32)


def augment_audio(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    apply_noise: bool = True,
    apply_stretch: bool = True,
    apply_pitch: bool = True,
    noise_p: float = 0.5,
    stretch_p: float = 0.5,
    pitch_p: float = 0.5,
) -> np.ndarray:
    """Apply a random combination of audio-level augmentations.

    Each transform is applied independently with its own probability.
    At least one transform is guaranteed to fire (re-rolled if none trigger).
    """
    aug = audio.copy()
    fired = False

    if apply_noise and random.random() < noise_p:
        aug = add_noise(aug)
        fired = True
    if apply_stretch and random.random() < stretch_p:
        aug = time_stretch(aug)
        fired = True
    if apply_pitch and random.random() < pitch_p:
        aug = pitch_shift(aug, sr=sr)
        fired = True

    # Guarantee at least one transform
    if not fired:
        aug = add_noise(aug)

    return aug


# ---------------------------------------------------------------------------
# MFCC-level augmentations (SpecAugment — Park et al. 2019)
# Operate on a single MFCC matrix of shape (n_mfcc, time_frames).
# ---------------------------------------------------------------------------

def freq_mask(mfcc: np.ndarray, F: int = 8) -> np.ndarray:
    """Zero out up to F consecutive frequency (MFCC coefficient) rows."""
    mfcc = mfcc.copy()
    n_mels = mfcc.shape[0]
    f = random.randint(1, min(F, n_mels - 1))
    f0 = random.randint(0, n_mels - f)
    mfcc[f0 : f0 + f, :] = 0.0
    return mfcc


def time_mask(mfcc: np.ndarray, T: int = 19) -> np.ndarray:
    """Zero out up to T consecutive time frames (max ~24% of 78 frames)."""
    mfcc = mfcc.copy()
    n_frames = mfcc.shape[1]
    t = random.randint(1, min(T, n_frames - 1))
    t0 = random.randint(0, n_frames - t)
    mfcc[:, t0 : t0 + t] = 0.0
    return mfcc


def spec_augment(
    mfcc: np.ndarray,
    num_freq_masks: int = 1,
    num_time_masks: int = 1,
    F: int = 8,
    T: int = 20,
) -> np.ndarray:
    """Apply SpecAugment: repeated frequency + time masking on one MFCC sample."""
    aug = mfcc.copy()
    for _ in range(num_freq_masks):
        aug = freq_mask(aug, F=F)
    for _ in range(num_time_masks):
        aug = time_mask(aug, T=T)
    return aug


# ---------------------------------------------------------------------------
# High-level helpers used by training notebooks / scripts
# ---------------------------------------------------------------------------

def augment_dataset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    multiplier: int = 2,
    num_freq_masks: int = 1,
    num_time_masks: int = 1,
    F: int = 8,
    T: int = 20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand the training set using SpecAugment on already-extracted MFCCs.

    For each original sample, `multiplier` augmented copies are generated and
    appended.  The originals are kept, so the returned set is
    (1 + multiplier) × N_train samples.

    Parameters
    ----------
    X_train : (N, n_mfcc, time_frames) float32
    y_train : (N,) int32
    multiplier : how many augmented copies per sample

    Returns
    -------
    X_aug, y_aug — shuffled, same dtype as inputs
    """
    set_seed(seed)
    X_aug_list = [X_train]
    y_aug_list = [y_train]

    for _ in range(multiplier):
        batch = np.array(
            [
                spec_augment(x, num_freq_masks=num_freq_masks,
                             num_time_masks=num_time_masks, F=F, T=T)
                for x in X_train
            ],
            dtype=np.float32,
        )
        X_aug_list.append(batch)
        y_aug_list.append(y_train.copy())

    X_out = np.concatenate(X_aug_list, axis=0)
    y_out = np.concatenate(y_aug_list, axis=0)

    # Shuffle so augmented copies are not always at the end
    idx = np.random.permutation(len(X_out))
    return X_out[idx], y_out[idx]


def process_and_augment_dataset(
    state_dir: str | Path,
    label_map: dict[str, int],
    sr: int = SAMPLE_RATE,
    duration: float = DURATION,
    n_mfcc: int = N_MFCC,
    multiplier: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load raw audio, apply audio-level augmentation, extract MFCCs.

    This is a heavier alternative to augment_dataset() — augmentation happens
    before MFCC extraction so time-stretch and pitch-shift affect the features
    more realistically.  Use when audio-level diversity matters more than speed.

    Returns
    -------
    X : (N_orig + N_orig × multiplier, n_mfcc, time_frames) float32
    y : (N_total,) int32
    label_names : list[str]
    """
    set_seed(seed)
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
        print(f"  [{label}] {class_name:<30s} {len(wav_files):>4d} files → "
              f"{len(wav_files) * (1 + multiplier)} after augmentation")

        for wav_path in wav_files:
            try:
                audio = load_audio(str(wav_path), sr=sr, duration=duration)

                # Original
                X_list.append(extract_mfcc(audio, sr=sr, n_mfcc=n_mfcc))
                y_list.append(label)

                # Augmented copies
                for _ in range(multiplier):
                    aug_audio = augment_audio(audio, sr=sr)
                    X_list.append(extract_mfcc(aug_audio, sr=sr, n_mfcc=n_mfcc))
                    y_list.append(label)

            except Exception as exc:
                print(f"    [ERROR] {wav_path.name}: {exc}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    # Shuffle
    idx = np.random.permutation(len(X))
    return X[idx], y[idx], label_names


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from src.preprocess import load_braking_dataset

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"

    print("=== SpecAugment path ===")
    X, y, labels = load_braking_dataset(data_dir)
    print(f"Before: X={X.shape}")
    X_aug, y_aug = augment_dataset(X, y, multiplier=2)
    print(f"After : X={X_aug.shape}  (3× original)")

    print("\n=== Audio-level augment path (Braking, 1 sample) ===")
    wav = list((Path(data_dir) / "braking state" / "normal_brakes").glob("*.wav"))[0]
    audio = load_audio(str(wav))
    aug = augment_audio(audio)
    mfcc = extract_mfcc(aug)
    print(f"Augmented MFCC shape: {mfcc.shape}  dtype={mfcc.dtype}")
    print("Smoke test PASSED")
