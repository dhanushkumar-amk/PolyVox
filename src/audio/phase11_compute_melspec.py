import sys
from pathlib import Path
from typing import Union
import librosa
import numpy as np

# Resolve root directory and default input file
BASE_DIR = Path(__file__).resolve().parent.parent.parent
# Prefer the mono-converted output from Phase 10; fallback to resampled or sample.wav
DEFAULT_AUDIO_PATH = BASE_DIR / "data" / "outputs" / "test_mono.wav"
if not DEFAULT_AUDIO_PATH.exists():
    DEFAULT_AUDIO_PATH = BASE_DIR / "data" / "outputs" / "test_resampled.wav"
if not DEFAULT_AUDIO_PATH.exists():
    DEFAULT_AUDIO_PATH = BASE_DIR / "data" / "uploads" / "sample.wav"


def compute_melspec_db(
    audio: np.ndarray,
    sr: int,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Reusable function to compute a Mel Spectrogram and its Decibel (dB) representation

    directly from a NumPy audio array and sample rate.

    Returns:
        tuple of (raw_melspec, db_melspec)
    """
    raw_melspec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    db_melspec = librosa.power_to_db(raw_melspec, ref=np.max)
    return raw_melspec, db_melspec


def compute_mel_spectrogram(
    audio_path: Union[Path, str] = DEFAULT_AUDIO_PATH,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Computes a Mel Spectrogram and its Decibel (dB) equivalent from an audio file.

    Parameters:
        audio_path: Path to the input WAV file.
        n_mels: Number of Mel frequency bands (bins) to generate (typically 80 or 128).
        n_fft: Length of the FFT window (analysis frame size in samples).
        hop_length: Number of samples between successive time frames.

    Returns:
        tuple of (raw_melspec, db_melspec, sample_rate)
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Target audio file not found: {path}")

    print("=" * 65)
    print("  PHASE 11: COMPUTE MEL SPECTROGRAM (RAW MATRIX INSPECTION)")
    print(f"  Target File: {path.name}")
    print("=" * 65 + "\n")

    # Step 1: Load audio at native sample rate as a mono array
    print(f"Loading '{path.name}' using librosa (sr=None, mono=True)...")
    y, sr = librosa.load(str(path), sr=None, mono=True)
    duration = len(y) / sr
    print(f"  Sample Rate:  {sr:,} Hz")
    print(f"  Total Audio:  {len(y):,} samples ({duration:.2f} seconds)\n")

    # Step 2: Compute Mel Spectrogram using reusable function
    print(f"Computing Mel Spectrogram with {n_mels} Mel bins, hop_length={hop_length}...")
    raw_melspec, db_melspec = compute_melspec_db(
        audio=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    print("Converting raw power values to logarithmic Decibel (dB) scale...")

    # Step 4: Explain the Shape and Dimensions
    n_bins, n_frames = db_melspec.shape
    print("\n" + "-" * 65)
    print(f"  MEL SPECTROGRAM ARRAY SHAPE: {db_melspec.shape}")
    print("-" * 65)
    print(f"  Dimension 0 (Rows) = {n_bins} Frequency 'Mel Bins'")
    print(f"    - Each row represents a specific pitch range warped to the Mel scale.")
    print(f"    - Row 0 corresponds to the lowest bass frequencies (~0 - 100 Hz).")
    print(f"    - Row {n_bins - 1} corresponds to the highest treble frequencies (~{sr // 2} Hz).")
    print(f"  Dimension 1 (Cols) = {n_frames} Time 'Frames'")
    print(f"    - Each column is a time slice of ~{hop_length / sr * 1000:.1f} milliseconds.")
    print(f"    - Total duration covered: {n_frames} frames * {hop_length} hop = {n_frames * hop_length:,} samples.\n")

    # Step 5: Print Numerical Range & Sample Values
    print("--- Numerical Ranges & Statistical Comparison ---")
    print(f"  Raw Power Scale:    Min = {raw_melspec.min():.6e}  |  Max = {raw_melspec.max():.6f}")
    print(f"  Decibel (dB) Scale: Min = {db_melspec.min():.2f} dB    |  Max = {db_melspec.max():.2f} dB")
    print(f"  Mean Energy (dB):   {db_melspec.mean():.2f} dB\n")

    # Print a small sub-matrix (top-left 5x5 corner)
    print("--- Sample Values: Top-Left Corner [First 5 Mel Bins x First 5 Time Frames] (dB) ---")
    sub_matrix = db_melspec[:5, :5]
    for row_idx, row in enumerate(sub_matrix):
        formatted_row = "  ".join(f"{val:+7.2f} dB" for val in row)
        print(f"  Bin {row_idx:02d}: [ {formatted_row} ]")

    print("\nSummary: The 1D audio waveform has been converted into a 2D matrix of shape")
    print(f"         {db_melspec.shape} ready for speech AI feature extraction or neural network processing.")
    print("=" * 65)

    return raw_melspec, db_melspec, sr


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_AUDIO_PATH
    compute_mel_spectrogram(target)
