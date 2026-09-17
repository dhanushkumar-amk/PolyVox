import contextlib
import io
import os
import sys
from pathlib import Path

# Resolve base project path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
AUDIO_FILE = BASE_DIR / "data" / "uploads" / "sample.wav"


def compare_audio_loaders(file_path: Path):
    print("=" * 65)
    print("  PHASE 6: COMPARING AUDIO LOADERS (soundfile vs librosa vs torchaudio)")
    print(f"  Target File: {file_path}")
    print("=" * 65 + "\n")

    if not file_path.exists():
        print(f"Error: Target audio file not found at {file_path}")
        return

    # =========================================================================
    # METHOD 1: soundfile (Low-level, raw, unopinionated reader)
    # Reads directly from disk into a NumPy array without implicit transformations.
    # =========================================================================
    print("--- [Method 1] soundfile ---")
    try:
        import soundfile as sf

        # soundfile.read returns (numpy.ndarray, int: sample_rate)
        data_sf, sr_sf = sf.read(file_path, dtype="float32")
        print("Status:        Success")
        print(f"Sample Rate:   {sr_sf} Hz")
        print(f"Data Shape:    {data_sf.shape} (1D array for mono)")
        print(f"Data Type:     {type(data_sf).__name__} (dtype={data_sf.dtype})")
        print(f"First Sample:  {data_sf[0]:+.6f}")
    except Exception as e:
        print(f"Status:        Failed ({e})")
    print()

    # =========================================================================
    # METHOD 2: librosa (High-level audio analysis and feature extraction)
    # In librosa, audio is read into a float32 NumPy array.
    # Note: On Windows 11 with Smart App Control (SAC), newly downloaded Numba
    # extension binaries on Python 3.14 may trigger OS App Control policies.
    # =========================================================================
    print("--- [Method 2] librosa ---")
    try:
        import librosa

        # librosa.load returns (numpy.ndarray, int: sample_rate)
        data_lb, sr_lb = librosa.load(file_path, sr=None)
        print("Status:        Success")
        print(f"Sample Rate:   {sr_lb} Hz")
        print(f"Data Shape:    {data_lb.shape}")
        print(f"Data Type:     {type(data_lb).__name__} (dtype={data_lb.dtype})")
        print(f"First Sample:  {data_lb[0]:+.6f}")
    except ImportError as e:
        print("Status:        OS / Application Control Limitation on Python 3.14")
        print(f"Detail:        {e}")
        print("Explanation:   Windows 11 Smart App Control blocked Numba's unsigned C-extension")
        print("               (_internal.pyd). Once approved or run on Python 3.11/3.12, librosa")
        print("               returns a 1D float32 NumPy array.")
    except Exception as e:
        print(f"Status:        Failed ({e})")
    print()

    # =========================================================================
    # METHOD 3: torchaudio (PyTorch-native audio loading & tensor pipeline)
    # Starting in TorchAudio 2.9+, torchaudio.load() delegates to TorchCodec
    # which requires FFmpeg C-libraries (DLLs) on Windows.
    # When FFmpeg DLLs are not on PATH, PyTorch pipelines ingest audio via
    # torch.from_numpy(soundfile.read(...)) into a PyTorch Tensor.
    # =========================================================================
    print("--- [Method 3] torchaudio ---")
    torchaudio_loaded = False
    try:
        import torch
        import torchaudio

        # Suppress internal probe stderr trace if TorchCodec searches for FFmpeg versions
        stderr_trap = io.StringIO()
        with contextlib.redirect_stderr(stderr_trap):
            waveform, sr_ta = torchaudio.load(file_path)

        print("Status:        Success (Native torchaudio.load)")
        print(f"Sample Rate:   {sr_ta} Hz")
        print(f"Data Shape:    {list(waveform.shape)} (2D Tensor: [channels, time])")
        print(f"Data Type:     {type(waveform).__name__} (dtype={waveform.dtype})")
        print(f"First Sample:  {waveform[0, 0].item():+.6f}")
        torchaudio_loaded = True
    except (ImportError, OSError):
        print("Status:        Dependency Requirement (FFmpeg DLLs needed for TorchCodec)")
        print("Explanation:   TorchAudio 2.9+ on Windows requires FFmpeg C-libraries on PATH")
        print("               for torchaudio.load().")

    if not torchaudio_loaded:
        import torch
        import soundfile as sf

        raw_audio, sr_fallback = sf.read(file_path, dtype="float32")
        # Standard PyTorch audio tensor format: [channels, num_samples]
        tensor = torch.from_numpy(raw_audio).unsqueeze(0)
        print("\n  -> PyTorch Tensor Audio Pipeline (Direct torch.from_numpy):")
        print(f"     Sample Rate:   {sr_fallback} Hz")
        print(f"     Tensor Shape:  {list(tensor.shape)} (2D Tensor: [channels, time])")
        print(f"     Tensor Type:   {type(tensor).__name__} (dtype={tensor.dtype})")
        print(f"     First Sample:  {tensor[0, 0].item():+.6f}")
    print()


if __name__ == "__main__":
    compare_audio_loaders(AUDIO_FILE)
