import sys
from pathlib import Path
from typing import Union
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Resolve project base directory and imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.audio.phase11_compute_melspec import compute_melspec_db

# Prefer the mono-converted output from Phase 10; fallback to resampled or sample.wav
DEFAULT_AUDIO_PATH = BASE_DIR / "data" / "outputs" / "test_mono.wav"
if not DEFAULT_AUDIO_PATH.exists():
    DEFAULT_AUDIO_PATH = BASE_DIR / "data" / "outputs" / "test_resampled.wav"
if not DEFAULT_AUDIO_PATH.exists():
    DEFAULT_AUDIO_PATH = BASE_DIR / "data" / "uploads" / "sample.wav"

DEFAULT_OUTPUT_IMAGE = BASE_DIR / "data" / "outputs" / "mel_spectrogram_combined.png"

# =============================================================================
# WHAT TO VISUALLY NOTICE WHEN INTERPRETING THIS COMBINED PLOT:
#
# 1. COLOR BAR MEANING (DECIBELS / LOUDNESS):
#    - The color bar represents acoustic energy on a logarithmic Decibel (dB)
#      scale normalized relative to the loudest peak (0 dB).
#    - Brighter / Warmer colors (Yellow, Orange, Pink):
#      Represent high energy / loud volume present at that exact frequency and time.
#    - Darker / Cooler colors (Deep Purple, Black):
#      Represent absence of sound, quiet background noise, or silence (< -60 to -80 dB).
#
# 2. WAVEFORM (Top Plot) vs. MEL SPECTROGRAM (Bottom Plot):
#    - Shared Time Axis (X-axis):
#      Notice how both plots line up perfectly from 0.0s to 2.0s.
#    - Pure Tones (Our Synthetic Sample):
#      In the waveform, you see continuous uniform oscillation. In the Mel spectrogram,
#      you see sharp, bright horizontal lines centered exactly at the tone's frequencies
#      (440 Hz / 660 Hz), with total darkness everywhere else!
#    - Real Human Speech (When You Upload Voice Audio Later):
#      * Silence / Pauses: Both plots will drop to near zero—waveform is a flat line,
#        and the spectrogram will look dark and empty.
#      * Vowels and Words: You will see rich, wavy bands of warm color (called Formants)
#        primarily clustered in the 100 Hz - 4000 Hz range where human vocal energy lives.
# =============================================================================


def plot_waveform_and_melspectrogram(
    audio_path: Union[Path, str] = DEFAULT_AUDIO_PATH,
    output_image: Union[Path, str] = DEFAULT_OUTPUT_IMAGE,
    n_mels: int = 128,
    hop_length: int = 512,
    show: bool = False,
) -> Path:
    """Computes and renders a stacked 2-panel figure displaying both the raw Waveform

    and the Mel Spectrogram sharing a synchronized time axis.
    """
    input_file = Path(audio_path)
    output_file = Path(output_image)

    if not input_file.exists():
        raise FileNotFoundError(f"Audio file not found: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  PHASE 12: VISUALIZE MEL SPECTROGRAM & WAVEFORM")
    print(f"  Input File:   {input_file.name}")
    print(f"  Output Image: {output_file.name}")
    print("=" * 65 + "\n")

    # Step 1: Load audio
    print(f"Loading '{input_file.name}' using librosa...")
    y, sr = librosa.load(str(input_file), sr=None, mono=True)
    duration = len(y) / sr
    print(f"  Audio Loaded: {len(y):,} samples, {sr:,} Hz, {duration:.2f}s duration\n")

    # Step 2: Compute Mel Spectrogram using reusable Phase 11 function
    print(f"Computing Mel Spectrogram ({n_mels} Mel bins, hop={hop_length})...")
    raw_melspec, db_melspec = compute_melspec_db(
        audio=y,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
    )
    print(f"  Spectrogram Matrix Shape: {db_melspec.shape} (Bins x Frames)\n")

    # Step 3: Create stacked subplots sharing the horizontal time axis
    print("Rendering combined 2-panel visualization (Waveform + Mel Spectrogram)...")
    fig, (ax_wave, ax_mel) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(11, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.8]},
        layout="constrained",
    )

    # Top Panel: Waveform (Amplitude vs. Time)
    librosa.display.waveshow(y, sr=sr, ax=ax_wave, color="#2b5c8f", alpha=0.85)
    ax_wave.set_title(f"Audio Analysis: {input_file.name} ({sr} Hz, {duration:.2f}s)", fontsize=13, fontweight="bold")
    ax_wave.set_ylabel("Amplitude\n(-1.0 to +1.0)", fontsize=10)
    ax_wave.set_ylim(-1.05, 1.05)
    ax_wave.axhline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
    ax_wave.grid(True, linestyle=":", alpha=0.5)

    # Bottom Panel: Mel Spectrogram (Mel Frequency vs. Time)
    # cmap='magma' or 'viridis' provides high contrast for human pitch perception
    img = librosa.display.specshow(
        db_melspec,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        fmax=sr // 2,
        ax=ax_mel,
        cmap="magma",
    )
    ax_mel.set_title("Mel Spectrogram (Frequency Energy Over Time)", fontsize=12, fontweight="bold")
    ax_mel.set_xlabel("Time (seconds)", fontsize=11)
    ax_mel.set_ylabel("Mel Frequency (Hz)", fontsize=11)

    # Add Color Bar for the Decibel Scale
    cbar = fig.colorbar(img, ax=ax_mel, format="%+2.0f dB")
    cbar.set_label("Energy / Loudness (dB)", fontsize=10)

    # Step 4: Save high-resolution PNG image
    fig.savefig(output_file, dpi=300)
    print(f"Combined plot saved successfully to:\n  {output_file}\n")

    # Step 5: Optional Interactive Display
    if show:
        print("Displaying interactive plot window...")
        plt.show()
    else:
        plt.close(fig)

    return output_file


if __name__ == "__main__":
    interactive_flag = "--show" in sys.argv or hasattr(sys, "ps1")
    custom_paths = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    target_audio = Path(custom_paths[0]) if custom_paths else DEFAULT_AUDIO_PATH

    plot_waveform_and_melspectrogram(audio_path=target_audio, show=interactive_flag)
