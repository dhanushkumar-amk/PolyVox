import sys
from pathlib import Path
from typing import Union
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Resolve the project root directory, default input sample, and output image destination
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_AUDIO_PATH = BASE_DIR / "data" / "uploads" / "sample.wav"
DEFAULT_OUTPUT_PATH = BASE_DIR / "data" / "outputs" / "waveform.png"

# =============================================================================
# WAVEFORM CONCEPTS IN PLAIN ENGLISH:
#
# 1. TIME (Horizontal X-Axis):
#    Represents the progression of the audio signal from start to finish,
#    measured in seconds. Each point along this axis corresponds to a specific
#    moment in the recording.
#
# 2. AMPLITUDE (Vertical Y-Axis):
#    Represents the sound wave's air pressure displacement / loudness at each
#    instant, normalized between -1.0 and +1.0.
#    - Values near 0.0 indicate silence or very quiet background noise.
#    - Peaks extending toward +1.0 or -1.0 represent high energy and louder sounds.
# =============================================================================


def plot_waveform(
    audio_path: Union[Path, str] = DEFAULT_AUDIO_PATH,
    output_path: Union[Path, str] = DEFAULT_OUTPUT_PATH,
    show: bool = False,
) -> Path:
    """Loads an audio file with librosa and plots its amplitude waveform over time.

    Saves the visualization to disk as a high-resolution PNG image, and optionally
    displays it in an interactive GUI window.
    """
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Ensure the output directory (e.g. data/outputs/) exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading '{audio_path.name}' using librosa...")
    # sr=None preserves original sampling rate, returning float32 NumPy array
    y, sr = librosa.load(str(audio_path), sr=None)
    duration = len(y) / sr
    print(f"Audio loaded: {len(y):,} samples, {sr} Hz ({duration:.2f}s)")

    print("Generating waveform plot...")
    # Create figure and axis using matplotlib
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")

    # librosa.display.waveshow maps the 1D sample array against time in seconds
    librosa.display.waveshow(y, sr=sr, ax=ax, color="#2b5c8f", alpha=0.85)

    # Customize title, labels, and grid
    ax.set_title(f"Audio Waveform: {audio_path.name} ({sr} Hz, {duration:.2f}s)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Amplitude (-1.0 to +1.0)", fontsize=11)
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Save the plot to the output folder (priority: works in headless & server environments)
    fig.savefig(output_path, dpi=300)
    print(f"Waveform plot saved successfully to: {output_path}")

    # Display plot if running interactively or explicitly requested via flag
    if show:
        print("Displaying interactive plot window...")
        plt.show()
    else:
        plt.close(fig)

    return output_path


if __name__ == "__main__":
    # Check if user requested interactive display via --show flag or standard interactive mode
    interactive_flag = "--show" in sys.argv or hasattr(sys, "ps1")

    # Custom audio path argument support (excluding flags)
    custom_paths = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    target_audio = Path(custom_paths[0]) if custom_paths else DEFAULT_AUDIO_PATH

    plot_waveform(audio_path=target_audio, show=interactive_flag)
