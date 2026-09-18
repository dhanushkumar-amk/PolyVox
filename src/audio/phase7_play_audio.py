import sys
from pathlib import Path
from typing import Union
import librosa
import numpy as np
import sounddevice as sd

# Resolve the project root directory and default audio sample
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_AUDIO_PATH = BASE_DIR / "data" / "uploads" / "sample.wav"


def play_audio_array(audio: np.ndarray, sample_rate: int) -> None:
    """Plays any NumPy audio array through the speakers using sounddevice.

    This function is reusable for directly previewing generated/cloned audio
    in-memory without having to save it to disk first.
    """
    print("Playing audio...")
    sd.play(audio, samplerate=sample_rate)
    sd.wait()  # Block execution until the audio stream finishes playing
    print("Playback finished.")


def play_audio_file(file_path: Union[Path, str] = DEFAULT_AUDIO_PATH) -> None:
    """Loads a target audio file using librosa and plays it back through the speakers."""
    path = Path(file_path)
    if not path.exists():
        print(f"Error: Target audio file not found at {path}")
        return

    print(f"Loading '{path.name}' using librosa...")
    # Load audio: sr=None preserves original sampling rate, returns 1D float32 NumPy array
    audio, sr = librosa.load(str(path), sr=None)
    duration = len(audio) / sr
    print(f"File Info: {len(audio):,} samples, {sr} Hz, {duration:.2f}s duration")

    play_audio_array(audio, sr)


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_AUDIO_PATH
    play_audio_file(target)
