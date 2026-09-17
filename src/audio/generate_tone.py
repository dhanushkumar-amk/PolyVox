from pathlib import Path
import numpy as np
import soundfile as sf

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_PATH = BASE_DIR / "data" / "uploads" / "sample.wav"


def generate_sample_tone(
    output_path: Path = OUTPUT_PATH,
    duration: float = 2.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
):
    """Generates a synthetic 440 Hz sine wave tone and saves it as a 16-bit PCM WAV file."""
    # Generate time steps: from 0 to duration
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # 440 Hz sine wave (Concert A note) with peak amplitude 0.5 to prevent clipping
    audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio_data.astype(np.float32), sample_rate, subtype="PCM_16")
    print(f"Sample WAV created at: {output_path}")
    print(f"Tone: {frequency} Hz | Duration: {duration}s | Sample Rate: {sample_rate} Hz")


if __name__ == "__main__":
    generate_sample_tone()
