import sys
from pathlib import Path
import soundfile as sf

# Resolve the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_AUDIO_PATH = BASE_DIR / "data" / "uploads" / "sample.wav"


def main():
    # Allow passing a custom audio file path via command-line argument
    audio_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_AUDIO_PATH

    print(f"Reading audio file: {audio_path}\n")

    if not audio_path.exists():
        print(f"Error: File not found at {audio_path}")
        return

    # 1. Read audio samples into a NumPy float32 array (-1.0 to 1.0 standard for audio/ML)
    data, sample_rate = sf.read(audio_path, dtype="float32")

    # 2. Extract on-disk file metadata for additional clarity
    file_info = sf.info(audio_path)

    # 3. Determine channels and total samples
    # Mono audio is 1D: (num_samples,)
    # Stereo / multichannel audio is 2D: (num_samples, num_channels)
    if data.ndim == 1:
        channels_desc = "1 (Mono)"
        num_samples = len(data)
    else:
        channels_desc = f"{data.shape[1]} (Stereo/Multichannel)"
        num_samples = data.shape[0]

    # 4. Calculate duration
    duration_seconds = num_samples / sample_rate

    # 5. Print clearly labeled properties
    print("--- Audio File Properties ---")
    print(f"Sample Rate:          {sample_rate} Hz")
    print(f"Number of Channels:   {channels_desc}")
    print(f"Total Samples:        {num_samples:,}")
    print(f"Duration:             {duration_seconds:.2f} seconds ({duration_seconds} s)")
    print(f"Sample Data Type:     {data.dtype} (on-disk subtype: {file_info.subtype})")

    # 6. Print first 10 raw audio sample values
    print("\n--- First 10 Raw Sample Values ---")
    for i in range(min(10, num_samples)):
        val = data[i]
        print(f"Sample [{i:02d}]: {val}")


if __name__ == "__main__":
    main()
