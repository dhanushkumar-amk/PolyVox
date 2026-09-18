import sys
from pathlib import Path
from typing import Union
import librosa
import numpy as np
import soundfile as sf

# Resolve root directory and import project configuration and Phase 7 playback
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from config import TARGET_SAMPLE_RATE
from src.audio.phase7_play_audio import play_audio_array

DEFAULT_INPUT_PATH = BASE_DIR / "data" / "uploads" / "sample.wav"
DEFAULT_OUTPUT_PATH = BASE_DIR / "data" / "outputs" / "test_resampled.wav"

# =============================================================================
# RESAMPLING CONCEPTS IN PLAIN ENGLISH:
#
# 1. WHY RESAMPLING IS NECESSARY:
#    Digital audio consists of a fixed number of measurements (samples) taken
#    every second. If an ML model expects 16,000 samples for every 1 second of
#    sound, but we feed it audio recorded at 44,100 samples per second, the model
#    cannot guess the recording speed. It treats every 16,000 samples as 1 second,
#    causing the audio to be severely slowed down and pitch-shifted downward
#    (or sped up like a chipmunk if fed lower-rate audio).
#
# 2. DOWNSAMPLING vs. UPSAMPLING:
#    - Downsampling (e.g., 44.1 kHz -> 16 kHz):
#      Reduces the number of measurements per second. It discards frequencies
#      above 8 kHz (Nyquist frequency) that speech models do not need, reducing
#      computational load and memory footprint.
#    - Upsampling (e.g., 16 kHz -> 24 kHz or 44.1 kHz):
#      Increases the number of samples per second using mathematical interpolation.
#      It creates smooth curves between existing points, but DOES NOT add real
#      acoustic information that was not originally recorded.
#
# 3. WHY DURATION STAYS THE SAME:
#    Duration = Total Samples / Sample Rate.
#    Resampling changes HOW FINELY we measure the same span of time, not the
#    amount of time that passed. When you reduce the sample rate by half, you also
#    reduce the number of samples by half, keeping the total duration constant.
# =============================================================================


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resamples a 1D NumPy audio array from orig_sr to target_sr using librosa."""
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def run_resampling_pipeline(
    input_path: Union[Path, str] = DEFAULT_INPUT_PATH,
    output_path: Union[Path, str] = DEFAULT_OUTPUT_PATH,
    target_sr: int = TARGET_SAMPLE_RATE,
    play_after: bool = True,
) -> None:
    """Loads an audio file at its native sample rate, resamples it to target_sr,

    verifies duration invariance, saves to disk, and tests playback.
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        print(f"Error: Input audio file not found at {input_file}")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  PHASE 9: AUDIO RESAMPLING PIPELINE")
    print(f"  Input File:         {input_file.name}")
    print(f"  Target Sample Rate: {target_sr} Hz")
    print("=" * 65 + "\n")

    # Step 1: Load audio at ORIGINAL sample rate (sr=None ensures no implicit resampling)
    print(f"Loading '{input_file.name}' at its original sampling rate (sr=None)...")
    orig_audio, orig_sr = librosa.load(str(input_file), sr=None)
    orig_samples = len(orig_audio)
    orig_duration = orig_samples / orig_sr

    print(f"  [Original] Sample Rate:   {orig_sr:,} Hz")
    print(f"  [Original] Sample Count:  {orig_samples:,} samples")
    print(f"  [Original] Duration:      {orig_duration:.4f} seconds\n")

    # Step 2: Resample audio to TARGET_SAMPLE_RATE using librosa
    print(f"Resampling audio from {orig_sr} Hz -> {target_sr} Hz using librosa.resample...")
    resampled_audio = resample_audio(orig_audio, orig_sr, target_sr)
    new_samples = len(resampled_audio)
    new_duration = new_samples / target_sr

    print(f"  [Resampled] Sample Rate:  {target_sr:,} Hz")
    print(f"  [Resampled] Sample Count: {new_samples:,} samples")
    print(f"  [Resampled] Duration:     {new_duration:.4f} seconds\n")

    # Step 3: Verify proportional scaling and duration consistency
    expected_samples = round(orig_samples * (target_sr / orig_sr))
    duration_difference = abs(orig_duration - new_duration)

    print("--- Verification Check ---")
    print(f"  Rate Scaling Factor:      {target_sr / orig_sr:.4f}x")
    print(f"  Expected Sample Count:    {expected_samples:,}")
    print(f"  Actual Sample Count:      {new_samples:,} (Match: {expected_samples == new_samples})")
    print(f"  Duration Difference:      {duration_difference:.6f} seconds")

    # Duration sanity check (< 1 ms difference due to rounding)
    assert duration_difference < 0.001, "Error: Audio duration changed during resampling!"
    print("  Status:                   [PASSED] Duration perfectly preserved.\n")

    # Step 4: Save the resampled audio to disk using soundfile
    print(f"Saving resampled audio to: {output_file}")
    sf.write(str(output_file), resampled_audio, target_sr, subtype="PCM_16")
    print(f"  File saved successfully ({output_file.stat().st_size:,} bytes).\n")

    # Step 5: Play back using Phase 7 playback function
    if play_after:
        print("Playing back resampled audio to confirm quality and timing...")
        play_audio_array(resampled_audio, target_sr)


def demonstrate_44k_downsampling():
    """Generates an in-memory 44.1 kHz audio tone and downsamples it to 16 kHz

    to clearly show the proportional reduction from standard microphone audio.
    """
    print("\n" + "=" * 65)
    print("  BONUS DEMONSTRATION: Microphone Downsampling (44.1 kHz -> 16 kHz)")
    print("=" * 65)

    duration = 2.0
    mic_sr = 44100
    t = np.linspace(0, duration, int(mic_sr * duration), endpoint=False)
    mic_audio = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    print(f"  Synthetic 44.1 kHz Audio: {len(mic_audio):,} samples at {mic_sr} Hz ({duration:.2f}s)")

    downsampled = resample_audio(mic_audio, mic_sr, TARGET_SAMPLE_RATE)
    down_duration = len(downsampled) / TARGET_SAMPLE_RATE

    print(f"  Downsampled to 16 kHz:   {len(downsampled):,} samples at {TARGET_SAMPLE_RATE} Hz ({down_duration:.2f}s)")
    print(f"  Sample Reduction Ratio:  {len(downsampled) / len(mic_audio):.4f} (expected: {TARGET_SAMPLE_RATE / mic_sr:.4f})")
    print("  Notice: Exactly 88,200 samples shrank to 32,000 samples, preserving 2.00s!\n")


if __name__ == "__main__":
    custom_target = None
    # Support optional target sample rate from command line
    for arg in sys.argv[1:]:
        if arg.isdigit():
            custom_target = int(arg)

    target_rate = custom_target if custom_target else TARGET_SAMPLE_RATE
    run_resampling_pipeline(target_sr=target_rate)

    # If the input was already 16 kHz, run the 44.1 kHz -> 16 kHz downsampling demonstration
    demonstrate_44k_downsampling()
