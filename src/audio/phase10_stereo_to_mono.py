import sys
from pathlib import Path
from typing import Union
import librosa
import numpy as np
import soundfile as sf

# Resolve root directory and import Phase 7 playback function
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.audio.phase7_play_audio import play_audio_array

DEFAULT_STEREO_PATH = BASE_DIR / "data" / "uploads" / "sample_stereo.wav"
DEFAULT_MONO_INPUT_PATH = BASE_DIR / "data" / "uploads" / "sample.wav"
DEFAULT_MONO_OUTPUT_PATH = BASE_DIR / "data" / "outputs" / "test_mono.wav"

# =============================================================================
# STEREO vs. MONO CONCEPTS IN PLAIN ENGLISH:
#
# 1. WHY MONO IS REQUIRED FOR VOICE CLONING & TTS:
#    - Stereo audio contains two separate audio channels (Left & Right) recorded
#      to create spatial width and 3D positioning for human ears.
#    - Speech recognition, speaker embedding models (e.g., WavLM, Resemblyzer),
#      and TTS acoustic models only care about the acoustic characteristics of
#      the speaker's voice (pitch, formant frequencies, phonemes), NOT whether
#      the voice originated slightly from the left or right ear.
#    - Feeding stereo channels into a speech model doubles the memory footprint
#      and confuses models trained on single-channel speech.
#
# 2. THE MATHEMATICS OF STEREO-TO-MONO CONVERSION:
#    - Converting stereo to mono is achieved by taking the arithmetic mean
#      (average) of the left and right channel samples at every single time step:
#           Mono_Sample[t] = (Left_Sample[t] + Right_Sample[t]) / 2.0
#    - Averaging preserves all acoustic content from both channels equally,
#      blending them into a single coherent channel while preventing clipping.
# =============================================================================


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Converts any multi-channel audio NumPy array into a single-channel 1D mono array.

    This function is reusable across the PolyVox audio ingestion & validation pipeline.
    It automatically handles:
      1. Already-mono audio: 1D array of shape (N,) -> returned unchanged.
      2. Soundfile stereo format: 2D array of shape (N, channels) -> averaged along axis 1.
      3. Librosa stereo format: 2D array of shape (channels, N) -> averaged along axis 0.

    Args:
        audio (np.ndarray): Input audio array.

    Returns:
        np.ndarray: 1D float32 mono audio array of shape (N,).
    """
    # Case 1: Audio is already single-channel mono (1D)
    if audio.ndim == 1:
        return audio

    # Case 2: Audio is multi-channel (2D)
    if audio.ndim == 2:
        # Librosa channel convention: (channels, num_samples) where channels <= 8
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            return np.mean(audio, axis=0, dtype=np.float32)

        # Soundfile channel convention: (num_samples, channels)
        return np.mean(audio, axis=1, dtype=np.float32)

    raise ValueError(f"Expected 1D or 2D audio array, but received array with shape: {audio.shape}")


def ensure_test_stereo_file(file_path: Path = DEFAULT_STEREO_PATH) -> Path:
    """Generates a 2-second dual-frequency stereo WAV file if it does not already exist.

    Left Channel = 440 Hz (Concert A note)
    Right Channel = 660 Hz (Concert E note)
    """
    if file_path.exists():
        return file_path

    file_path.parent.mkdir(parents=True, exist_ok=True)
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Distinct left and right channel signals
    left_channel = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    right_channel = (0.5 * np.sin(2 * np.pi * 660.0 * t)).astype(np.float32)

    # Stack into 2D stereo array: shape (num_samples, 2)
    stereo_data = np.stack([left_channel, right_channel], axis=1)

    sf.write(str(file_path), stereo_data, sr, subtype="PCM_16")
    print(f"Generated synthetic stereo test file: {file_path.name}")
    print(f"  Left channel: 440 Hz | Right channel: 660 Hz | Rate: {sr} Hz")
    return file_path


def run_stereo_to_mono_pipeline(
    stereo_path: Union[Path, str] = DEFAULT_STEREO_PATH,
    output_path: Union[Path, str] = DEFAULT_MONO_OUTPUT_PATH,
    play_after: bool = True,
) -> None:
    """Demonstrates loading a stereo file, converting it to mono using to_mono(),

    saving the result, and validating playback.
    """
    stereo_file = Path(stereo_path)
    output_file = Path(output_path)

    ensure_test_stereo_file(stereo_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  PHASE 10: STEREO TO MONO CONVERSION")
    print(f"  Input File:  {stereo_file.name}")
    print(f"  Output File: {output_file.name}")
    print("=" * 65 + "\n")

    # Inspect file metadata using soundfile.info to see physical channel count
    file_info = sf.info(str(stereo_file))
    print(f"[File Inspection]")
    print(f"  On-Disk Channels: {file_info.channels} ({'Stereo' if file_info.channels == 2 else 'Mono'})")
    print(f"  Sample Rate:      {file_info.samplerate:,} Hz")
    print(f"  Total Frames:     {file_info.frames:,}\n")

    # Load stereo audio with librosa (mono=False is essential to prevent automatic conversion)
    print(f"Loading '{stereo_file.name}' with librosa (mono=False, sr=None)...")
    stereo_audio, sr = librosa.load(str(stereo_file), sr=None, mono=False)

    num_channels = stereo_audio.shape[0] if stereo_audio.ndim == 2 else 1
    print(f"  Raw Array Shape:   {stereo_audio.shape}")
    print(f"  Detected Channels: {num_channels} (channels-first format [channels, samples])\n")

    # Convert to mono using our reusable function
    print("Converting to single-channel mono via channel averaging: (Left + Right) / 2.0...")
    mono_audio = to_mono(stereo_audio)

    print(f"  Mono Array Shape:  {mono_audio.shape} (1D array [samples])")
    print(f"  Mono Data Type:    {mono_audio.dtype}")
    print(f"  Total Samples:     {len(mono_audio):,}\n")

    # Save converted mono audio to disk
    print(f"Saving mono audio to: {output_file}")
    sf.write(str(output_file), mono_audio, sr, subtype="PCM_16")

    saved_info = sf.info(str(output_file))
    print(f"  Saved File Verified: {saved_info.channels} channel(s), {saved_info.samplerate} Hz\n")

    # Play back mono version to confirm acoustic quality
    if play_after:
        print("Playing back the converted mono audio through speakers...")
        play_audio_array(mono_audio, sr)


def test_already_mono_handling(mono_path: Union[Path, str] = DEFAULT_MONO_INPUT_PATH) -> None:
    """Verifies that to_mono() safely handles an already-mono audio file without error."""
    mono_file = Path(mono_path)
    if not mono_file.exists():
        print(f"Skipping already-mono test (file not found: {mono_file})")
        return

    print("\n" + "=" * 65)
    print("  EDGE CASE TEST: Input is Already Mono")
    print("=" * 65)

    audio_mono, sr = librosa.load(str(mono_file), sr=None, mono=False)
    print(f"  Input File:        {mono_file.name}")
    print(f"  Input Shape:       {audio_mono.shape} (1D array, already mono)")

    # Run to_mono
    result = to_mono(audio_mono)
    print(f"  Output Shape:      {result.shape}")
    print(f"  Identical Array:   {np.array_equal(audio_mono, result)}")
    print("  Status:            [PASSED] Mono input passed through untouched without error.\n")


if __name__ == "__main__":
    run_stereo_to_mono_pipeline()
    test_already_mono_handling()
