# Phase 15: Coqui XTTS-v2 API Reference & Model Cheat-Sheet

This reference document details the exact input, output, and configuration specifications for **Coqui XTTS-v2**. Use this cheat-sheet during Phase 16 (Voice Generation) to ensure precise parameter types and avoid runtime errors.

---

## 1. Model Initialization & Hardware Placement

### Primary Class: `TTS.api.TTS`
The high-level wrapper provided by the `coqui-tts` library handles model discovery, config parsing, checkpoint loading, and device management.

```python
import os
# Ensure non-commercial license agreement is acknowledged automatically
os.environ["COQUI_TOS_AGREED"] = "1"

from TTS.api import TTS

# 1. Instantiate the model (downloads to cache if not already present)
tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    progress_bar=True,
    gpu=True  # Automatically moves layers to torch.device("cuda")
)

# Alternatively, explicit device assignment:
# tts.to("cuda")   # For NVIDIA RTX 3050 (recommended)
# tts.to("cpu")    # Fallback mode
```

---

## 2. Generation Methods & Return Types

XTTS-v2 offers two generation methods depending on whether you want an **in-memory audio array** or a **direct file on disk**.

### Method A: In-Memory Synthesis — `tts.tts(...)`
Use this when you want to inspect, transform, or play the audio directly in Python using `sounddevice` (from Phase 7) without intermediate disk I/O.

```python
audio_samples = tts.tts(
    text="Hello from PolyVox!",
    speaker_wav="data/uploads/reference_speaker.wav",
    language="en",
    speed=1.0,
    split_sentences=True
)
```
* **Return Type:** `list[float]` (standard Python list of floating-point PCM values between `-1.0` and `+1.0`).
* **Conversion to NumPy:**
  ```python
  import numpy as np
  audio_array = np.array(audio_samples, dtype=np.float32)
  ```
* **Output Sample Rate:** **24,000 Hz (24 kHz)**.

---

### Method B: Direct File Synthesis — `tts.tts_to_file(...)`
Use this when you want Coqui to automatically encode and save the output audio to a WAV file on disk.

```python
output_file_path = tts.tts_to_file(
    text="Hello from PolyVox!",
    speaker_wav="data/uploads/reference_speaker.wav",
    language="en",
    file_path="data/outputs/cloned_output.wav",
    speed=1.0,
    split_sentences=True
)
```
* **Return Type:** `str` (the absolute or relative path to the saved file).
* **On-Disk File Format:** 16-bit PCM WAV at **24,000 Hz (Mono)**.

---

## 3. Parameter Specifications

| Parameter | Type | Required? | Description & Allowed Values |
| :--- | :--- | :--- | :--- |
| `text` | `str` | **Yes** | The sentence or paragraph to be synthesized. Must contain standard punctuation (`.`, `,`, `!`, `?`) to guide natural rhythm. |
| `speaker_wav` | `str` \| `Path` \| `list` | **Yes** | Path (or list of paths) to the reference voice audio clip to clone. |
| `language` | `str` | **Yes** | Target language ISO code. Must be one of the **17 supported codes** (see table below). |
| `file_path` | `str` \| `Path` | Only for `tts_to_file` | Destination path for the generated WAV file (e.g., `data/outputs/generated.wav`). |
| `speed` | `float` | No (Default `1.0`) | Playback speed multiplier. Values: `0.5` (slow) to `2.0` (fast). |
| `split_sentences`| `bool` | No (Default `True`) | Automatically splits long text into sentences for more natural pauses and lower VRAM spikes. |
| `temperature` | `float` | No (Default `0.75`)| Sampling randomness. Lower (`0.5`) = more stable/consistent; Higher (`0.85`) = more expressive/variable. |
| `repetition_penalty`| `float`| No (Default `5.0`)| Penalizes the autoregressive model from repeating words or getting stuck in vocal loops. |
| `top_k` | `int` | No (Default `50`) | Top-K candidate token filtering during autoregressive decoding. |
| `top_p` | `float` | No (Default `0.85`)| Nucleus sampling probability threshold. |

### Supported Language Codes (17 Languages)
Language codes must be passed as exact lowercase strings:

| Code | Language | Code | Language | Code | Language |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `"en"` | English | `"es"` | Spanish | `"fr"` | French |
| `"de"` | German | `"it"` | Italian | `"pt"` | Portuguese |
| `"pl"` | Polish | `"tr"` | Turkish | `"ru"` | Russian |
| `"nl"` | Dutch | `"cs"` | Czech | `"ar"` | Arabic |
| `"zh-cn"`| Chinese (Simplified)| `"ja"` | Japanese | `"hu"` | Hungarian |
| `"ko"` | Korean | `"hi"` | Hindi | | |

---

## 4. Reference Audio Requirements (Connecting to Phases 9 & 10)

XTTS-v2's zero-shot conditioning extracts a speaker latent vector directly from the `speaker_wav` file. For optimal cloning results:

1. **Duration:**
   * **Minimum:** 3 seconds.
   * **Recommended Sweet Spot:** **10 to 20 seconds** of continuous, clean speech.
   * **Maximum:** ~30 seconds. Audio longer than 30 seconds does not improve quality and increases VRAM allocation during embedding extraction.
2. **Channel Format:**
   * **Single-Channel Mono:** While Coqui has internal fallbacks, passing raw stereo audio can degrade speaker embedding accuracy. **Always preprocess uploaded reference audio through our Phase 10 `to_mono()` function.**
3. **Acoustic Cleanliness:**
   * Reference audio must contain clean voice only. Background music, heavy room reverb, air conditioning hum, or multiple overlapping speakers will bleed into the cloned voice timbre.
4. **Input Sample Rate vs. Internal Resampling:**
   * XTTS-v2's speaker encoder internally processes conditioning audio at **22,050 Hz**.
   * It will automatically resample inputs on the fly, but passing clean 16 kHz or 22.05 kHz audio avoids aliasing artifacts.

---

## 5. Sample Rate Alignment Notice: `TARGET_SAMPLE_RATE`

> [!IMPORTANT]
> **Sample Rate Flag:**
> * In **Phase 9**, we initialized `TARGET_SAMPLE_RATE = 16000` in [config.py](file:///c:/Users/dhanu/Documents/PROJECTS/PolyVox/config.py) (which is standard for Whisper ASR, WavLM, and general speech ML).
> * **Coqui XTTS-v2's HiFi-GAN vocoder natively generates audio at 24,000 Hz (24 kHz).**
> * If we attempt to play back or save XTTS-v2's output array at `16000 Hz`, the audio will sound slowed down by 33% (pitched downward).
> * **Recommended Architecture:**
>   * `INPUT_SAMPLE_RATE = 16000` (for incoming mic/reference audio preprocessing & future Whisper transcription).
>   * `MODEL_SAMPLE_RATE = 24000` (for XTTS-v2 synthesized output).

---

## 6. Ready for Phase 16 Checklist

Before running our first voice generation test in Phase 16, confirm the following:
- [x] Model package `coqui-tts` installed and verified ([phase14_verify_install.py](file:///c:/Users/dhanu/Documents/PROJECTS/PolyVox/phase14_verify_install.py)).
- [x] Model weights cache directory confirmed (`%LOCALAPPDATA%\tts\tts_models--multilingual--multi-dataset--xtts_v2\`).
- [x] Target text string prepared with proper punctuation.
- [x] Target language code selected from the 17 supported ISO strings (e.g. `"en"`).
- [x] Reference speaker WAV selected (minimum 6s, mono, clean vocal).
- [x] Output playback pipeline understands the **24,000 Hz** sample rate.
