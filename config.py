import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file
load_dotenv(BASE_DIR / ".env")

# Configuration constants
PROJECT_NAME = os.getenv("PROJECT_NAME", "PolyVox")

# Target sampling rate for audio processing and TTS model pipelines.
# Defaulted to 16,000 Hz (standard for speech recognition, embeddings, and models like Whisper / WavLM).
# Note: This may need to be adjusted (e.g., 22,050 Hz or 24,000 Hz) depending on which TTS/vocoder model we select later.
TARGET_SAMPLE_RATE = int(os.getenv("TARGET_SAMPLE_RATE", 16000))
