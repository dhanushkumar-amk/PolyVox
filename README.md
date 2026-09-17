# PolyVox

Self-hosted, multilingual voice cloning application in Python.

---

## Project Structure

The codebase is organized into clear architectural layers—cleanly separating **core logic** (audio processing & ML models) from **interfaces** (FastAPI backend & Streamlit UI), **runtime data**, and **deployment assets**:

```text
PolyVox/
├── config.py              # Central configuration loader and environment constants
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
├── .env                   # Local environment variables and secrets (gitignored)
├── .gitignore             # Git ignore patterns
│
├── data/                  # Runtime data storage (contents gitignored except .gitkeep)
│   ├── embeddings/        # Cached speaker embeddings extracted from reference audio
│   ├── outputs/           # Synthesized and cloned audio files produced by the model
│   └── uploads/           # Reference voice audio samples uploaded by users
│
├── docker/                # Dockerfiles and container deployment configurations
│
├── src/                   # Application source code
│   ├── __init__.py        # Makes src a top-level Python package
│   │
│   ├── api/               # FastAPI backend routes, schemas, and API application logic
│   ├── audio/             # Audio preprocessing utilities (VAD, resampling, quality checks)
│   ├── evaluation/        # Evaluation pipelines (Whisper transcription & speaker similarity)
│   ├── frontend/          # Streamlit user interface application
│   └── tts/               # TTS and voice cloning model wrappers and inference engine
│
└── tests/                 # Automated unit, integration, and pipeline test suites
```

### Directory Purposes

- **`data/`**: Root directory for dynamic runtime data, separating user files and cache from source code.
  - **`data/embeddings/`**: Stores precomputed and cached speaker embeddings to avoid redundant feature extraction.
  - **`data/outputs/`**: Stores generated audio files produced by the voice cloning and synthesis pipelines (gitignored).
  - **`data/uploads/`**: Stores raw audio samples provided by users as reference voices for cloning (gitignored).
- **`docker/`**: Reserved for Dockerfiles, Compose files, and container setup scripts for Tier 3 deployment.
- **`src/`**: Houses all core application code, structured to avoid tight coupling between components.
  - **`src/api/`**: Implements the FastAPI backend endpoints, request validation models, and API orchestration.
  - **`src/audio/`**: Provides audio preprocessing tools such as Voice Activity Detection (VAD), audio normalization, and resampling.
  - **`src/evaluation/`**: Houses evaluation scripts for Whisper-based speech recognition accuracy and voice similarity scoring.
  - **`src/frontend/`**: Contains the standalone Streamlit web application providing the user interface.
  - **`src/tts/`**: Contains the TTS model wrapper, voice cloning inference logic, and model management routines.
- **`tests/`**: Contains test modules for unit testing audio filters, model wrappers, API endpoints, and end-to-end flows.
