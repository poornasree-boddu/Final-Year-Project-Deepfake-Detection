# Multimodal Deepfake Detection System

A Python-based multimodal deepfake detection project that combines image, video, and audio models with a fusion layer, plus a Streamlit interface for demos.

## What Is Included In This Repository
- Source code for image, video, audio, and fusion modules
- Streamlit app (`app.py`)
- Configuration (`config.py`)
- Dependency list (`requirements.txt`)

## What Is Excluded From GitHub
To keep the repository lightweight, heavy data and model binaries are ignored:
- `datasets/`
- `demo_media/`
- model weight files in `models/` such as `*.pth`, `*.pt`, `*.ckpt`
- local virtual environment (`multimodal_env/`)

## Setup
1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your datasets and trained model files locally:
- Place datasets under `datasets/`
- Place model weights under `models/` (for example: `audio_model.pth`, `image_model.pth`, `video_model.pth`)

## Run The App

```bash
streamlit run app.py
```

## Notes
- This repository is configured to avoid uploading large datasets and binaries.
- If you need to share model files, use cloud storage and document download links here.
