import os

# ==============================
# PROJECT ROOT
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================
# DATASET PATHS
# ==============================

VIDEO_DATASET_PATH = os.path.join(BASE_DIR, "datasets", "video_dataset")

IMAGE_DATASET_PATH = os.path.join(BASE_DIR, "datasets", "image_dataset")

AUDIO_DATASET_PATH = os.path.join(BASE_DIR, "datasets", "audio_dataset")

# ==============================
# MODEL PATHS
# ==============================

VIDEO_MODEL_PATH = os.path.join(BASE_DIR, "models", "video_model.pth")

IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "models", "image_model.pth")

AUDIO_MODEL_PATH = os.path.join(BASE_DIR, "models", "audio_model.pth")

# ==============================
# DEMO MEDIA PATHS
# ==============================

DEMO_VIDEO_PATH = os.path.join(BASE_DIR, "demo_media", "videos")

DEMO_IMAGE_PATH = os.path.join(BASE_DIR, "demo_media", "images")

DEMO_AUDIO_PATH = os.path.join(BASE_DIR, "demo_media", "audio")

# ==============================
# VIDEO PROCESSING SETTINGS
# ==============================

FRAME_COUNT = 40
FRAME_SIZE = 224

# ==============================
# IMAGE PROCESSING SETTINGS
# ==============================

IMAGE_SIZE = 224
IMAGE_BATCH_SIZE = 32

# ==============================
# MODEL PARAMETERS
# ==============================

FEATURE_DIM = 1024
LSTM_HIDDEN_SIZE = 256

# ==============================
# FUSION SETTINGS
# ==============================

FUSION_THRESHOLD = 0.5
