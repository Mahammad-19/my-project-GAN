# ============================================================
# config.py — Global configuration for GAN-based Medical Image Synthesis
# Project: Medical Image Synthesis Using GANs for Pulmonary Chest X-rays
# ============================================================

import os

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data", "sample")
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")
MODEL_DIR       = os.path.join(OUTPUT_DIR, "models")
IMAGE_DIR       = os.path.join(OUTPUT_DIR, "images")
LOG_DIR         = os.path.join(OUTPUT_DIR, "logs")

# ── Image Settings ────────────────────────────────────────────────────────────
IMG_HEIGHT      = 64          # Increase to 128 or 256 for higher resolution
IMG_WIDTH       = 64
IMG_CHANNELS    = 1           # Grayscale chest X-rays
IMG_SHAPE       = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# ── GAN Hyperparameters ───────────────────────────────────────────────────────
LATENT_DIM      = 100         # Size of the noise vector fed to the Generator
EPOCHS          = 10000       # Total training epochs
BATCH_SIZE      = 32
LEARNING_RATE   = 0.0002
BETA_1          = 0.5         # Adam β₁ (recommended for GAN training)

# ── Training Control ──────────────────────────────────────────────────────────
SAMPLE_INTERVAL = 500         # Save sample images every N epochs
CHECKPOINT_INTERVAL = 1000   # Save model weights every N epochs

# ── Evaluation ────────────────────────────────────────────────────────────────
N_EVAL_IMAGES   = 10          # Number of images used for metric evaluation

# ── Supported Formats ─────────────────────────────────────────────────────────
SUPPORTED_FORMATS = [".png", ".jpg", ".jpeg", ".dcm"]

# ── Create required directories ───────────────────────────────────────────────
for _dir in [DATA_DIR, OUTPUT_DIR, MODEL_DIR, IMAGE_DIR, LOG_DIR]:
    os.makedirs(_dir, exist_ok=True)
