# ============================================================
# modules/preprocessing.py — Preprocessing Module
# Resize, normalise, denoise, and enhance contrast of X-ray images.
# Produces TensorFlow Dataset objects for efficient training.
# ============================================================

import numpy as np
import tensorflow as tf

from config import IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, BATCH_SIZE


class Preprocessor:
    """
    Image preprocessing pipeline for chest X-ray images.

    Steps
    -----
    1. Resize  → (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    2. Normalise pixel values to [-1, 1]  (required for tanh GAN output)
    3. Optional Gaussian denoise
    4. Optional CLAHE contrast enhancement
    """

    def __init__(
        self,
        img_height: int = IMG_HEIGHT,
        img_width: int = IMG_WIDTH,
        img_channels: int = IMG_CHANNELS,
        apply_denoise: bool = True,
        apply_clahe: bool = True,
    ):
        self.h = img_height
        self.w = img_width
        self.c = img_channels
        self.apply_denoise = apply_denoise
        self.apply_clahe = apply_clahe

    # ── Public API ────────────────────────────────────────────────────────────

    def preprocess(self, raw_images: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline to a NumPy image array.

        Parameters
        ----------
        raw_images : np.ndarray  shape (N, H, W, C), dtype float32, range [0,255]

        Returns
        -------
        np.ndarray  shape (N, H, W, C), dtype float32, range [-1, 1]
        """
        processed = []
        for img in raw_images:
            img = self._resize(img)
            if self.apply_clahe:
                img = self._clahe(img)
            if self.apply_denoise:
                img = self._gaussian_denoise(img)
            img = self._normalise(img)
            processed.append(img)

        result = np.stack(processed, axis=0)
        print(f"[Preprocessor] Preprocessing done  shape={result.shape}  range=[{result.min():.2f}, {result.max():.2f}]")
        return result

    def build_tf_dataset(self, images: np.ndarray, batch_size: int = BATCH_SIZE) -> tf.data.Dataset:
        """
        Wrap a preprocessed NumPy array in a tf.data.Dataset.

        Returns a shuffled, batched, prefetched dataset.
        """
        dataset = (
            tf.data.Dataset.from_tensor_slices(images)
            .shuffle(buffer_size=len(images))
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )
        print(f"[Preprocessor] tf.data.Dataset ready  batch_size={batch_size}")
        return dataset

    def train_val_split(
        self, images: np.ndarray, val_ratio: float = 0.15
    ):
        """Split (N, …) array into train and validation subsets."""
        n = len(images)
        n_val = int(n * val_ratio)
        indices = np.random.permutation(n)
        val_idx, train_idx = indices[:n_val], indices[n_val:]
        return images[train_idx], images[val_idx]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _resize(self, img: np.ndarray) -> np.ndarray:
        """Resize a single image array to (H, W, C)."""
        import cv2
        h, w = img.shape[:2]
        if (h, w) != (self.h, self.w):
            img = cv2.resize(
                img.squeeze(),
                (self.w, self.h),
                interpolation=cv2.INTER_AREA,
            )
            img = img[..., np.newaxis]
        return img.astype(np.float32)

    @staticmethod
    def _normalise(img: np.ndarray) -> np.ndarray:
        """Map pixel values from [0, 255] → [-1, 1]."""
        return (img / 127.5) - 1.0

    @staticmethod
    def _clahe(img: np.ndarray) -> np.ndarray:
        """Apply CLAHE contrast-limited adaptive histogram equalisation."""
        import cv2
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        uint8 = img.squeeze().clip(0, 255).astype(np.uint8)
        enhanced = clahe.apply(uint8).astype(np.float32)
        return enhanced[..., np.newaxis]

    @staticmethod
    def _gaussian_denoise(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Apply Gaussian blur for noise reduction."""
        import cv2
        denoised = cv2.GaussianBlur(img.squeeze(), (kernel_size, kernel_size), 0)
        return denoised.astype(np.float32)[..., np.newaxis]
