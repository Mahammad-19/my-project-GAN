# ============================================================
# modules/output_manager.py — Output Generation Module
# Saves high-quality synthetic chest X-ray images to disk.
# Integrates with the GAN model for batch image production.
# ============================================================

import os
import numpy as np
import cv2
from datetime import datetime

from config import IMAGE_DIR, LATENT_DIM


class OutputManager:
    """
    Manages the saving and organisation of synthetic chest X-ray images.

    Images are stored under ``IMAGE_DIR/<session>/``.
    Each call to ``generate_and_save`` creates a timestamped session.
    """

    def __init__(self, output_dir: str = IMAGE_DIR, latent_dim: int = LATENT_DIM):
        self.output_dir = output_dir
        self.latent_dim = latent_dim
        os.makedirs(self.output_dir, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_and_save(self, generator, n_images: int = 10) -> list:
        """
        Generate ``n_images`` synthetic images and save them as PNG files.

        Parameters
        ----------
        generator  : Keras Model (the trained GAN Generator)
        n_images   : int  number of images to generate

        Returns
        -------
        list[str]  paths to saved images
        """
        import tensorflow as tf

        session = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(self.output_dir, f"generated_{session}")
        os.makedirs(session_dir, exist_ok=True)

        noise = tf.random.normal([n_images, self.latent_dim])
        fake_imgs = generator(noise, training=False).numpy()  # (N, H, W, C)

        saved_paths = []
        for i, img in enumerate(fake_imgs):
            path = os.path.join(session_dir, f"synthetic_{i:04d}.png")
            self._save_image(img, path)
            saved_paths.append(path)

        print(f"[OutputManager] {n_images} synthetic images saved → {session_dir}")
        return saved_paths

    def save_numpy_array(self, images: np.ndarray, prefix: str = "img") -> list:
        """
        Save a NumPy array of images (N, H, W, C) in range [-1, 1] as PNGs.
        """
        session_dir = os.path.join(
            self.output_dir, f"saved_{datetime.now().strftime('%H%M%S')}"
        )
        os.makedirs(session_dir, exist_ok=True)

        paths = []
        for i, img in enumerate(images):
            path = os.path.join(session_dir, f"{prefix}_{i:04d}.png")
            self._save_image(img, path)
            paths.append(path)

        return paths

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _save_image(img: np.ndarray, path: str):
        """
        Convert a single image in [-1, 1] to uint8 and write to disk.
        """
        # Denormalise: [-1, 1] → [0, 255]
        img_uint8 = ((img.squeeze() + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        cv2.imwrite(path, img_uint8)

    def list_saved(self) -> list:
        """Return all PNG files under the output directory."""
        paths = []
        for root, _, files in os.walk(self.output_dir):
            for f in files:
                if f.endswith(".png"):
                    paths.append(os.path.join(root, f))
        return sorted(paths)
