# ============================================================
# modules/training.py — Training Module
# Orchestrates the full GAN adversarial training loop.
# Saves sample images and model checkpoints periodically.
# ============================================================

import os
import csv
import time
import numpy as np
import tensorflow as tf

from config import (
    EPOCHS, LATENT_DIM, SAMPLE_INTERVAL, CHECKPOINT_INTERVAL,
    MODEL_DIR, IMAGE_DIR, LOG_DIR,
)
from modules.gan import GAN
from utils.visualization import save_sample_grid


class Trainer:
    """
    Handles the complete GAN training lifecycle.

    Parameters
    ----------
    gan          : GAN instance
    dataset      : tf.data.Dataset (batched, preprocessed images)
    epochs       : int  (default from config)
    """

    def __init__(
        self,
        gan: GAN,
        dataset: tf.data.Dataset,
        epochs: int = EPOCHS,
        latent_dim: int = LATENT_DIM,
    ):
        self.gan = gan
        self.dataset = dataset
        self.epochs = epochs
        self.latent_dim = latent_dim

        # Fixed noise for reproducible sample images across epochs
        self.fixed_noise = tf.random.normal([16, latent_dim])

        # CSV log file
        self.log_path = os.path.join(LOG_DIR, "training_log.csv")
        self._init_log()

    # ── Public API ────────────────────────────────────────────────────────────

    def train(self):
        """Run the full training loop for ``self.epochs`` epochs."""
        print(f"\n[Trainer] Starting training  epochs={self.epochs}")
        print("=" * 60)

        for epoch in range(1, self.epochs + 1):
            start = time.time()

            epoch_g_losses = []
            epoch_d_losses = []

            for batch in self.dataset:
                losses = self.gan.train_step(batch)
                epoch_g_losses.append(float(losses["g_loss"]))
                epoch_d_losses.append(float(losses["d_loss"]))

            mean_g = np.mean(epoch_g_losses)
            mean_d = np.mean(epoch_d_losses)
            elapsed = time.time() - start

            self._log_epoch(epoch, mean_g, mean_d, elapsed)

            if epoch % SAMPLE_INTERVAL == 0 or epoch == 1:
                self._save_samples(epoch)

            if epoch % CHECKPOINT_INTERVAL == 0:
                self._save_checkpoint(epoch)

            if epoch % 100 == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:>6}/{self.epochs}  "
                    f"G_loss={mean_g:.4f}  D_loss={mean_d:.4f}  "
                    f"time={elapsed:.1f}s"
                )

        print("[Trainer] Training complete.")
        self._save_checkpoint(self.epochs, label="final")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _save_samples(self, epoch: int):
        fake_imgs = self.gan.generator(self.fixed_noise, training=False).numpy()
        path = os.path.join(IMAGE_DIR, f"epoch_{epoch:06d}.png")
        save_sample_grid(fake_imgs, path, title=f"Epoch {epoch}")
        print(f"  [Trainer] Sample saved → {path}")

    def _save_checkpoint(self, epoch: int, label: str = ""):
        tag = label or f"epoch_{epoch:06d}"
        gen_path  = os.path.join(MODEL_DIR, f"generator_{tag}.weights.h5")
        disc_path = os.path.join(MODEL_DIR, f"discriminator_{tag}.weights.h5")
        self.gan.save_weights(gen_path, disc_path)

    def _init_log(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "g_loss", "d_loss", "time_sec"])

    def _log_epoch(self, epoch: int, g_loss: float, d_loss: float, elapsed: float):
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{g_loss:.6f}", f"{d_loss:.6f}", f"{elapsed:.2f}"])
