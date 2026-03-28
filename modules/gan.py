# ============================================================
# modules/gan.py — GAN Integration Module
# Combines Generator and Discriminator into a unified adversarial model.
# Manages optimisers and the @tf.function training steps.
# ============================================================

import tensorflow as tf

from config import LATENT_DIM, LEARNING_RATE, BETA_1
from modules.generator import build_generator, generator_loss
from modules.discriminator import build_discriminator, discriminator_loss


class GAN:
    """
    Unified GAN model.

    Attributes
    ----------
    generator       : Keras Model
    discriminator   : Keras Model
    g_optimizer     : Adam optimiser for the generator
    d_optimizer     : Adam optimiser for the discriminator
    latent_dim      : int
    """

    def __init__(
        self,
        img_shape: tuple,
        latent_dim: int = LATENT_DIM,
        lr: float = LEARNING_RATE,
        beta_1: float = BETA_1,
    ):
        self.latent_dim = latent_dim
        self.img_shape = img_shape

        # Build sub-networks
        self.generator = build_generator(latent_dim, img_shape)
        self.discriminator = build_discriminator(img_shape)

        # Separate optimisers are critical — they must NOT share state
        self.g_optimizer = tf.keras.optimizers.Adam(lr, beta_1=beta_1)
        self.d_optimizer = tf.keras.optimizers.Adam(lr, beta_1=beta_1)

        # Track losses
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")

        self.generator.summary()
        self.discriminator.summary()

    # ── Training step ─────────────────────────────────────────────────────────

    @tf.function
    def train_step(self, real_images: tf.Tensor):
        """
        Single alternating GAN training step.

        1. Sample noise → generate fake images
        2. Update Discriminator (maximise ability to tell real from fake)
        3. Update Generator     (maximise discriminator's mistake on fakes)
        """
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, self.latent_dim])

        # ── Discriminator update ──────────────────────────────────────────────
        with tf.GradientTape() as disc_tape:
            fake_images = self.generator(noise, training=True)

            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)

            d_loss = discriminator_loss(real_output, fake_output)

        d_grads = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(d_grads, self.discriminator.trainable_variables)
        )

        # ── Generator update ──────────────────────────────────────────────────
        noise = tf.random.normal([batch_size, self.latent_dim])
        with tf.GradientTape() as gen_tape:
            fake_images = self.generator(noise, training=True)
            fake_output = self.discriminator(fake_images, training=False)
            g_loss = generator_loss(fake_output)

        g_grads = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_grads, self.generator.trainable_variables)
        )

        # Update running metrics
        self.g_loss_metric.update_state(g_loss)
        self.d_loss_metric.update_state(d_loss)

        return {"g_loss": g_loss, "d_loss": d_loss}

    # ── Utility helpers ───────────────────────────────────────────────────────

    def generate_images(self, n: int) -> tf.Tensor:
        """Generate ``n`` synthetic images from random noise."""
        noise = tf.random.normal([n, self.latent_dim])
        return self.generator(noise, training=False)

    def save_weights(self, gen_path: str, disc_path: str):
        self.generator.save_weights(gen_path)
        self.discriminator.save_weights(disc_path)
        print(f"[GAN] Weights saved → {gen_path}, {disc_path}")

    def load_weights(self, gen_path: str, disc_path: str):
        self.generator.load_weights(gen_path)
        self.discriminator.load_weights(disc_path)
        print(f"[GAN] Weights loaded ← {gen_path}, {disc_path}")
