# ============================================================
# modules/discriminator.py — Discriminator Network Module
# Classifies an image as REAL or SYNTHETIC.
# Architecture: Conv2D blocks with LeakyReLU + Dropout → Dense sigmoid
# ============================================================

import tensorflow as tf
from tensorflow.keras import layers, Model

from config import IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS


def build_discriminator(
    img_shape: tuple = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
) -> Model:
    """
    Build and return the Discriminator model.

    Architecture
    ------------
    Input  : image  (H, W, C)
    Output : scalar logit — positive → real, negative → fake

    Uses spectral-normalization-style design (LeakyReLU, no BN in first
    layer, Dropout for regularisation) as recommended for stable GAN training.
    """

    img_input = layers.Input(shape=img_shape, name="img_input")

    # ── Conv block 1 — no BN on first layer ───────────────────────────────────
    x = layers.Conv2D(32, kernel_size=4, strides=2, padding="same")(img_input)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)                             # (32, 32, 32)

    # ── Conv block 2 ─────────────────────────────────────────────────────────
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)                             # (16, 16, 64)

    # ── Conv block 3 ─────────────────────────────────────────────────────────
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)                             # (8, 8, 128)

    # ── Conv block 4 ─────────────────────────────────────────────────────────
    x = layers.Conv2D(256, kernel_size=4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)                             # (4, 4, 256)

    # ── Classification head ───────────────────────────────────────────────────
    x = layers.Flatten()(x)
    validity = layers.Dense(1, name="validity")(x)         # raw logit

    model = Model(img_input, validity, name="Discriminator")
    return model


def discriminator_loss(
    real_output: tf.Tensor, fake_output: tf.Tensor
) -> tf.Tensor:
    """
    Standard GAN discriminator loss.

    D is penalised for incorrectly labelling real images as fake, and
    fake images as real.  Uses binary cross-entropy with from_logits=True.
    """
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_labels = tf.ones_like(real_output)
    fake_labels = tf.zeros_like(fake_output)

    real_loss = cross_entropy(real_labels, real_output)
    fake_loss = cross_entropy(fake_labels, fake_output)
    return real_loss + fake_loss
