# ============================================================
# modules/generator.py — Generator Network Module
# Transforms a latent noise vector z into a synthetic chest X-ray image.
# Architecture: Dense → Reshape → Conv2DTranspose blocks → tanh output
# ============================================================

import tensorflow as tf
from tensorflow.keras import layers, Model

from config import IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, LATENT_DIM


def build_generator(
    latent_dim: int = LATENT_DIM,
    img_shape: tuple = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
) -> Model:
    """
    Build and return the Generator model.

    Architecture
    ------------
    Input  : noise vector z  (latent_dim,)
    Output : synthetic image  (H, W, C)  values in [-1, 1]

    The network uses transposed convolutions (deconvolutions) to
    progressively upsample from a 4×4 feature map to the target resolution.
    Batch Normalisation and LeakyReLU are used throughout for stability.
    The final activation is tanh to match the [-1, 1] normalisation used
    in the Preprocessor.
    """

    h, w, c = img_shape
    # Starting spatial dimensions before upsampling
    init_h, init_w = h // 16, w // 16   # e.g. 4×4 for 64×64 images

    noise_input = layers.Input(shape=(latent_dim,), name="noise_input")

    # ── Dense projection ──────────────────────────────────────────────────────
    x = layers.Dense(256 * init_h * init_w, use_bias=False)(noise_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((init_h, init_w, 256))(x)          # (4, 4, 256)

    # ── Upsample block 1  4→8 ────────────────────────────────────────────────
    x = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)                           # (8, 8, 128)

    # ── Upsample block 2  8→16 ───────────────────────────────────────────────
    x = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)                           # (16, 16, 64)

    # ── Upsample block 3  16→32 ──────────────────────────────────────────────
    x = layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)                           # (32, 32, 32)

    # ── Upsample block 4  32→64 ──────────────────────────────────────────────
    x = layers.Conv2DTranspose(c, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
    img_output = layers.Activation("tanh", name="img_output")(x)  # (64, 64, 1)

    model = Model(noise_input, img_output, name="Generator")
    return model


def generator_loss(fake_output: tf.Tensor) -> tf.Tensor:
    """
    Non-saturating generator loss.

    The generator tries to fool the discriminator, so it maximises
    log D(G(z)), equivalent to minimising -log D(G(z)).
    We use binary cross-entropy with *real* labels (1s) for the fake images.
    """
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_labels = tf.ones_like(fake_output)
    return cross_entropy(real_labels, fake_output)
