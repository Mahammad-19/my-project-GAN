# ============================================================
# utils/visualization.py — Visualization Utilities
# Plotting helpers for training loss curves, sample grids, and comparisons.
# ============================================================

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server environments
import matplotlib.pyplot as plt


# ── Sample image grids ────────────────────────────────────────────────────────

def save_sample_grid(
    images: np.ndarray,
    save_path: str,
    title: str = "",
    nrow: int = 4,
    figsize: tuple = (8, 8),
):
    """
    Save a grid of synthetic images to disk.

    Parameters
    ----------
    images    : np.ndarray  (N, H, W, C)  range [-1, 1]
    save_path : str         full path for the output PNG
    title     : str         plot title
    nrow      : int         images per row
    """
    n = len(images)
    ncol = nrow
    nrow = int(np.ceil(n / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    axes = np.array(axes).flatten()

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            img = images[i].squeeze()
            img_01 = (img + 1.0) / 2.0   # [-1,1] → [0,1]
            ax.imshow(img_01, cmap="gray", vmin=0, vmax=1)

    if title:
        fig.suptitle(title, fontsize=14, y=1.01)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def save_comparison_grid(
    real_images: np.ndarray,
    fake_images: np.ndarray,
    save_path: str,
    figsize: tuple = (10, 5),
):
    """
    Save a two-row comparison grid: real images (top) vs. synthetic (bottom).

    Parameters
    ----------
    real_images : (N, H, W, C)  real chest X-rays,    range [-1, 1]
    fake_images : (N, H, W, C)  generated images,     range [-1, 1]
    save_path   : str
    """
    n = min(len(real_images), len(fake_images))
    fig, axes = plt.subplots(2, n, figsize=figsize)

    for i in range(n):
        for row, imgs, label in [(0, real_images, "Real"), (1, fake_images, "Synthetic")]:
            ax = axes[row][i] if n > 1 else axes[row]
            ax.axis("off")
            img = (imgs[i].squeeze() + 1.0) / 2.0
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            if i == 0:
                ax.set_title(label, fontsize=11, pad=4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"[Visualization] Comparison grid saved → {save_path}")


# ── Loss curves ───────────────────────────────────────────────────────────────

def plot_loss_curves(log_csv_path: str, save_path: str):
    """
    Read the CSV training log and plot G/D loss curves.

    Parameters
    ----------
    log_csv_path : str  path to training_log.csv
    save_path    : str  path for the output PNG
    """
    import csv

    epochs, g_losses, d_losses = [], [], []

    with open(log_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            g_losses.append(float(row["g_loss"]))
            d_losses.append(float(row["d_loss"]))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, g_losses, label="Generator Loss",     color="#E07B54", linewidth=1.5)
    ax.plot(epochs, d_losses, label="Discriminator Loss", color="#5B9BD5", linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("GAN Training Loss Curves", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"[Visualization] Loss curves saved → {save_path}")
