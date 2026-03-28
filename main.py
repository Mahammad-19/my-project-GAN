#!/usr/bin/env python3
# ============================================================
# main.py — Entry Point
# Medical Image Synthesis Using GANs for Pulmonary Chest X-rays
#
# Usage:
#   python main.py                 # full pipeline (train + eval + generate)
#   python main.py --mode train    # training only
#   python main.py --mode eval     # evaluation only (needs saved weights)
#   python main.py --mode generate # generate images only
#
# Place your chest X-ray images (PNG/JPEG/DICOM) inside:
#   data/sample/
# ============================================================

import argparse
import os
import numpy as np

import config  # ensures output directories are created

from modules.data_acquisition import DataAcquisition
from modules.preprocessing     import Preprocessor
from modules.gan               import GAN
from modules.training          import Trainer
from modules.evaluation        import Evaluator
from modules.output_manager    import OutputManager
from utils.visualization       import plot_loss_curves


# ── CLI argument parsing ──────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="GAN-based Pulmonary Chest X-ray Image Synthesis"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "generate", "full"],
        default="full",
        help="Pipeline mode (default: full)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
        help=f"Number of training epochs (default: {config.EPOCHS})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.BATCH_SIZE,
        help=f"Batch size (default: {config.BATCH_SIZE})",
    )
    parser.add_argument(
        "--n_generate",
        type=int,
        default=10,
        help="Number of synthetic images to generate (default: 10)",
    )
    parser.add_argument(
        "--gen_weights",
        type=str,
        default=None,
        help="Path to pre-trained generator weights (.weights.h5)",
    )
    parser.add_argument(
        "--disc_weights",
        type=str,
        default=None,
        help="Path to pre-trained discriminator weights (.weights.h5)",
    )
    return parser.parse_args()


# ── Pipeline stages ───────────────────────────────────────────────────────────

def load_and_preprocess(batch_size: int):
    """Stage 1 & 2: Data acquisition + preprocessing."""
    print("\n[Pipeline] Stage 1 — Data Acquisition")
    acq  = DataAcquisition(data_dir=config.DATA_DIR)
    raw  = acq.load_images(target_size=(config.IMG_HEIGHT, config.IMG_WIDTH))

    print("\n[Pipeline] Stage 2 — Preprocessing")
    prep = Preprocessor(apply_denoise=True, apply_clahe=True)
    data = prep.preprocess(raw)

    train_data, val_data = prep.train_val_split(data, val_ratio=0.15)
    print(f"  Train samples: {len(train_data)}  Val samples: {len(val_data)}")

    tf_dataset = prep.build_tf_dataset(train_data, batch_size=batch_size)
    return tf_dataset, train_data, val_data


def build_gan():
    """Stage 3: Build Generator + Discriminator + GAN."""
    print("\n[Pipeline] Stage 3 — Building GAN")
    gan = GAN(
        img_shape=config.IMG_SHAPE,
        latent_dim=config.LATENT_DIM,
        lr=config.LEARNING_RATE,
        beta_1=config.BETA_1,
    )
    return gan


def run_training(gan, tf_dataset, epochs: int):
    """Stage 4: Adversarial training."""
    print("\n[Pipeline] Stage 4 — Training")
    trainer = Trainer(
        gan=gan,
        dataset=tf_dataset,
        epochs=epochs,
        latent_dim=config.LATENT_DIM,
    )
    trainer.train()

    # Plot loss curves after training
    log_csv = os.path.join(config.LOG_DIR, "training_log.csv")
    loss_plot = os.path.join(config.LOG_DIR, "loss_curves.png")
    if os.path.exists(log_csv):
        plot_loss_curves(log_csv, loss_plot)


def run_evaluation(gan, real_images: np.ndarray):
    """Stage 5: Evaluation using SSIM, PSNR, FID."""
    print("\n[Pipeline] Stage 5 — Evaluation")
    evaluator = Evaluator(n_images=config.N_EVAL_IMAGES)

    n = min(config.N_EVAL_IMAGES, len(real_images))
    generated = gan.generate_images(n).numpy()

    metrics = evaluator.evaluate(real_images[:n], generated)
    evaluator.save_comparison(real_images[:n], generated)
    return metrics


def run_generation(gan, n_images: int):
    """Stage 6: Generate and save synthetic images."""
    print("\n[Pipeline] Stage 6 — Output Generation")
    output_mgr = OutputManager(
        output_dir=config.IMAGE_DIR,
        latent_dim=config.LATENT_DIM,
    )
    paths = output_mgr.generate_and_save(gan.generator, n_images=n_images)
    print(f"  Generated {len(paths)} synthetic chest X-ray images.")
    return paths


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    print("\n" + "=" * 60)
    print("  Medical Image Synthesis Using GANs")
    print("  Pulmonary Chest X-ray Generation")
    print("=" * 60)

    # ── Full pipeline ─────────────────────────────────────────────────────────
    if args.mode in ("full", "train"):
        tf_dataset, train_data, val_data = load_and_preprocess(args.batch_size)
        gan = build_gan()

        if args.gen_weights and args.disc_weights:
            print("[main] Loading pre-trained weights …")
            gan.load_weights(args.gen_weights, args.disc_weights)

        if args.mode in ("full", "train"):
            run_training(gan, tf_dataset, args.epochs)

        if args.mode == "full":
            run_evaluation(gan, val_data)
            run_generation(gan, args.n_generate)

    # ── Eval-only ─────────────────────────────────────────────────────────────
    elif args.mode == "eval":
        _, train_data, val_data = load_and_preprocess(args.batch_size)
        gan = build_gan()
        if args.gen_weights and args.disc_weights:
            gan.load_weights(args.gen_weights, args.disc_weights)
        else:
            print("[WARN] No weights provided — evaluating untrained model.")
        run_evaluation(gan, val_data)

    # ── Generate-only ─────────────────────────────────────────────────────────
    elif args.mode == "generate":
        gan = build_gan()
        if args.gen_weights and args.disc_weights:
            gan.load_weights(args.gen_weights, args.disc_weights)
        else:
            print("[WARN] No weights provided — generating from untrained model.")
        run_generation(gan, args.n_generate)

    print("\n[Pipeline] Done. Outputs saved to:", config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
