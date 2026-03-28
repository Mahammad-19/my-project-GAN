# ============================================================
# modules/evaluation.py — Evaluation Module
# Computes SSIM, PSNR, and FID metrics for generated images.
# ============================================================

import os
import numpy as np
import tensorflow as tf

from config import N_EVAL_IMAGES, IMAGE_DIR


class Evaluator:
    """
    Evaluates the quality of GAN-generated synthetic chest X-ray images
    using standard image quality metrics:

    * SSIM  — Structural Similarity Index  (higher is better, max 1.0)
    * PSNR  — Peak Signal-to-Noise Ratio   (higher is better, in dB)
    * FID   — Fréchet Inception Distance   (lower is better)
    """

    def __init__(self, n_images: int = N_EVAL_IMAGES):
        self.n_images = n_images

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        real_images: np.ndarray,
        generated_images: np.ndarray,
    ) -> dict:
        """
        Compute all metrics and return a summary dictionary.

        Parameters
        ----------
        real_images      : np.ndarray  (N, H, W, C)  range [-1, 1]
        generated_images : np.ndarray  (M, H, W, C)  range [-1, 1]

        Returns
        -------
        dict with keys: ssim, psnr, fid
        """
        n = min(self.n_images, len(real_images), len(generated_images))
        real   = real_images[:n]
        fake   = generated_images[:n]

        ssim_val = self._compute_ssim(real, fake)
        psnr_val = self._compute_psnr(real, fake)
        fid_val  = self._compute_fid(real, fake)

        results = {
            "ssim": round(float(ssim_val), 4),
            "psnr": round(float(psnr_val), 4),
            "fid":  round(float(fid_val), 4),
        }

        self._print_results(results)
        return results

    # ── Metric implementations ────────────────────────────────────────────────

    @staticmethod
    def _compute_ssim(real: np.ndarray, fake: np.ndarray) -> float:
        """
        Mean SSIM across N image pairs.
        Images are expected in [-1, 1]; we shift to [0, 1] for the metric.
        """
        from skimage.metrics import structural_similarity as ssim

        real_01 = (real + 1.0) / 2.0
        fake_01 = (fake + 1.0) / 2.0

        scores = []
        for r, f in zip(real_01, fake_01):
            r_sq = r.squeeze()
            f_sq = f.squeeze()
            # win_size must be odd and ≤ min image dimension
            win = min(7, r_sq.shape[0], r_sq.shape[1])
            if win % 2 == 0:
                win -= 1
            score = ssim(r_sq, f_sq, data_range=1.0, win_size=win)
            scores.append(score)

        return float(np.mean(scores))

    @staticmethod
    def _compute_psnr(real: np.ndarray, fake: np.ndarray) -> float:
        """
        Mean PSNR across N image pairs.
        Uses MSE; handles the edge case of identical images (infinite PSNR).
        """
        scores = []
        for r, f in zip(real, fake):
            mse = np.mean((r.astype(np.float64) - f.astype(np.float64)) ** 2)
            if mse == 0:
                scores.append(100.0)   # Identical images → cap at 100 dB
            else:
                scores.append(10.0 * np.log10(4.0 / mse))  # data range = 2 → max²=4
        return float(np.mean(scores))

    @staticmethod
    def _compute_fid(real: np.ndarray, fake: np.ndarray) -> float:
        """
        Simplified FID using raw pixel features (no Inception network).

        A full FID requires an Inception v3 network and large batches.
        This lightweight version uses flattened pixel vectors to compute
        the Fréchet distance between Gaussian fits of real and fake distributions.
        For a production system, use the `torch-fidelity` or `tensorflow-gan`
        library for proper Inception-based FID.
        """
        from scipy.linalg import sqrtm

        def stats(imgs):
            flat = imgs.reshape(len(imgs), -1).astype(np.float64)
            mu   = flat.mean(axis=0)
            cov  = np.cov(flat, rowvar=False)
            return mu, cov

        mu1, cov1 = stats(real)
        mu2, cov2 = stats(fake)

        diff = mu1 - mu2
        # Compute sqrt of product of covariance matrices
        covmean, _ = sqrtm(cov1 @ cov2, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = float(diff @ diff + np.trace(cov1 + cov2 - 2.0 * covmean))
        return max(0.0, fid)   # Numerical noise can push it slightly negative

    # ── Reporting ─────────────────────────────────────────────────────────────

    @staticmethod
    def _print_results(results: dict):
        print("\n" + "=" * 40)
        print("  Evaluation Results")
        print("=" * 40)
        print(f"  SSIM  : {results['ssim']:.4f}  (higher is better, max 1.0)")
        print(f"  PSNR  : {results['psnr']:.4f} dB  (higher is better)")
        print(f"  FID   : {results['fid']:.4f}  (lower is better)")
        print("=" * 40 + "\n")

    def save_comparison(
        self,
        real_images: np.ndarray,
        generated_images: np.ndarray,
        save_path: str = None,
    ):
        """Save a side-by-side visual comparison of real vs. synthetic images."""
        from utils.visualization import save_comparison_grid

        save_path = save_path or os.path.join(IMAGE_DIR, "comparison.png")
        n = min(4, len(real_images), len(generated_images))
        save_comparison_grid(real_images[:n], generated_images[:n], save_path)
        print(f"[Evaluator] Comparison grid saved → {save_path}")
