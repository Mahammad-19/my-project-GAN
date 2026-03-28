# ============================================================
# modules/data_acquisition.py — Data Acquisition Module
# Loads chest X-ray images from local directories.
# Supports PNG, JPEG, and DICOM (.dcm) formats.
# ============================================================

import os
import numpy as np
from pathlib import Path

try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False

from config import DATA_DIR, SUPPORTED_FORMATS


class DataAcquisition:
    """
    Loads chest X-ray images from a directory.

    Supports PNG / JPEG (via OpenCV) and DICOM (.dcm via pydicom).
    Returns raw NumPy arrays of shape (N, H, W, C).
    """

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    # ── Public API ────────────────────────────────────────────────────────────

    def load_images(self, target_size: tuple = (64, 64)) -> np.ndarray:
        """
        Recursively scan ``self.data_dir`` and load all supported images.

        Parameters
        ----------
        target_size : (height, width)

        Returns
        -------
        np.ndarray  shape (N, H, W, 1), dtype float32, range [0, 255]
        """
        import cv2

        image_paths = self._collect_paths()
        if not image_paths:
            print(f"[DataAcquisition] No images found in '{self.data_dir}'.")
            print("  → Generating synthetic random data for demonstration.")
            return self._make_dummy_data(target_size)

        images = []
        for path in image_paths:
            img = self._load_single(path, target_size)
            if img is not None:
                images.append(img)

        if not images:
            print("[DataAcquisition] All files failed to load — using dummy data.")
            return self._make_dummy_data(target_size)

        dataset = np.stack(images, axis=0)          # (N, H, W, 1)
        print(f"[DataAcquisition] Loaded {len(dataset)} images  shape={dataset.shape}")
        return dataset

    # ── Private helpers ───────────────────────────────────────────────────────

    def _collect_paths(self) -> list:
        paths = []
        for root, _, files in os.walk(self.data_dir):
            for fname in files:
                if Path(fname).suffix.lower() in SUPPORTED_FORMATS:
                    paths.append(os.path.join(root, fname))
        return sorted(paths)

    def _load_single(self, path: str, target_size: tuple):
        import cv2

        ext = Path(path).suffix.lower()
        try:
            if ext == ".dcm":
                return self._load_dicom(path, target_size)
            else:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    return None
                img = cv2.resize(img, (target_size[1], target_size[0]))
                return img[..., np.newaxis].astype(np.float32)
        except Exception as exc:
            print(f"  [WARN] Could not load {path}: {exc}")
            return None

    def _load_dicom(self, path: str, target_size: tuple):
        import cv2

        if not DICOM_AVAILABLE:
            raise ImportError("Install pydicom to read .dcm files: pip install pydicom")
        ds = pydicom.dcmread(path)
        pixel_array = ds.pixel_array.astype(np.float32)
        # Normalise to 0-255
        lo, hi = pixel_array.min(), pixel_array.max()
        if hi > lo:
            pixel_array = (pixel_array - lo) / (hi - lo) * 255.0
        img = cv2.resize(pixel_array, (target_size[1], target_size[0]))
        return img[..., np.newaxis].astype(np.float32)

    @staticmethod
    def _make_dummy_data(target_size: tuple, n: int = 100) -> np.ndarray:
        """Return random noise as placeholder when no real data is available."""
        h, w = target_size
        data = np.random.randint(0, 256, (n, h, w, 1)).astype(np.float32)
        print(f"[DataAcquisition] Dummy dataset created  shape={data.shape}")
        return data
