#!/usr/bin/env python3
# ============================================================
# app.py — Flask Web Application (FIXED)
# Medical Image Synthesis Using GANs for Pulmonary Chest X-rays
#
# FIXES APPLIED:
#  1. Generate route now uses REAL GAN model (not fake numpy shapes)
#  2. n_images input from form is properly respected
#  3. Old images are NOT deleted — gallery accumulates over time
#  4. Per-session timestamped output folders
#  5. Gallery route fixed
#  6. training_state context passed to ALL templates via context_processor
#  7. Improved fallback synthetic generator (varied, not identical)
# ============================================================

import os
import json
import time
import threading
import numpy as np
from datetime import datetime
from flask import (
    Flask, render_template, request, redirect,
    url_for, jsonify, send_from_directory, flash
)
from werkzeug.utils import secure_filename

import config

app = Flask(__name__)
app.secret_key = "medical_gan_secret_key_2024"

UPLOAD_FOLDER = os.path.join(config.BASE_DIR, "uploads")
ALLOWED_EXT   = {"png", "jpg", "jpeg", "dcm"}
app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

training_state = {
    "running": False, "epoch": 0, "total_epochs": 0,
    "g_loss": [], "d_loss": [], "status": "idle", "message": "",
}

# ── Makes training_state available in every template automatically ────────────
@app.context_processor
def inject_globals():
    return {"training_state": training_state}

# ── Helpers ───────────────────────────────────────────────────────────────────

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def get_uploaded_files():
    files = []
    for f in os.listdir(UPLOAD_FOLDER):
        if allowed_file(f):
            full = os.path.join(UPLOAD_FOLDER, f)
            files.append({
                "name": f,
                "size": round(os.path.getsize(full) / 1024, 1),
                "time": datetime.fromtimestamp(os.path.getmtime(full)).strftime("%d %b %Y %H:%M"),
            })
    return sorted(files, key=lambda x: x["time"], reverse=True)

def get_generated_images():
    imgs = []
    for root, _, files in os.walk(config.IMAGE_DIR):
        for f in sorted(files):
            if f.endswith(".png") and "comparison" not in f:
                rel = os.path.relpath(os.path.join(root, f), config.BASE_DIR)
                imgs.append(rel.replace("\\", "/"))
    return imgs

def get_training_log():
    log_path = os.path.join(config.LOG_DIR, "training_log.csv")
    if not os.path.exists(log_path):
        return []
    import csv
    rows = []
    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

# ── FIX #7: Vastly improved fallback synthetic generator ─────────────────────

def _generate_synthetic_fallback(n_images, reference_imgs=None):
    import cv2
    SIZE = config.IMG_HEIGHT
    ref_mean, ref_std = 0.45, 0.15
    if reference_imgs:
        flat = np.concatenate([r.flatten() / 255.0 for r in reference_imgs])
        ref_mean = float(np.mean(flat))
        ref_std  = float(np.std(flat))

    results = []
    for seed_offset in range(n_images):
        rng = np.random.default_rng(seed=int(time.time() * 1000) + seed_offset * 97)
        img = np.zeros((SIZE, SIZE), dtype=np.float32)

        # Background gradient
        base = rng.uniform(ref_mean - 0.1, ref_mean + 0.05)
        grad = rng.uniform(0.08, 0.20)
        for row in range(SIZE):
            img[row, :] = base + grad * (row / SIZE)

        y_grid, x_grid = np.ogrid[:SIZE, :SIZE]

        # Lungs
        left_cx  = rng.integers(int(SIZE * 0.22), int(SIZE * 0.38))
        right_cx = rng.integers(int(SIZE * 0.62), int(SIZE * 0.78))
        lung_cy  = rng.integers(int(SIZE * 0.38), int(SIZE * 0.58))
        lung_rx  = rng.uniform(0.07, 0.11)
        lung_ry  = rng.uniform(0.11, 0.16)
        left_mask  = ((x_grid - left_cx)**2  / (lung_rx * SIZE**2) + (y_grid - lung_cy)**2 / (lung_ry * SIZE**2)) < 1
        right_mask = ((x_grid - right_cx)**2 / (lung_rx * SIZE**2) + (y_grid - lung_cy)**2 / (lung_ry * SIZE**2)) < 1
        lung_val = rng.uniform(ref_mean - 0.25, ref_mean - 0.15)
        img[left_mask]  = lung_val + rng.uniform(-0.03, 0.03)
        img[right_mask] = lung_val + rng.uniform(-0.03, 0.03)
        lung_texture = rng.normal(0, 0.025, img.shape).astype(np.float32)
        img[left_mask | right_mask] += lung_texture[left_mask | right_mask]

        # Spine
        spine_x = SIZE // 2 + rng.integers(-2, 3)
        spine_w = rng.integers(2, 4)
        img[:, max(0, spine_x - spine_w): spine_x + spine_w] = rng.uniform(ref_mean + 0.32, ref_mean + 0.48)

        # Clavicles
        clav_y   = rng.integers(int(SIZE * 0.12), int(SIZE * 0.22))
        clav_val = rng.uniform(ref_mean + 0.18, ref_mean + 0.32)
        for col in range(int(SIZE * 0.15), int(SIZE * 0.85)):
            row = max(0, min(SIZE - 1, clav_y + int(rng.uniform(-1.5, 1.5))))
            img[row, col] = clav_val

        # Ribs
        n_ribs    = rng.integers(6, 10)
        rib_start = int(SIZE * 0.18)
        rib_gap   = rng.uniform(4.5, 7.0)
        rib_val   = rng.uniform(ref_mean + 0.12, ref_mean + 0.28)
        for rib_idx in range(n_ribs):
            rib_y  = int(rib_start + rib_idx * rib_gap + rng.uniform(-1.5, 1.5))
            if rib_y >= SIZE:
                break
            curvature  = rng.uniform(0.03, 0.08)
            brightness = rib_val + rng.uniform(-0.05, 0.05)
            for col in range(int(SIZE * 0.12), int(SIZE * 0.88)):
                curve_y = int(rib_y + curvature * (col - SIZE / 2)**2 / SIZE)
                if 0 <= curve_y < SIZE:
                    img[curve_y, col] = max(img[curve_y, col], brightness)

        # Heart shadow
        heart_cx = spine_x - rng.integers(4, 10)
        heart_cy = lung_cy  + rng.integers(2, 8)
        heart_mask = (
            (x_grid - heart_cx)**2 / (rng.uniform(0.04, 0.07) * SIZE**2) +
            (y_grid - heart_cy)**2 / (rng.uniform(0.05, 0.09) * SIZE**2)
        ) < 1
        img[heart_mask] = np.maximum(img[heart_mask], rng.uniform(ref_mean + 0.08, ref_mean + 0.18))

        # Diaphragm
        diaphragm_y = lung_cy + int(SIZE * rng.uniform(0.18, 0.24))
        for col in range(int(SIZE * 0.08), int(SIZE * 0.92)):
            dy = int(diaphragm_y - 0.04 * (col - SIZE / 2)**2 / SIZE)
            if 0 <= dy < SIZE:
                img[dy, col] = max(img[dy, col], ref_mean + rng.uniform(0.15, 0.25))

        # Film noise + vignette
        img += rng.normal(0, rng.uniform(0.02, ref_std * 0.5), img.shape).astype(np.float32)
        dist = np.sqrt((y_grid - SIZE/2)**2 + (x_grid - SIZE/2)**2)
        img *= (1.0 - rng.uniform(0.08, 0.18) * (dist / (SIZE * 0.7))**2).astype(np.float32)

        img = np.clip(img, 0.0, 1.0)
        img_u8 = (img * 255).astype(np.uint8)
        img_u8 = cv2.GaussianBlur(img_u8, (3, 3), 0)
        results.append(img_u8)
    return results

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    stats = {
        "uploaded":    len(get_uploaded_files()),
        "generated":   len(get_generated_images()),
        "epochs_done": training_state["epoch"],
        "status":      training_state["status"],
    }
    return render_template("index.html", stats=stats)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if "files" not in request.files:
            flash("No files selected.", "error")
            return redirect(request.url)
        files = request.files.getlist("files")
        saved = 0
        for f in files:
            if f and f.filename and allowed_file(f.filename):
                f.save(os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(f.filename)))
                saved += 1
        flash(f"✓ {saved} file(s) uploaded!" if saved else "No valid files found.", "success" if saved else "error")
        return redirect(url_for("upload"))
    return render_template("upload.html", files=get_uploaded_files())


@app.route("/upload/delete/<filename>", methods=["POST"])
def delete_file(filename):
    path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(filename))
    if os.path.exists(path):
        os.remove(path)
        flash(f"Deleted {filename}", "success")
    else:
        flash("File not found.", "error")
    return redirect(url_for("upload"))


@app.route("/train")
def train():
    return render_template("train.html", training=training_state, log=get_training_log())


@app.route("/api/train/start", methods=["POST"])
def api_train_start():
    global training_state
    if training_state["running"]:
        return jsonify({"error": "Training already in progress."}), 400
    data       = request.get_json() or {}
    epochs     = int(data.get("epochs", 100))
    batch_size = int(data.get("batch_size", 32))
    latent_dim = int(data.get("latent_dim", 100))
    training_state.update({"running": True, "epoch": 0, "total_epochs": epochs,
                            "g_loss": [], "d_loss": [], "status": "training", "message": "Initialising…"})

    def _train():
        global training_state
        try:
            from modules.data_acquisition import DataAcquisition
            from modules.preprocessing    import Preprocessor
            from modules.gan              import GAN
            from utils.visualization      import save_sample_grid
            import tensorflow as tf

            training_state["message"] = "Loading dataset…"
            acq  = DataAcquisition(data_dir=UPLOAD_FOLDER)
            raw  = acq.load_images(target_size=(config.IMG_HEIGHT, config.IMG_WIDTH))
            training_state["message"] = "Preprocessing…"
            prep = Preprocessor()
            data_arr = prep.preprocess(raw)
            ds = prep.build_tf_dataset(data_arr, batch_size=batch_size)
            training_state["message"] = "Building GAN…"
            gan = GAN(img_shape=config.IMG_SHAPE, latent_dim=latent_dim)
            fixed_noise = tf.random.normal([16, latent_dim])
            training_state["message"] = "Training…"
            for epoch in range(1, epochs + 1):
                if not training_state["running"]:
                    break
                g_l, d_l = [], []
                for batch in ds:
                    loss = gan.train_step(batch)
                    g_l.append(float(loss["g_loss"]))
                    d_l.append(float(loss["d_loss"]))
                mg, md = np.mean(g_l), np.mean(d_l)
                training_state["epoch"] = epoch
                training_state["g_loss"].append(round(float(mg), 4))
                training_state["d_loss"].append(round(float(md), 4))
                training_state["message"] = f"Epoch {epoch}/{epochs} — G:{mg:.4f} D:{md:.4f}"
                save_every = max(1, epochs // 20)
                if epoch % save_every == 0 or epoch == 1:
                    imgs = gan.generator(fixed_noise, training=False).numpy()
                    save_sample_grid(imgs, os.path.join(config.IMAGE_DIR, f"epoch_{epoch:06d}.png"), title=f"Epoch {epoch}")
            gan.save_weights(
                os.path.join(config.MODEL_DIR, "generator_final.weights.h5"),
                os.path.join(config.MODEL_DIR, "discriminator_final.weights.h5"),
            )
            training_state.update({"running": False, "status": "done", "message": "✓ Training complete!"})
        except Exception as exc:
            training_state.update({"running": False, "status": "error", "message": str(exc)})

    threading.Thread(target=_train, daemon=True).start()
    return jsonify({"status": "started", "epochs": epochs})


@app.route("/api/train/status")
def api_train_status():
    return jsonify(training_state)


@app.route("/api/train/stop", methods=["POST"])
def api_train_stop():
    training_state.update({"running": False, "status": "idle", "message": "Stopped by user."})
    return jsonify({"status": "stopped"})


# ── FIXED generate route ──────────────────────────────────────────────────────
@app.route("/generate", methods=["GET", "POST"])
def generate():
    generated = []
    if request.method == "POST":
        # FIX #2: Properly read n_images
        try:
            n_images = max(1, min(int(request.form.get("n_images", 10)), 50))
        except (ValueError, TypeError):
            n_images = 10

        try:
            # FIX #3 & #4: Timestamped session folder, never delete old images
            session_dir = os.path.join(
                config.IMAGE_DIR, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            os.makedirs(session_dir, exist_ok=True)

            gen_w  = os.path.join(config.MODEL_DIR, "generator_final.weights.h5")
            disc_w = os.path.join(config.MODEL_DIR, "discriminator_final.weights.h5")

            if os.path.exists(gen_w) and os.path.exists(disc_w):
                # ── FIX #1: Use REAL GAN generator ────────────────────────
                import cv2
                import tensorflow as tf
                from modules.gan import GAN

                gan = GAN(img_shape=config.IMG_SHAPE)
                gan.load_weights(gen_w, disc_w)

                # Fresh noise every call → unique images every time
                noise     = tf.random.normal([n_images, config.LATENT_DIM])
                fake_imgs = gan.generator(noise, training=False).numpy()

                saved = []
                for i, img in enumerate(fake_imgs):
                    img_u8 = ((img.squeeze() + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
                    path   = os.path.join(session_dir, f"synthetic_{i:04d}.png")
                    cv2.imwrite(path, img_u8)
                    saved.append(path)

                flash(f"✓ {n_images} GAN-generated synthetic X-ray images created!", "success")

            else:
                # ── FIX #7: Improved fallback (varied, calibrated to uploads) ─
                from PIL import Image as PILImage
                ref_imgs = []
                for fname in os.listdir(UPLOAD_FOLDER):
                    if allowed_file(fname):
                        try:
                            im = PILImage.open(os.path.join(UPLOAD_FOLDER, fname)).convert("L")
                            im = im.resize((config.IMG_WIDTH, config.IMG_HEIGHT))
                            ref_imgs.append(np.array(im))
                        except Exception:
                            pass

                synth = _generate_synthetic_fallback(n_images, ref_imgs or None)
                saved = []
                for i, img_u8 in enumerate(synth):
                    path = os.path.join(session_dir, f"synthetic_{i:04d}.png")
                    PILImage.fromarray(img_u8).save(path)
                    saved.append(path)

                flash(
                    f"⚠ No trained model found. {n_images} procedural X-ray images generated. "
                    "Train the model first for real GAN outputs.", "success"
                )

            generated = [
                os.path.relpath(p, config.BASE_DIR).replace("\\", "/")
                for p in saved
            ]

        except Exception as exc:
            flash(f"Generation error: {exc}", "error")

    return render_template("generate.html", generated=generated, all_images=get_generated_images())


@app.route("/evaluation")
def evaluation():
    metrics, comparison = None, None
    mp = os.path.join(config.OUTPUT_DIR, "metrics.json")
    if os.path.exists(mp):
        with open(mp) as f:
            metrics = json.load(f)
    cp = os.path.join(config.IMAGE_DIR, "comparison.png")
    if os.path.exists(cp):
        comparison = os.path.relpath(cp, config.BASE_DIR).replace("\\", "/")
    return render_template("evaluation.html", metrics=metrics, comparison=comparison, log=get_training_log())


@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    try:
        from modules.data_acquisition import DataAcquisition
        from modules.preprocessing    import Preprocessor
        from modules.gan              import GAN
        from modules.evaluation       import Evaluator
        import tensorflow as tf

        acq  = DataAcquisition(data_dir=UPLOAD_FOLDER)
        raw  = acq.load_images(target_size=(config.IMG_HEIGHT, config.IMG_WIDTH))
        prep = Preprocessor()
        data = prep.preprocess(raw)
        gan  = GAN(img_shape=config.IMG_SHAPE)
        gp   = os.path.join(config.MODEL_DIR, "generator_final.weights.h5")
        dp   = os.path.join(config.MODEL_DIR, "discriminator_final.weights.h5")
        if os.path.exists(gp):
            gan.load_weights(gp, dp)
        evaluator = Evaluator(n_images=config.N_EVAL_IMAGES)
        n  = min(config.N_EVAL_IMAGES, len(data))
        gen = gan.generate_images(n).numpy()
        metrics = evaluator.evaluate(data[:n], gen)
        evaluator.save_comparison(data[:n], gen)
        with open(os.path.join(config.OUTPUT_DIR, "metrics.json"), "w") as f:
            json.dump(metrics, f)
        return jsonify(metrics)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── FIX #5: Gallery returns correct template ──────────────────────────────────
@app.route("/gallery")
def gallery():
    return render_template("gallery.html", images=get_generated_images())


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(config.OUTPUT_DIR, filename)


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  MediGAN — Medical X-Ray Synthesis")
    print("  Open: http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
