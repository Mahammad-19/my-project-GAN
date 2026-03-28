# Medical Image Synthesis Using GANs for Pulmonary Chest X-rays

A final-year BCA project that implements a **Generative Adversarial Network (GAN)**
to synthesise realistic pulmonary chest X-ray images.  
The system addresses the challenge of limited medical imaging datasets by generating
high-quality synthetic radiographs that preserve important anatomical structures
(lung fields, rib patterns, soft-tissue details) while protecting patient privacy.

---

## 📂 Project Structure

```
medical_image_gan/
├── main.py                    ← Entry point
├── config.py                  ← Global configuration
├── requirements.txt
│
├── data/
│   └── sample/                ← Place your chest X-ray images here
│                                (PNG / JPEG / DICOM supported)
│
├── modules/
│   ├── data_acquisition.py    ← Module 1: Load dataset
│   ├── preprocessing.py       ← Module 2: Resize, normalise, denoise, CLAHE
│   ├── generator.py           ← Module 3: Generator network (noise → image)
│   ├── discriminator.py       ← Module 4: Discriminator network (real/fake)
│   ├── gan.py                 ← Module 5: GAN integration + training step
│   ├── training.py            ← Module 6: Full training loop + checkpointing
│   ├── evaluation.py          ← Module 7: SSIM, PSNR, FID metrics
│   └── output_manager.py      ← Module 8: Save synthetic images
│
├── utils/
│   └── visualization.py       ← Loss curves, sample grids, comparisons
│
└── outputs/                   ← Auto-created; stores all outputs
    ├── images/                ← Synthetic images + epoch samples
    ├── models/                ← Model weight checkpoints
    └── logs/                  ← training_log.csv + loss_curves.png
```

---

## ⚙️ Hardware & Software Requirements

| Requirement | Specification |
|---|---|
| Processor | Intel Core i5 or better |
| RAM | 8 GB minimum |
| GPU | NVIDIA GPU with CUDA support (recommended) |
| Storage | 200 GB free disk space |
| OS | Windows 10 / Linux |
| Language | Python 3.9+ |
| Framework | TensorFlow 2.12+ |
| IDE | VS Code / Jupyter Notebook |

---

## 🚀 Setup & Installation

```bash
# 1. Clone or extract the project
cd medical_image_gan

# 2. Create a virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your chest X-ray dataset
#    Place PNG / JPEG / DICOM images in:  data/sample/
```

> **Note:** If you don't have a dataset yet, the system will automatically
> generate random placeholder data so you can test the full pipeline.
>
> Public datasets you can use:
> - [NIH Chest X-ray Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data)
> - [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)

---

## ▶️ Running the Project

### Full Pipeline (recommended)
```bash
python main.py --mode full --epochs 10000 --batch_size 32
```

### Training Only
```bash
python main.py --mode train --epochs 5000
```

### Generate Images from Pre-trained Weights
```bash
python main.py --mode generate \
    --gen_weights outputs/models/generator_final.weights.h5 \
    --disc_weights outputs/models/discriminator_final.weights.h5 \
    --n_generate 20
```

### Evaluation Only
```bash
python main.py --mode eval \
    --gen_weights outputs/models/generator_final.weights.h5 \
    --disc_weights outputs/models/discriminator_final.weights.h5
```

---

## 📊 Evaluation Metrics

| Metric | Description | Goal |
|---|---|---|
| **SSIM** | Structural Similarity Index | Higher is better (max 1.0) |
| **PSNR** | Peak Signal-to-Noise Ratio (dB) | Higher is better |
| **FID** | Fréchet Inception Distance | Lower is better |

---

## 🏗️ GAN Architecture

```
Noise z (100-dim)
       │
  ┌────▼────────────────────────────────────┐
  │           GENERATOR                     │
  │  Dense(256×4×4) → Reshape(4,4,256)     │
  │  Conv2DTranspose(128) → (8,8,128)      │
  │  Conv2DTranspose(64)  → (16,16,64)     │
  │  Conv2DTranspose(32)  → (32,32,32)     │
  │  Conv2DTranspose(1)   → (64,64,1)      │
  │  Activation: tanh                       │
  └────────────────────────────┬────────────┘
                               │ Synthetic Image
  ┌────────────────────────────▼────────────┐
  │           DISCRIMINATOR                 │
  │  Conv2D(32)  → LeakyReLU → Dropout     │
  │  Conv2D(64)  → BN → LeakyReLU         │
  │  Conv2D(128) → BN → LeakyReLU         │
  │  Conv2D(256) → BN → LeakyReLU         │
  │  Flatten → Dense(1)  [logit]           │
  └─────────────────────────────────────────┘
```

---

## 📁 Output Files

After training you will find:

```
outputs/
├── images/
│   ├── epoch_000001.png       ← Sample grid after epoch 1
│   ├── epoch_000500.png       ← Sample grid every 500 epochs
│   ├── comparison.png         ← Real vs synthetic side-by-side
│   └── generated_<date>/      ← Final batch of synthetic images
│       ├── synthetic_0000.png
│       └── ...
├── models/
│   ├── generator_epoch_001000.weights.h5
│   └── discriminator_final.weights.h5
└── logs/
    ├── training_log.csv
    └── loss_curves.png
```

---

## 👤 Project Information

**Project Title:** Medical Image Synthesis Using GANs for Pulmonary Chest X-rays  
**Degree:** Bachelor of Computer Applications (BCA)  
**Technology Stack:** Python · TensorFlow · Keras · OpenCV · NumPy · Matplotlib
