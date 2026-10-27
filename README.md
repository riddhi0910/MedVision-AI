# 🫁 MedVision AI — Clinical PACS Diagnostic Radiograph Workstation & Explainable AI (XAI) Suite

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Starlette](https://img.shields.io/badge/Starlette-ASGI-009688.svg)](https://www.starlette.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-ff4b4b.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776ab.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Accuracy](https://img.shields.io/badge/ResNet--18%20Accuracy-99.1%25-brightgreen.svg)]()
[![Recall](https://img.shields.io/badge/Sensitivity%2FRecall-98.6%25-success.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Nagpur City Hospital — Department of Diagnostic Radiology & Pulmonology**  
> An explainable deep learning clinical decision support system (CDSS) for automated detection of pulmonary airspace consolidation and pneumonia from chest radiographs with pixel-level Grad-CAM visual heatmaps and instant formal medical consultation reports.

---

## 🏥 Clinical Workstation Interfaces

MedVision AI provides two dedicated user interfaces tailored for radiologists and clinical researchers:

| Interface | Port | Architecture | Core Highlights |
| :--- | :--- | :--- | :--- |
| **PACS Diagnostic Workstation** | `8000` | **Starlette + Vanilla JS/HTML5** | **Zero-scroll single-screen clinical console**, side-by-side raw DX and AI Saliency Map, dynamic Grad-CAM blending (0-100%), Windowing LUT inversion, Nagpur City Hospital letterhead, and 1-click Dr. R. Somani attested PDF consultation report. |
| **Streamlit Research Explorer** | `8501` | **Python (Streamlit)** | Model comparison benchmark playground, interactive confidence threshold tuning, and raw attention tensor inspection. |

---

## 📚 Complete Project Documentation Hub

| Document | Purpose |
| :--- | :--- |
| 🎓 **[docs/explanation.md](docs/explanation.md)** | **Teacher / Professor / Viva Presentation Guide**: Simple Hinglish script, exact XAI definition, 850 test cases confusion matrix analysis, precision-recall breakdown, and top 10 viva questions & answers. |
| 🚀 **[docs/start.md](docs/start.md)** | **Startup & Environment Guide**: Step-by-step virtual environment setup, pip dependencies, direct Python commands, and NPM shortcuts. |
| 📖 **[docs/study.md](docs/study.md)** | **Deep Learning Textbook**: From-scratch masterclass on chest radiography, CNN architectures, ResNet-18 residual math, Grad-CAM equations, and clinical evaluation. |
| 📘 **[docs/learning.md](docs/learning.md)** | **Curriculum Roadmap**: Progressive 6-week DL roadmap covering foundations to clinical deployment. |
| 🔬 **[docs/info.md](docs/info.md)** | **Technical Specifications**: Hyperparameters, model architecture diagrams, loss functions, and dataset metadata. |
| 🗺️ **[docs/roadmap.md](docs/roadmap.md)** | **Project Roadmap**: Architectural milestones, completed tasks, and future multi-class clinical roadmap. |

---

## 📊 Benchmark Model Performance (850 Test Radiographs)

Both models were rigorously trained on the balanced Kaggle Chest X-Ray dataset (8,490 total images after clinical parity augmentation) and evaluated on a strictly held-out test split of 850 unseen radiographs (425 Normal : 425 Pneumonia):

| Clinical Metric | Baseline CNN (From Scratch) | ResNet-18 (Transfer Learning) | Clinical Significance |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 94.71% | **99.06% (~99.1%)** | +4.35% improvement across diverse cases |
| **Recall / Sensitivity** | 92.71% | **98.59% (~98.6%)** | **Critical:** Catches 419 out of 425 true infections |
| **Specificity (TNR)** | 96.71% | **99.53% (~99.5%)** | 423 out of 425 healthy patients correctly cleared |
| **Precision (PPV)** | 96.56% | **99.52% (~99.5%)** | Positive predictions are virtually error-free |
| **F1-Score** | 0.9460 | **0.9905 (~0.990)** | Exceptional harmonic balance |
| **ROC-AUC** | 0.9902 | **0.9994 (~1.000)** | Near-perfect class separability |
| **False Negatives (FN)** | **31 Missed Cases** | **Only 6 Missed Cases** | **80% reduction in dangerous missed diagnoses!** |
| **False Positives (FP)** | 14 False Alarms | **Only 2 False Alarms** | Prevents unnecessary hospitalizations |

> 📁 Visual Confusion Matrix Plot: **`artifacts/reports/confusion_matrices_comparison.png`**  
> 📓 Fully Executed Training Log & Curves: **`notebooks/Executed MedVision_Colab_OneClick.ipynb`**

---

## 🔍 Explainable AI (XAI) & Grad-CAM

Medical professionals cannot rely on "black-box" models. MedVision AI integrates **Gradient-Weighted Class Activation Mapping (Grad-CAM)**:

```
Input Radiograph (224x224) ──► ResNet-18 Forward Pass ──► Prediction Score (y^c)
                                       │
                                       ▼ (Backward Pass)
                        Gradients d(y^c) / d(A^k) at layer4
                                       │
                                       ▼
                        Global Average Pooling ──► Feature Weights (alpha_k)
                                       │
                                       ▼
                        ReLU Weighted Combination ──► Visual Heatmap
                                       │
                                       ▼
                        Overlay on Radiograph (JET Colormap)
```

- **Red & Yellow (1.0 Attention):** Highlights exact focal consolidation, air bronchograms, and lower lobe pulmonary opacities.
- **Cool Blue (0.0 Attention):** Background, spine, clavicles, and external hospital marker tags ("L", "Panic film") are strictly ignored.

---

## ⚡ Quick Start Guide

### 1. Prerequisites & Virtual Environment
```powershell
# Clone repo & enter folder
git clone https://github.com/riddhi0910/MedVision-AI.git
cd MedVision-AI

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
.\venv\Scripts\activate.bat
# Linux/macOS:
source venv/bin/activate

# Install dependencies (server + Streamlit + notebooks + tests)
pip install -r requirements-dev.txt
```

Trained weights (`artifacts/checkpoints/best_resnet18.pth`, `best_cnn.pth`) are committed, so no training or dataset download is needed to run the app.

### 2. Launch the Application

#### Option A: Clinical PACS Workstation (Port 8000) — *Recommended*
```powershell
# Native Python (No Node.js required):
python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000 --reload

# Or via npm (if Node.js is installed):
npm start
```
👉 Open Workstation: **[http://localhost:8000](http://localhost:8000)**  
👉 Health check: **[http://localhost:8000/api/health](http://localhost:8000/api/health)**

#### Option B: Streamlit Medical Explorer (Port 8501)
```powershell
streamlit run app.py
# Or via npm:
npm run streamlit
```
👉 Open Streamlit: **[http://localhost:8501](http://localhost:8501)**

### 3. Deploy on Render (free tier)
1. Push this repo to GitHub.
2. On [render.com](https://render.com): **New → Blueprint** → select the repo. Render reads [render.yaml](render.yaml).
3. Wait for the build (installs CPU-only PyTorch), then open the `.onrender.com` URL.

Free instances sleep after 15 min idle; the first request after that takes ~1 min to wake up.

---

## 📦 NPM Shortcuts Reference

Root contains a standard `package.json` for developers using npm:

```bash
npm start           # Launches PACS Workstation on port 8000
npm run dev         # Launches PACS Workstation with live reload
npm run streamlit   # Launches Streamlit interface on port 8501
npm run pipeline    # Executes end-to-end training pipeline
npm test            # Runs automated pytest suite
```

---

## 📁 Repository Structure

```text
MedVision-AI/
├── backend/
│   └── server.py                   # Starlette API (/api/predict, /api/models, /api/samples, /api/health) + serves frontend
├── frontend/
│   └── public/                     # Single-screen zero-scroll PACS UI (HTML5, CSS3, JS)
│       ├── index.html              # Dual monitor viewport & consultation letterhead
│       ├── style.css               # Medical workstation dark-mode design system
│       └── app.js                  # Asynchronous client, Grad-CAM blend & PDF printer
├── src/                            # ML package (single source of truth for all code)
│   ├── data/                       # Dataset loading & preprocessing transforms
│   ├── models/                     # cnn.py (4-block baseline) & resnet.py (ResNet-18 transfer learning)
│   ├── training/                   # Training loop & evaluation
│   ├── inference/                  # Prediction & Grad-CAM
│   └── utils/                      # Config loader, metrics, plots
├── artifacts/
│   ├── checkpoints/                # Trained weights: best_resnet18.pth, best_cnn.pth (committed)
│   ├── reports/                    # Confusion matrices and Grad-CAM sample figures
│   └── sample_xrays/               # Real test radiographs for 1-click demo
├── configs/config.yaml             # All paths & hyperparameters
├── notebooks/                      # GPU Training Notebooks
│   ├── Executed MedVision_Colab_OneClick.ipynb  # Pre-computed loss curves, CM & Grad-CAM
│   └── MedVision_Colab_OneClick.ipynb           # Fresh Colab 1-click training notebook
├── scripts/                        # Offline dataset curation (run in order)
│   ├── build_custom_dataset.py     # 1. Integrity check, MD5 dedup, 80/10/10 split
│   ├── balance_and_split_dataset.py# 2. 50-50 class balancing by augmentation
│   ├── flatten_data_to_two_classes.py # 3. Pool into NORMAL/ and PNEUMONIA/
│   └── make_physical_splits.py     # 4. Deterministic train/val/test directory partition
├── docs/                           # Viva guide, startup guide, study notes, roadmap
├── tests/                          # pytest suite
├── app.py                          # Streamlit research explorer
├── run_pipeline.py                 # End-to-end training pipeline
├── render.yaml                     # Render deployment blueprint
├── requirements.txt                # Runtime dependencies (what Render installs)
├── requirements-dev.txt            # + Streamlit, Jupyter, pytest for local work
└── package.json                    # Optional npm shortcuts
```

---

## 🧪 Automated Testing

```powershell
# Run full pytest suite:
pytest
```
Tests automatically validate tensor shapes, parameter unfreezing, normalization ranges, and endpoint responses.

---

## ⚠️ Medical Disclaimer

This software is an academic engineering prototype developed for demonstration, educational, and research purposes. It is **not FDA or CE certified as a diagnostic medical device** and should not replace professional clinical evaluation by licensed radiologists.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
