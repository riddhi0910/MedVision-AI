# 🫁 MedVision AI — Clinical PACS Diagnostic Radiograph Workstation & Explainable AI (XAI) Suite

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
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
| **PACS Diagnostic Workstation** | `8000` | **FastAPI + Vanilla JS/HTML5** | **Zero-scroll single-screen clinical console**, side-by-side raw DX and AI Saliency Map, dynamic Grad-CAM blending (0-100%), Windowing LUT inversion, Nagpur City Hospital letterhead, and 1-click Dr. R. Somani attested PDF consultation report. |
| **Streamlit Research Explorer** | `8501` | **Python (Streamlit)** | Model comparison benchmark playground, interactive confidence threshold tuning, and raw attention tensor inspection. |

---

## 📚 Complete Project Documentation Hub

| Document | Purpose |
| :--- | :--- |
| 🎓 **[explanation.md](explanation.md)** | **Teacher / Professor / Viva Presentation Guide**: Simple Hinglish script, exact XAI definition, 850 test cases confusion matrix analysis, precision-recall breakdown, and top 10 viva questions & answers. |
| 🚀 **[start.md](start.md)** | **Startup & Environment Guide**: Step-by-step virtual environment setup, pip dependencies, direct Python commands, and NPM shortcuts. |
| 📖 **[study.md](study.md)** | **Deep Learning Textbook**: From-scratch masterclass on chest radiography, CNN architectures, ResNet-18 residual math, Grad-CAM equations, and clinical evaluation. |
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

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Application

#### Option A: Clinical PACS Workstation (Port 8000) — *Recommended*
```powershell
# Native Python (No Node.js required):
python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000 --reload

# Or via npm (if Node.js is installed):
npm start
```
👉 Open Workstation: **[http://localhost:8000](http://localhost:8000)**  
👉 Interactive OpenAPI / Swagger Docs: **[http://localhost:8000/docs](http://localhost:8000/docs)**

#### Option B: Streamlit Medical Explorer (Port 8501)
```powershell
streamlit run app.py
# Or via npm:
npm run streamlit
```
👉 Open Streamlit: **[http://localhost:8501](http://localhost:8501)**

---

## 📦 NPM Shortcuts Reference

Root contains a standard `package.json` for developers using npm:

```bash
npm start           # Launches PACS Workstation on port 8000
npm run dev         # Launches PACS Workstation with live reload
npm run streamlit   # Launches Streamlit interface on port 8501
npm run demo        # Generates test checkpoints and sample X-rays
npm run pipeline    # Executes end-to-end training pipeline
npm test            # Runs automated pytest suite
```

---

## 📁 Repository Structure

```text
MedVision-AI/
├── backend/                        # FastAPI REST API & In-Memory Inference Engine
│   └── server.py                   # Endpoints (/api/analyze, /api/models, /api/health)
├── frontend/                       # Clinical Diagnostic PACS Workstation
│   └── public/                     # Single-screen zero-scroll UI (HTML5, CSS3, JS)
│       ├── index.html              # Dual monitor viewport & consultation letterhead
│       ├── style.css               # Medical workstation dark-mode design system
│       └── app.js                  # Asynchronous client, Grad-CAM blend & PDF printer
├── explanation.md                  # 🎓 Full Teacher / Viva Presentation Guide (Hinglish)
├── start.md                        # 🚀 Comprehensive Startup & Environment Guide
├── study.md                        # 📖 From-Scratch Deep Learning Masterclass Textbook
├── models/                         # PyTorch Neural Architectures
│   ├── checkpoints/                # Model weights (.pth ignored; tracked via .gitkeep)
│   ├── cnn.py                      # Custom 4-block CNN architecture
│   └── resnet.py                   # ResNet-18 residual transfer learning architecture
├── notebooks/                      # GPU Training Notebooks
│   ├── Executed MedVision_Colab_OneClick.ipynb  # Pre-computed loss curves, CM & Grad-CAM
│   └── MedVision_Colab_OneClick.ipynb           # Fresh Colab 1-click training notebook
├── scripts/                        # Utility & Preprocessing Scripts
│   ├── create_demo_checkpoints.py  # Instant test weights & sample radiograph generator
│   ├── balance_and_split_dataset.py# 50-50 clinical data augmentation and split
│   └── make_physical_splits.py     # Deterministic train/val/test directory partition
├── artifacts/                      # Experiment artifacts & report plots
│   └── reports/                    # Confusion matrices and Grad-CAM sample figures
├── docs/                           # Synced documentation mirrors
├── src/                            # Modular Python backend package
├── app.py                          # Streamlit interactive diagnostic dashboard
├── run_pipeline.py                 # End-to-end automated ML pipeline
├── package.json                    # NPM developer script shortcuts
├── requirements.txt                # Production Python dependencies
└── .gitignore                      # Excludes >2.5GB datasets, virtual environments, binaries
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
