# 🫁 MedVision AI — Explainable Chest X-Ray Classification

> **Final-Year B.Tech Computer Science / AI-ML Project**
> An explainable deep learning research prototype that classifies chest X-ray images and visualises model attention using Grad-CAM.

---

## ⚠️ Medical Disclaimer

This application is an **educational research prototype** built as a final-year B.Tech project.  
It is **NOT a certified medical device**, has **NOT been clinically validated**, and must **NOT** be used to make any clinical or diagnostic decisions.  
Always consult a qualified radiologist or physician for medical advice.

---

## 📋 Project Overview

| Item | Details |
|---|---|
| **Task** | Binary classification: NORMAL vs PNEUMONIA |
| **Dataset** | Chest X-Ray Images (Pneumonia) — Kaggle |
| **Models** | Baseline CNN + ResNet-18 (transfer learning) |
| **Explainability** | Grad-CAM heatmaps |
| **UI** | Streamlit web application |
| **Language** | Python 3.10+ |
| **Framework** | PyTorch 2.1+ |

---

## 📚 Documentation & Study Hub

| Guide | Description |
| :--- | :--- |
| 🚀 **[docs/start.md](docs/start.md)** | **Project Startup Guide**: One-click run commands for PACS (8000) and Streamlit (8501). |
| 📖 **[docs/study.md](docs/study.md)** | **Study Masterclass**: From-scratch textbook on CXR, CNNs, ResNet-18, Grad-CAM math & viva Q&A. |
| 📘 **[docs/learning.md](docs/learning.md)** | **Curriculum Roadmap**: 6-week progressive study plan and learning philosophy. |
| 🔬 **[docs/info.md](docs/info.md)** | **Technical Specifications**: Pipeline internals, hyperparameters, and dataset stats. |
| 🗺️ **[docs/roadmap.md](docs/roadmap.md)** | **Development Blueprint**: Architectural milestones and future clinical roadmap. |

---

## 📁 Project Structure

```text
MedVision-AI/
├── backend/                   # FastAPI backend server & inference engine (Port 8000)
│   └── server.py              # ASGI application, endpoints & in-memory image streaming
├── frontend/                  # Clinical Diagnostic PACS Workstation
│   └── public/                # 100vh single-screen PACS UI (index.html, style.css, app.js)
├── models/                    # Model definitions & trained weights
│   ├── checkpoints/           # Saved PyTorch weights (best_resnet18.pth, best_cnn.pth)
│   ├── cnn.py                 # Custom 4-block CNN architecture
│   └── resnet.py              # Transfer learning ResNet-18 architecture
├── docs/                      # Documentation & study masterclass
│   ├── start.md               # Quick startup guide & command cheat sheet
│   ├── study.md               # Comprehensive from-scratch master textbook guide
│   ├── learning.md            # 6-week curriculum & viva prep roadmap
│   ├── info.md                # Technical specifications & project overview
│   └── roadmap.md             # Development architecture blueprint
├── src/                       # Core Python modular library
│   ├── data/                  # Dataset loader & image transformations
│   ├── models/                # PyTorch neural network modules
│   ├── training/              # Training loop, evaluation & device management
│   ├── inference/             # Model inference & Grad-CAM engine
│   └── utils/                 # Config loader, metrics & visualization
├── configs/                   # Configuration
│   └── config.yaml            # Hyperparameters, paths & model thresholds
├── data/                      # Dataset splits (NORMAL / PNEUMONIA)
├── notebooks/                 # Google Colab training & evaluation notebooks
├── scripts/                   # Data balancing, physical splitting & utility scripts
├── tests/                     # Automated Pytest suite
├── reports/                   # Performance charts & evaluation plots
├── app.py                     # Streamlit web application (Port 8501)
├── run_pipeline.py            # End-to-end training & evaluation pipeline
└── requirements.txt           # Python dependencies
```

---

## 🚀 Setup Instructions (Windows)

### Step 1 — Clone or download the project

```powershell
# If using Git
git clone <your-repo-url>
cd MedVision-AI

# Or just navigate to the project folder
cd "C:\Users\YourName\deep leaning final year project\MedVision-AI"
```

### Step 2 — Create a Virtual Environment

A virtual environment is an isolated Python installation for this project.  
It prevents conflicts with other Python packages on your computer.

```powershell
# Create the virtual environment (run once)
python -m venv venv

# Activate it (run every time you open a new terminal)
venv\Scripts\activate

# You should now see (venv) at the start of your prompt
```

### Step 3 — Install Dependencies

```powershell
pip install -r requirements.txt
```

> **GPU users (NVIDIA):** Replace the PyTorch lines with the CUDA build:
> ```powershell
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> pip install -r requirements.txt --ignore-installed torch torchvision
> ```

### Step 4 — Configure the Dataset Path

```powershell
# Copy the example .env file
Copy-Item .env.example .env

# Open .env in Notepad and set your dataset path:
notepad .env
```

Set `DATASET_ROOT` to the path of your extracted `chest_xray` folder, e.g.:
```
DATASET_ROOT=C:\Users\YourName\Downloads\chest_xray
```

### Step 5 — Download the Dataset

1. Go to [Kaggle — Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. Click **Download** (you need a free Kaggle account)
3. Extract the ZIP file  
4. Find the `chest_xray` folder (contains `train/`, `val/`, `test/`)
5. Set the path in your `.env` file

> **Important:** The dataset is ~1.2 GB. Do NOT commit it to Git.

---

## 🏋️ Training the Models

### Option A — Local Training (CPU only, slower)

```powershell
# Make sure (venv) is active
python run_pipeline.py
```

This runs all phases automatically:
- Phase 2: Dataset exploration + charts
- Phase 3: DataLoader setup
- Phase 4: Baseline CNN training
- Phase 5: ResNet-18 transfer learning + fine-tuning

### Option B — Google Colab (Recommended, free GPU)

1. Open `notebooks/MedVision_AI.ipynb` in Google Colab
2. Enable GPU: **Runtime → Change runtime type → T4 GPU**
3. Upload or mount your dataset
4. Run all cells in order
5. Download the saved `.pth` weight files
6. Place them in `artifacts/checkpoints/`

---

## 🌐 Running the Web App

```powershell
streamlit run app.py
```

Your browser will open at `http://localhost:8501`

**Features:**
- Upload a chest X-ray (JPEG/PNG)
- Select model: ResNet-18 or Baseline CNN
- View prediction + confidence scores
- Inspect Grad-CAM heatmap
- Download the heatmap image

---

## 🧪 Running Unit Tests

```powershell
pytest tests/ -v
```

Tests cover:
- Model output shapes
- Frozen/unfrozen parameters
- NaN detection
- Inference return values

---

## 📊 Metrics We Track

| Metric | What it measures |
|---|---|
| **Accuracy** | % of all predictions that are correct |
| **Precision** | Of all PNEUMONIA predictions, how many are actually PNEUMONIA |
| **Recall** | Of all actual PNEUMONIA cases, how many we correctly detected |
| **F1 Score** | Harmonic mean of Precision and Recall |
| **Specificity** | Of all NORMAL cases, how many we correctly labelled NORMAL |
| **AUC-ROC** | Overall discriminative ability (0.5 = random, 1.0 = perfect) |

---

## 🧠 Model Architecture

### Baseline CNN
- 3 convolutional blocks (each: Conv → BN → ReLU → MaxPool)
- Global Average Pooling
- FC(128→64) → FC(64→2)
- ~200k parameters

### ResNet-18 (Transfer Learning)
- Pretrained on ImageNet (1.28M images, 1000 classes)
- All backbone layers initially frozen
- Final layer replaced: Linear(512→2)
- Phase A: Train head only (~70k trainable params)
- Phase B: Unfreeze last 2 layer groups for fine-tuning

---

## 📚 Dataset

**Source:** [Kaggle — Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
**License:** CC BY 4.0  
**Citation:** Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), "Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification", Mendeley Data

Original split:
- Train: 5,216 images (1,341 NORMAL / 3,875 PNEUMONIA)
- Val:   16   images (8 NORMAL / 8 PNEUMONIA) ← too small
- Test:  624  images (234 NORMAL / 390 PNEUMONIA)

We create a new validation split (15% of training) for proper evaluation.

---

## 🔍 Grad-CAM Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) shows **which regions of the X-ray the model focused on**.

- Red/Yellow = high model attention
- Blue/Green = low model attention
- In PNEUMONIA cases, attention typically clusters in the **lower lung fields** where consolidation or opacification appears

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV 2017.

---

## 📈 Progress Log

| Phase | Status | Description |
|---|---|---|
| Phase 1 | ✅ Complete | Project structure + config |
| Phase 2 | ✅ Complete | Dataset exploration scripts |
| Phase 3 | ✅ Complete | Preprocessing transforms |
| Phase 4 | ⏳ Requires dataset | Baseline CNN training |
| Phase 5 | ⏳ Requires dataset | ResNet-18 transfer learning |
| Web App | ✅ Complete | Streamlit app (requires trained weights) |

---

## 🤝 Contributing / Academic Use

This project is developed as a final-year academic project.  
Do not plagiarise — understand every line of code before submitting.
