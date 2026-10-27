# 🧭 MedVision AI — Master Project Roadmap & Architectural Guide
> **Explainable Chest X-Ray Classification Using Deep Learning**  
> *Final Year B.Tech Computer Science / AI & Data Science Monolithic Project*

---

## 📌 Executive Summary
MedVision AI is an end-to-end, medical-grade deep learning system that classifies pediatric chest radiographs into **NORMAL** and **PNEUMONIA** while providing **Grad-CAM visual heatmaps** to explain its diagnostic decisions to physicians.

The project is structured as a clean, modular **Monolith** with strict folder separation:
- `frontend/public/` ➡️ PACS workstation UI (HTML/CSS/JS), served by `backend/server.py`
- `src/` ➡️ Data processing, model architectures (CNN, ResNet), training, inference & Grad-CAM
- `artifacts/checkpoints/` ➡️ Trained weights (`.pth`)
- `app.py` ➡️ Streamlit research explorer

---

## 📂 1. Clean Monolithic Folder Structure

See **[README → Repository Structure](../README.md#-repository-structure)** for the current folder layout (single source of truth).

---

## 🫁 2. Dataset Specifications & Clinical Background

| Parameter | Value / Description |
| :--- | :--- |
| **Dataset Source** | Kaggle: *Chest X-Ray Images (Pneumonia)* by Kermany et al. |
| **Medical Institution** | Guangzhou Women and Children’s Medical Center |
| **Total Images** | **5,863 chest radiograms** (JPEG format) |
| **Patient Cohort** | Pediatric patients aged 1 to 5 years old (AP position) |
| **Classes** | `NORMAL` (Healthy lungs) vs. `PNEUMONIA` (Bacterial & Viral infection) |
| **Dataset Size** | ~1.15 GB uncompressed |

### ⚠️ Handling Class Imbalance
In clinical data, positive cases often outnumber negative cases:
- **Normal**: ~1,583 images (~27%)
- **Pneumonia**: ~4,273 images (~73%)

**Our Mitigation Strategy:**
1. **Weighted Loss**: We compute inverse class frequency weights and pass them to `CrossEntropyLoss`, penalizing false negatives heavily.
2. **Data Augmentation**: Normal images undergo random horizontal flips, subtle rotations (±10°), and contrast adjustments to enrich feature diversity.
3. **Medical Metric**: We optimize for **Sensitivity/Recall** ($\frac{TP}{TP + FN}$), ensuring infected patients are not missed.

---

## 🧠 3. Model Architectures & Weights

### A. Pretrained ResNet-18 Transfer Learning (`src/models/resnet.py`)
- **Backbone**: ResNet-18 pretrained on 1.4 million ImageNet images.
- **Skip Connections (Residual Blocks)**: Solves the vanishing gradient problem via $\mathcal{H}(x) = \mathcal{F}(x) + x$.
- **Custom Classifier Head**: Replaces original 1000-class FC layer with:
  `Linear(512 -> 256) -> BatchNorm1d -> LeakyReLU(0.1) -> Dropout(0.4) -> Linear(256 -> 2)`
- **Saved Weights**: `artifacts/checkpoints/best_resnet18.pth` (~44.7 MB).

### B. Custom Baseline CNN (`src/models/cnn.py`)
- **Architecture**: 4 Sequential Convolutional blocks built from scratch:
  - Block 1: `Conv2d(3 -> 32, 3x3) -> BatchNorm2d -> LeakyReLU -> MaxPool2d`
  - Block 2: `Conv2d(32 -> 64, 3x3) -> BatchNorm2d -> LeakyReLU -> MaxPool2d`
  - Block 3: `Conv2d(64 -> 128, 3x3) -> BatchNorm2d -> LeakyReLU -> MaxPool2d`
  - Block 4: `Conv2d(128 -> 256, 3x3) -> BatchNorm2d -> LeakyReLU -> MaxPool2d`
  - Global Average Pooling -> `Linear(256 -> 64) -> Dropout(0.5) -> Linear(64 -> 2)`.
- **Saved Weights**: `artifacts/checkpoints/best_cnn.pth` (~1.2 MB).

---

## 🔍 4. Explainable AI: Grad-CAM (`src/inference/gradcam.py`)

### The "Black Box" Problem
Traditional AI gives predictions without explanation. Doctors cannot accept a diagnosis without understanding *why*.

### How Grad-CAM Works (Native PyTorch Implementation)
1. **Forward Hook**: Captures feature activations $A^k$ at the final convolutional block (`model.backbone.layer4[-1]`).
2. **Backward Hook**: Calculates gradients $\frac{\partial y^c}{\partial A^k}$ of the target diagnostic class score $y^c$.
3. **Importance Weight ($\alpha_k^c$)**: Global average pooling across gradients:
   $$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}$$
4. **Weighted Combination & ReLU**:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
5. **Bilinear Upsampling**: Stretches the heatmap to $224 \times 224$ and blends it over the patient's X-ray in Jet colormap.

---

## 🚀 5. How to Run the Project (Commands Cheatsheet)

### Step 1: Activate Virtual Environment
Open PowerShell in your project folder:
```powershell
.\MedVision-AI\venv\Scripts\Activate.ps1
```

### Step 2: Run the Streamlit Web Application
Run either from the root or inside `MedVision-AI`:
```powershell
# From root directory:
streamlit run app.py

# OR from MedVision-AI directory:
cd MedVision-AI
streamlit run app.py
```
> The dashboard will automatically open at `http://localhost:8501`.

### Step 3: Run Automated Unit Tests
```powershell
python -m pytest tests/test_model.py tests/test_inference.py -v
```

### Step 4: Training on Google Colab (Free GPU)
1. Open [Google Colab](https://colab.research.google.com).
2. Upload `MedVision-AI/notebooks/MedVision_AI.ipynb`.
3. Set Runtime to **GPU (T4)**.
4. Run all cells (~10 minutes).
5. Download `best_resnet18.pth` and place it in `MedVision-AI/artifacts/checkpoints/`.

---

## 🎓 6. Top Viva / Defense Questions & Model Answers

### Q1: Why did you choose a monolithic architecture over microservices?
> **Answer**: For a medical diagnostic workstation, a clean modular monolith minimizes deployment latency, prevents network failure points between client and server, eliminates container orchestration overhead, and keeps image processing directly connected to the GPU tensor pipeline.

### Q2: Why is Grad-CAM implemented using native PyTorch hooks rather than an external library?
> **Answer**: External wrappers often introduce version incompatibilities and add black-box layers. Writing forward and backward hooks directly via PyTorch gives us exact control over feature tensor extraction, avoids redundant CUDA context switches, and allows us to verify mathematical gradients directly from the loss graph.

### Q3: Why is ResNet-18 superior to a simple CNN for this task?
> **Answer**: ResNet-18 features residual skip connections ($\mathcal{H}(x) = \mathcal{F}(x) + x$) that eliminate the vanishing gradient problem, allowing the network to train effectively across deep layers. Furthermore, pre-training on ImageNet gives it powerful general visual filters (edge detection, textural gradients) that transfer directly to radiological features.

### Q4: Why is Accuracy a dangerous metric in medical AI?
> **Answer**: If a dataset has 90% positive cases, a naive model that predicts positive for every patient will achieve 90% accuracy while failing completely on healthy patients. In clinical diagnostics, **Recall/Sensitivity** ($\frac{TP}{TP + FN}$) is the primary safety metric because a False Negative (missing a sick patient) can be fatal.
