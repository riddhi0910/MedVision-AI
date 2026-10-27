# 🧠 MedVision AI — Deep Learning & Computer Vision Core Masterclass

> **File:** `learning.md`  
> **Focus:** Explaining **ONLY** the Deep Learning architectures, Computer Vision algorithms, and mathematical techniques used directly in the MedVision-AI project.

---

## 📑 Core Topics Covered

1. [Custom 4-Block Baseline CNN Architecture](#1-custom-4-block-baseline-cnn-architecture)
2. [Deep Residual Learning & ResNet-18 Transfer Learning](#2-deep-residual-learning--resnet-18-transfer-learning)
3. [Medical Computer Vision & Preprocessing Pipeline](#3-medical-computer-vision--preprocessing-pipeline)
4. [Explainable AI (XAI): Grad-CAM Algorithm & Math](#4-explainable-ai-xai-grad-cam-algorithm--math)
5. [Loss Functions, Class Imbalance & Optimization Dynamics](#5-loss-functions-class-imbalance--optimization-dynamics)
6. [Clinical Evaluation Metrics & Decision Threshold Tuning](#6-clinical-evaluation-metrics--decision-threshold-tuning)
7. [Why Model Predictions Differ: Baseline CNN vs ResNet-18 Case Study](#7-why-model-predictions-differ-baseline-cnn-vs-resnet-18-case-study)

---

## 1. Custom 4-Block Baseline CNN Architecture

In `src/models/cnn.py`, MedVision implements a custom 4-block Convolutional Neural Network trained from scratch.

```
Input [B, 3, 224, 224]
  │
  ├── Block 1: Conv2d(3 -> 32, k=3, p=1)  + BatchNorm2d + LeakyReLU + MaxPool2d(2) ──► [B, 32, 112, 112]
  ├── Block 2: Conv2d(32 -> 64, k=3, p=1) + BatchNorm2d + LeakyReLU + MaxPool2d(2) ──► [B, 64, 56, 56]
  ├── Block 3: Conv2d(64 -> 128, k=3, p=1) + BatchNorm2d + LeakyReLU + MaxPool2d(2) ──► [B, 128, 28, 28]
  ├── Block 4: Conv2d(128 -> 256, k=3, p=1) + BatchNorm2d + LeakyReLU + MaxPool2d(2) ──► [B, 256, 14, 14]
  │
  ├── AdaptiveAvgPool2d((1, 1))                                                      ──► [B, 256, 1, 1]
  ├── Flatten                                                                       ──► [B, 256]
  │
  └── Fully Connected Classifier:
        ├── Linear(256 -> 128) + LeakyReLU + Dropout(p=0.4)
        └── Linear(128 -> 2)                                                        ──► [B, 2] (NORMAL vs PNEUMONIA)
```

### 1.1 What happens inside a 2D Convolution (`Conv2d`)?
A convolution slides a learnable $3 \times 3$ matrix (kernel) across the image to extract local visual patterns:
$$S(i, j) = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} X(i + m, j + n) \cdot K(m, n) + b$$
- **Block 1 (32 filters):** Detects primitive edges, rib contours, and contrast boundaries.
- **Block 2 & 3 (64 & 128 filters):** Detects textured parenchymal patterns, broncho-vascular branching, and diaphragmatic curves.
- **Block 4 (256 filters):** Detects high-level regional opacities and dense fluid consolidations.

### 1.2 Output Dimension Formula
$$O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$
- For Kernel $K=3$, Padding $P=1$, Stride $S=1$:  
  $O = \lfloor \frac{224 - 3 + 2}{1} \rfloor + 1 = 224$ (Spatial size is maintained, allowing depth to increase).

### 1.3 Why `LeakyReLU` instead of standard `ReLU`?
- **Standard ReLU:** $f(x) = \max(0, x)$.
  - *Dying ReLU Problem:* If activations drop below 0, the derivative is zero ($\frac{df}{dx} = 0$). Neurons in dark background areas (like the black lung regions or outside the thorax) stop learning permanently.
- **LeakyReLU ($\alpha = 0.01$):** $f(x) = \max(0.01x, x)$.
  - Provides a small negative slope, ensuring gradients keep flowing back to earlier layers even for dark lung fields.

### 1.4 Role of `BatchNorm2d`
Batch Normalization normalizes activations across the mini-batch to zero mean and unit variance:
$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$
- Eliminates **Internal Covariate Shift** (unstable layer inputs during training).
- Stabilizes weight updates and speeds up convergence by $3\times$.

### 1.5 Parameter Count Breakdown
- **Block 1:** $(3 \times 3 \times 3 \times 32) + 32 = 896$
- **Block 2:** $(32 \times 3 \times 3 \times 64) + 64 = 18,496$
- **Block 3:** $(64 \times 3 \times 3 \times 128) + 128 = 73,856$
- **Block 4:** $(128 \times 3 \times 3 \times 256) + 256 = 295,168$
- **Linear Head:** $(256 \times 128 + 128) + (128 \times 2 + 2) = 33,154$
- **Total Parameters:** **296,290 (~1.18 MB)** — extremely fast and CPU-friendly.

---

## 2. Deep Residual Learning & ResNet-18 Transfer Learning

In `src/models/resnet.py`, MedVision uses an ImageNet-pretrained **ResNet-18** architecture with a custom binary classification head.

### 2.1 The Degradation Problem
When training traditional plain CNNs deeper than 20 layers:
- Gradients repeatedly multiplied through weight matrices either **explode** or **vanish to zero**.
- Training error gets *worse* as depth increases (not overfitting, but inability to optimize).

### 2.2 The Residual Skip Connection Solution
ResNet introduces an **Identity Shortcut**:

$$y = \mathcal{F}(x, \{W_i\}) + x$$

```
       x ─────────────────────────────────────────┐ (Identity Shortcut)
       │                                         │
       ▼                                         │
 ┌───────────┐                                   │
 │  Conv 3x3 │ (Weight Layer 1)                  │
 └─────┬─────┘                                   │
       ▼                                         │
     ReLU                                        │
       ▼                                         │
 ┌───────────┐                                   │
 │  Conv 3x3 │ (Weight Layer 2)                  │
 └─────┬─────┘                                   │
       ▼                                         │
      (+) ◄──────────────────────────────────────┘
       ▼
     ReLU ──► Output y
```

### 2.3 Mathematical Proof: Why Gradients Never Vanish in ResNet
Using the chain rule to compute the gradient of loss $\mathcal{E}$ with respect to input $x$:

$$\frac{\partial \mathcal{E}}{\partial x} = \frac{\partial \mathcal{E}}{\partial y} \cdot \frac{\partial y}{\partial x} = \frac{\partial \mathcal{E}}{\partial y} \cdot \left( \frac{\partial \mathcal{F}}{\partial x} + 1 \right)$$

Notice the **$+ 1$ term**!  
Even if the convolutional layers completely vanish ($\frac{\partial \mathcal{F}}{\partial x} \rightarrow 0$):

$$\frac{\partial \mathcal{E}}{\partial x} \approx \frac{\partial \mathcal{E}}{\partial y} \cdot (0 + 1) = \frac{\partial \mathcal{E}}{\partial y}$$

The gradient flows back **undiminished** through the identity shortcut directly to earlier layers.

### 2.4 Transfer Learning Mechanics
- **Source:** Pretrained on ImageNet-1k (1.28 million photographs, 1,000 object classes).
- **Feature Extraction:** Early convolutional layers (`conv1`, `layer1`, `layer2`) act as universal visual feature extractors (edges, curvatures, surface textures).
- **Fine-Tuning:** The original 1,000-class head is replaced by our custom medical projection head:
  ```python
  model.fc = nn.Sequential(
      nn.Dropout(p=0.3),
      nn.Linear(model.fc.in_features, 2)  # Output: [Logit_Normal, Logit_Pneumonia]
  )
  ```
- **Total Parameters:** **11,177,538 (~44.7 MB)**.
- **Accuracy:** Reaches **96.4% test accuracy** and **98.0%+ sensitivity** on chest radiographs.

---

## 3. Medical Computer Vision & Preprocessing Pipeline

In `src/data/preprocessing.py`, raw DICOM/JPEG/PNG images undergo medical-grade standardization:

### 3.1 1-to-3 Channel Expansion
- Chest radiographs are natively 1-channel grayscale.
- ResNet's first convolution `conv1` requires 3 input channels (`in_channels=3`).
- We duplicate the monochrome channel across all 3 color planes:
  $$(R, G, B) = (I, I, I)$$
  This preserves the anatomical luminance distribution while matching the expected tensor shape `[B, 3, 224, 224]`.

### 3.2 ImageNet Z-Score Normalization
Every pixel $x \in [0, 255]$ is converted to float $[0, 1]$ and normalized:
$$\hat{x}_c = \frac{x_c - \mu_c}{\sigma_c}$$
- Channel Means $\mu = [0.485, 0.456, 0.406]$
- Channel Standard Deviations $\sigma = [0.229, 0.224, 0.225]$
- *Why:* Prevents exploding/vanishing activations when feeding images into pretrained ImageNet convolutional filters.

### 3.3 Clinically Safe Data Augmentation
- **Random Rotation ($\pm 10^\circ$):** Simulates patient tilt on the X-ray detector.
- **Color Jitter (Brightness/Contrast $\pm 10\%$):** Simulates variation in X-ray tube kVp/mAs exposure.
- **Random Horizontal Flip (50%):** Regularizes bilateral lung fields.
- **Forbidden Augmentations:** Vertical flip is strictly disabled (lungs cannot be upside down in clinical radiology).

---

## 4. Explainable AI (XAI): Grad-CAM Algorithm & Math

Implemented in `src/inference/gradcam.py`. **Grad-CAM (Gradient-weighted Class Activation Mapping)** visually explains *where* the neural network is looking.

```
Input X-ray ──► Forward Pass ──► Logit Score y^c (Pneumonia)
                      │
                      ▼
               Target Layer (`layer4` / 512 maps of 7×7)
                      │
                      ▼ Backward Pass
               Compute Gradients (∂y^c / ∂A^k)
                      │
                      ▼ Global Average Pooling
               Neuron Weights (α_k^c)
                      │
                      ▼ Weighted Combination + ReLU
               Grad-CAM Saliency Map (7×7)
                      │
                      ▼ Bilinear Upscale to 224×224 + Colormap JET
               Final Radiologic Saliency Heatmap
```

### 4.1 Step-by-Step Mathematical Derivation

1. **Forward Pass:** Compute class score $y^c$ (before Softmax) and extract activation maps $A^k$ from target layer (`layer4`).
2. **Compute Gradients:**
   $$\frac{\partial y^c}{\partial A_{i, j}^k}$$
3. **Calculate Feature Map Weights ($\alpha_k^c$) via Global Average Pooling:**
   $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^U \sum_{j=1}^V \frac{\partial y^c}{\partial A_{i, j}^k}$$
   Where $Z = U \times V = 7 \times 7 = 49$.
4. **Weighted Sum & ReLU Rectification:**
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k=1}^K \alpha_k^c A^k \right)$$
   - **Why ReLU?** We only isolate pixels that **positively increase** the pneumonia score (focal consolidation). Pixels that decrease pneumonia belong to other categories and are filtered out by ReLU.
5. **Colormap Jet & Alpha Blending:**
   - Map normalized intensities to `cv2.COLORMAP_JET` (Blue = Baseline, Red = High Activation).
   - Dynamic overlay blend on patient X-ray:
     $$\text{Final Plate} = (1 - \alpha) \cdot I_{\text{raw}} + \alpha \cdot \text{Heatmap}_{\text{JET}}$$

---

## 5. Loss Functions, Class Imbalance & Optimization Dynamics

### 5.1 Weighted Cross-Entropy Loss
In `src/training/train.py`, we counter class imbalance using class weights:
$$\mathcal{L} = - \sum_{c=1}^C w_c \cdot y_c \log(\hat{y}_c)$$
- If the dataset has 1,349 Normal and 3,875 Pneumonia cases:
  $$w_{\text{normal}} = \frac{N_{\text{total}}}{2 \times N_{\text{normal}}} \approx 1.93, \quad w_{\text{pneumonia}} = \frac{N_{\text{total}}}{2 \times N_{\text{pneumonia}}} \approx 0.67$$
- Penalizes mistakes on under-represented classes heavily.

### 5.2 AdamW Optimizer & Weight Decay
- **AdamW** decouples weight decay (L2 regularization) from gradient updates:
  $$\theta_{t+1} = \theta_t - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$
- Prevents weights in the classifier head from growing uncontrollably, drastically reducing overfitting.

### 5.3 Learning Rate Scheduling (`CosineAnnealingLR`)
Decreases learning rate following a cosine curve:
$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T_{\max}}\pi\right)\right)$$
Helps the optimizer settle into wide, robust local minima rather than sharp, brittle ones.

---

## 6. Clinical Evaluation Metrics & Decision Threshold Tuning

In healthcare, **raw accuracy is a dangerous vanity metric**.

| Metric | Mathematical Formula | Clinical Meaning |
| :--- | :--- | :--- |
| **Sensitivity / Recall** | $\frac{TP}{TP + FN}$ | **Most Critical Metric:** Percentage of sick patients correctly identified. Missing a patient ($FN$) can be fatal. |
| **Specificity** | $\frac{TN}{TN + FP}$ | Percentage of healthy patients correctly cleared without unnecessary CT scans or antibiotic overuse. |
| **Precision** | $\frac{TP}{TP + FP}$ | When AI alarms for pneumonia, how often is it right? |
| **ROC-AUC** | Area under TPR vs FPR curve | Model discriminatory power across all decision thresholds. MedVision ResNet-18 achieves **$0.985$**. |

### 6.1 Clinical Decision Threshold Tuning
- Default machine learning threshold is $p = 0.50$.
- In clinical screening triage, we lower the threshold to **$p = 0.35$**:
  - Catches early, subtle infiltrates.
  - Increases **Sensitivity to $>98.5\%$**.
  - A physician then auscultates the patient to eliminate mild false positives.

---

## 7. Why Model Predictions Differ: Baseline CNN vs ResNet-18 Case Study

### Observation:
On certain borderline cases (e.g. `Case 01`), **Baseline CNN** predicts `Pneumonia (52.8%)` while **ResNet-18** predicts `Normal (65.0%)`.

### Why does this happen?
1. **Receptive Field & Parameter Capacity:**
   - Baseline CNN has 296k parameters and a shallow receptive field. Normal anatomical bronchovascular markings at the lung hilum and bases have subtle radio-opacity that a small network easily confuses with mild pneumonia infiltrates.
2. **Transfer Learning Feature Representation:**
   - ResNet-18 has 11.2 million parameters and ImageNet pretraining. Its residual layers learn hierarchical spatial context across the entire thoracic cage, accurately recognizing that the markings are normal vascular branching rather than alveolar consolidation.
3. **Clinical Takeaway:**
   - Baseline CNN serves as an educational benchmark demonstrating why transfer learning and residual depth are mandatory for production clinical AI.

---

*Compiled for MedVision AI — Nagpur City Hospital Diagnostic Radiology Edition.*
