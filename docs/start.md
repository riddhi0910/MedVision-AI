# 🚀 MedVision AI — Project Startup Guide

Yeh guide MedVision AI ke dono frontends (Clinical PACS Workstation & Streamlit App) aur pipeline ko start karne ke liye simple step-by-step commands provide karta hai.

---

## ⚡ 1. Quick Start (Dono Apps Ek Saath)

### Option A: Clinical PACS Workstation (Port 8000)
> **Recommended**: Modern hospital UI jisme side-by-side raw X-ray, live Grad-CAM saliency map, zero-scroll single-screen view, Nagpur City Hospital branding, aur Dr. R. Somani ka formal PDF consultation report download hota hai.

```powershell
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 2. Start PACS Backend & Frontend Server
python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000 --reload
```
👉 Browser me open karein: **[http://localhost:8000](http://localhost:8000)**

---

### Option B: Streamlit Web Application (Port 8501)
> Original dark-themed medical application jisme file upload, threshold tuning, aur Grad-CAM inspection shamil hai.

```powershell
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 2. Run Streamlit App
streamlit run app.py
```
👉 Browser me open karein: **[http://localhost:8501](http://localhost:8501)**

---

## 🛠️ 2. Environment Setup (First-time Setup)

Agar aap kisi naye system par project clone kar rahe hain:

```powershell
# 1. Virtual Environment banayein
python -m venv venv

# 2. Virtual Environment activate karein
.\venv\Scripts\Activate.ps1

# 3. Dependencies install karein
pip install -r requirements.txt
```

---

## 🧠 3. ML Pipeline & Model Training

```powershell
# End-to-end pipeline run karein (data check, validation, evaluation)
python run_pipeline.py

# Sirf test suite run karne ke liye
pytest
```

---

## 🌐 4. Port Summary

| Service | Port | URL | Description |
| :--- | :--- | :--- | :--- |
| **PACS Diagnostic Workstation** | `8000` | `http://localhost:8000` | Nagpur City Hospital Full PACS Suite (FastAPI + JS) |
| **Streamlit Interface** | `8501` | `http://localhost:8501` | Streamlit Interactive Dashboard |
| **PACS Backend API Docs** | `8000` | `http://localhost:8000/docs` | Swagger / OpenAPI Interactive Documentation |

---

## 💡 Troubleshooting Tips

- **Port already in use:**
  Agar port 8000 ya 8501 busy dikhaye:
  ```powershell
  # Port 8000 check & kill
  netstat -ano | findstr :8000
  taskkill /PID <PID> /F

  # Port 8501 check & kill
  netstat -ano | findstr :8501
  taskkill /PID <PID> /F
  ```
- **Execution Policy Error in PowerShell:**
  Agar `Activate.ps1` execute na ho:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
