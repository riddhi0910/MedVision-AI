# 🚀 MedVision AI — Comprehensive Startup & Setup Guide

Yeh complete step-by-step documentation hai jisme **Environment Setup (venv & requirements)**, **Application Startup (PACS Workstation & Streamlit)**, **NPM Scripts**, aur **Git Push Instructions** detail me diye gaye hain.

---

## 📋 1. Prerequisites (Zaroori Cheezein)

- **Python**: Version `3.10` ya `3.11` recommended (check with `python --version`)
- **Git**: Code version control ke liye (check with `git --version`)
- *(Optional)* **Node.js & npm**: Agar aap `npm start` ya `npm run dev` use karna chahein. *(Frontend static HTML/JS me build hai jo FastAPI backend hi serve karta hai, isliye Node.js mandatory nahi hai).*

---

## 🛠️ 2. First-Time Setup (Virtual Environment & Dependencies)

Agar aap kisi naye laptop/PC par repository clone kar rahe hain ya fresh setup kar rahe hain:

### Step 1: Clone Repository
```bash
git clone https://github.com/<your-username>/MedVision-AI.git
cd MedVision-AI
```

### Step 2: Virtual Environment Banayein
Virtual environment banane se aapke system ke global Python packages safe rehte hain:

```powershell
# Windows (PowerShell / CMD):
python -m venv venv
```
*(macOS / Linux users: `python3 -m venv venv`)*

### Step 3: Virtual Environment Activate Karein

- **Windows PowerShell:**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(💡 Agar execution policy error aaye, toh terminal me run karein: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

- **Windows Command Prompt (CMD):**
  ```cmd
  .\venv\Scripts\activate.bat
  ```

- **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

Jab environment activate ho jayega, aapke terminal prompt ke aage `(venv)` likha aayega.

### Step 4: Dependencies Install Karein
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Demo Model Weights & Test Images Generate Karein
Agar aapke paas full dataset train karne ka time nahi hai aur aap turant app test karna chahte hain:
```powershell
python scripts/create_demo_checkpoints.py
```
Yeh command `artifacts/checkpoints/best_resnet18.pth`, `artifacts/checkpoints/best_cnn.pth`, aur test X-ray images create kar degi.

---

## ⚡ 3. Application Start Kaise Karein

Aapke paas application run karne ke **2 simple tarike** hain:

### Option A: Clinical PACS Workstation (Port 8000) — *Recommended*
> **Modern Hospital UI**: Side-by-side chest radiograph, live Grad-CAM visual heatmap, 1-screen zero-scroll layout, Nagpur City Hospital branding, aur Dr. R. Somani ka formal PDF consultation report download.

#### Tarika 1: Direct Python Command (Har system par chalta hai, No Node required)
```powershell
# Virtual environment activate rakhein
python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000 --reload
```

#### Tarika 2: NPM Command (Agar aapke system par Node.js/npm installed hai)
```powershell
# Dev mode me run karne ke liye
npm run dev

# Ya standard start:
npm start
```
👉 Browser me open karein: **[http://localhost:8000](http://localhost:8000)**  
👉 Interactive API Docs (Swagger): **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

### Option B: Streamlit Web Application (Port 8501)
> Original medical explorer dashboard jisme model comparison, interactive confidence threshold tuning, aur raw Grad-CAM visualization shamil hai.

#### Tarika 1: Direct Python Command
```powershell
# Virtual environment activate rakhein
streamlit run app.py
```

#### Tarika 2: NPM Script
```powershell
npm run streamlit
```
👉 Browser me open karein: **[http://localhost:8501](http://localhost:8501)**

---

## 📦 4. NPM Scripts Reference

Agar Node.js installed hai, toh root me maujood `package.json` se aap yeh shortcuts chala sakte hain:

| Command | Action |
| :--- | :--- |
| `npm start` | PACS Workstation start karta hai (`port 8000`) |
| `npm run dev` | PACS Workstation live auto-reload ke sath run karta hai |
| `npm run streamlit` | Streamlit Application start karta hai (`port 8501`) |
| `npm run demo` | Checkpoints & test X-rays generate karta hai |
| `npm run pipeline` | End-to-end ML training & evaluation pipeline chalata hai |
| `npm test` | PyTest automated test suite execute karta hai |

---

## 🧪 5. Testing & ML Pipeline

```powershell
# Automated unit & integration tests run karein:
pytest

# Full end-to-end training, validation & evaluation pipeline chalayein:
python run_pipeline.py
```

---

## 🔒 6. Git Push Checklist (Safe & Clean Push to GitHub)

Humne `.gitignore` ko perfectly configure kar diya hai taaki koi bhi unwanted ya huge files GitHub par push na hon:

### Kya automatically ignore hoga (Git ignore list):
- ✅ `venv/`, `.venv/` (Virtual environment folder)
- ✅ `data/`, `data.zip`, `*.zip`, `*.tar.gz` (Kaggle dataset & archives > 2.5 GB — GitHub limit 100MB hoti hai)
- ✅ `*.pth`, `*.pt` (Model weight binaries > 40MB)
- ✅ `__pycache__/`, `*.pyc` (Python compile cache)
- ✅ `node_modules/` (Node modules if any)
- ✅ `.env` (Secret credentials)
- ✅ `.pytest_cache/`, `reports/*.png`

### Safe Git Push Commands:
```powershell
# 1. Check karein kya stage hone wala hai (data.zip ya venv list me NAHI hona chahiye):
git status

# 2. Saare safe changes add karein:
git add .

# 3. Dobara status check karein (Sab green files code aur docs hone chahiye):
git status

# 4. Commit banayein:
git commit -m "feat: complete MedVision AI PACS suite, documentation, and startup config"

# 5. Remote GitHub repository par push karein:
git push origin main
```

---

## 💡 7. Troubleshooting & FAQ

### Q1: `npm start` error de raha hai: "'node' is not recognized"?
**Karan:** Aapke machine par Node.js installed nahi hai ya PATH me nahi hai.  
**Solution:** Aapko Node.js ki zaroorat hi nahi hai! Direct Python command use karein:  
`python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000 --reload`

### Q2: Port 8000 ya 8501 already in use (Address already in use)?
Agar pichla process background me band nahi hua:
```powershell
# Port 8000 free karein:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Port 8501 free karein:
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

### Q3: `best_resnet18.pth` not found error?
Agar model weight missing ho:
```powershell
python scripts/create_demo_checkpoints.py
```
Yeh script turant working model checkpoints bana deti hai.

### Q4: PowerShell me `Activate.ps1 cannot be loaded` error?
Run karein:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Iske baad dobara `.\venv\Scripts\Activate.ps1` chalayein.
