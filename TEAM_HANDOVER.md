# HarmonyShield — Complete Team Handover & Emergency Recovery Guide

> [!IMPORTANT]
> **Emergency Recovery Notice:** If the primary project laptop becomes unavailable tomorrow, any teammate can follow this guide to set up, launch, and demonstrate HarmonyShield independently on any Windows, macOS, or Linux laptop.

---

## 1. Quick Emergency Recovery Checklist (60-Second Overview)

If you need to get the application running immediately for a viva or demo:

```bash
# 1. Clone the verified review-ready release
git clone -b review-ready https://github.com/ParamBande/HarmonyShiled.git
cd HarmonyShiled

# 2. Set up Python virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Download / Verify model checkpoints (Takes ~1-2 mins)
python scripts/download_checkpoints.py

# 5. Start Backend (Terminal 1)
python harmonyshield.py --backend
# Alternatively: uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 6. Start Frontend (Terminal 2)
cd frontend
npm install
npm run dev
```

Open your browser to: **`http://localhost:5173`** (or port displayed by Vite).

---

## 2. System Architecture & Prerequisites

### Hardware Requirements
- **RAM:** Minimum 8 GB (16 GB recommended).
- **GPU:** Optional. HarmonyShield runs seamlessly on **CPU-only laptops**. If an NVIDIA GPU with CUDA is present, it will automatically accelerate ResNet18 and AST inference.
- **Disk Space:** ~2 GB free disk space (for Python packages, checkpoints, and AST cache).

### Software Requirements
- **Python:** 3.10, 3.11, 3.12, or 3.13.
- **Node.js:** 18.x or later (includes `npm`).
- **Git:** Standard Git client.

---

## 3. Step-by-Step Setup Instructions

### Step 3.1: Clone the Repository
```bash
git clone -b review-ready https://github.com/ParamBande/HarmonyShiled.git
cd HarmonyShiled
```

### Step 3.2: Configure Python Virtual Environment
```bash
# Create venv
python -m venv venv

# Activate venv:
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.\venv\Scripts\activate.bat
# macOS / Linux:
source venv/bin/activate

# Upgrade pip and install wheel
python -m pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

### Step 3.3: Obtain & Verify Model Checkpoints

HarmonyShield uses three forensic detection approaches. Here is how their model weights are obtained:

| Method | Artifact Name | Source / Handling | Destination Path |
| :--- | :--- | :--- | :--- |
| **Method 1: Classical RF** | `best_rf_model.joblib` (364 KB) | **Included in Git** | `data/pilot_dataset/best_rf_model.joblib` |
| **Method 2: ResNet-18 Ensemble** | 5 `.pt` checkpoints (42.7 MB each) | **GitHub Release Assets** or **Team Drive** | `experiments/resnet18_generalization/checkpoints/` |
| **Method 3: AST Detector** | `ai_music_detection_large_60s` (~340 MB) | **Hugging Face Hub** (Automatic download on 1st run) | `~/.cache/huggingface/hub/` |

#### Automatic Checkpoint Download:
Run the provided automated downloader:
```bash
python scripts/download_checkpoints.py
```
This script downloads all 5 ResNet18 checkpoints directly from GitHub Release `v1.0.0-review-ready` and automatically verifies their SHA-256 hashes.

#### Manual Checkpoint Placement (Fallback):
If downloading manually from the team Google Drive or release page, place the 5 `.pt` files directly into:
`experiments/resnet18_generalization/checkpoints/`
- `LOGO_audioldm2_model.pt`
- `LOGO_MusicGen_medium_model.pt`
- `LOGO_musicldm_model.pt`
- `LOGO_mustango_model.pt`
- `LOGO_stable_audio_open_model.pt`

#### Verify Checksum Integrity:
```bash
python scripts/verify_models.py
```
Expected output:
```text
>>> ALL MODEL ARTIFACTS VERIFIED: PASS <<<
```

### Step 3.4: Configure the Frontend
In a separate terminal:
```bash
cd frontend
npm install
```

---

## 4. Launching the Application

### Option A: Dedicated Terminals (Recommended for Review)
1. **Terminal 1 (Backend):**
   ```bash
   .\venv\Scripts\activate
   uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   *Verify health:* Navigate to `http://localhost:8000/health` in your browser. It should respond:
   `{"status":"ok","device":"cpu" (or "cuda"),"models_loaded":[...]}`

2. **Terminal 2 (Frontend):**
   ```bash
   cd frontend
   npm run dev
   ```
   *Open UI:* `http://localhost:5173`

### Option B: Unified CLI Launcher
From the repository root:
```bash
python harmonyshield.py --backend
# In another terminal:
python harmonyshield.py --frontend
```

---

## 5. Demonstrating HarmonyShield During Project Review

### Live Demo Flow:
1. **Open the Dashboard:** Go to `http://localhost:5173`.
2. **Upload an Audio File:**
   - Supported formats: `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`.
   - You can use any song file, music clip, or generated audio from Suno, Udio, AudioLDM2, etc.
3. **Inspect the Forensic Analysis:**
   - **Mel-Spectrogram:** View the frequency representation showing time-frequency energy distribution.
   - **Feature Importance:** Classical model transparency chart highlighting top spectral features (e.g., Spectral Rolloff, Spectral Contrast Band 1).
   - **Three Individual Model Predictions:**
     - *Classical Random Forest:* Spectral feature classifier (first 10s).
     - *Deep Learning ResNet-18:* Spectrogram CNN ensemble (first 10s).
     - *External AST Transformer:* Audio Spectrogram Transformer evaluating 60s windows across the file.
4. **Majority-Voting Consensus Card:**
   - The final banner displays one of the 5 verified consensus states:
     - **AI-GENERATED** (3 AI votes)
     - **LIKELY AI-GENERATED** (2 AI votes, 1 Human vote)
     - **LIKELY HUMAN-CREATED** (2 Human votes, 1 AI vote)
     - **HUMAN-CREATED** (3 Human votes)
     - **UNABLE TO DETERMINE** (< 2 valid model outputs)
   - Diagnostic metrics: Mean Probability, Model Variance, and Disagreement badge.

### Navigating the UI:
- **Analyze (`/`):** Interactive forensics workspace.
- **Research (`/research`):** Complete experimental findings, Pilot 01 vs. Deployed model distinction, and LOGO generalization tables.
- **How It Works (`/how-it-works`):** Technical methodology and pipeline flowchart.

---

## 6. Running the Automated Test Suite

To verify system integrity in under 5 seconds:
```bash
.\venv\Scripts\activate
pytest tests/
```
Expected output:
```text
============================= 18 passed in ~1.5s =============================
tests/test_consensus.py ......          [ 33%] (Majority voting consensus tests)
tests/test_plan_a.py .........          [ 83%] (Preprocessing & feature extraction)
tests/test_predict.py ...               [100%] (FastAPI backend endpoints)
```

---

## 7. Common Troubleshooting & FAQs

### Q1: `uvicorn` fails with `ModuleNotFoundError`
**Fix:** Ensure your virtual environment is activated before starting the backend (`.\venv\Scripts\activate`).

### Q2: AST detector is slow on first launch
**Explanation:** On the first run, the `transformers` library downloads `AI-Music-Detection/ai_music_detection_large_60s` (~340 MB) from Hugging Face Hub. Once downloaded, it is permanently cached locally in `~/.cache/huggingface/hub/` and loads instantly on subsequent runs.

### Q3: Audio upload fails with `Cannot read audio` or `audioread error`
**Fix:** On Windows, `soundfile` handles `.wav` and `.flac`. For `.mp3` loading, `ffmpeg` or `imageio-ffmpeg` is utilized. Ensure you are uploading a valid audio file.

### Q4: One model displays "Offline" or "Inference Error"
**Fix:** Run `python scripts/verify_models.py`. If ResNet18 checkpoints are missing, run `python scripts/download_checkpoints.py`. If only 2 models succeed, the majority-voting engine will still safely compute consensus using the available valid models.

---

## 8. Team Contact & Git Pointers
- **GitHub Repository:** [https://github.com/ParamBande/HarmonyShiled](https://github.com/ParamBande/HarmonyShiled)
- **Release Branch:** `review-ready`
- **Release Tag:** `v1.0.0-review-ready`
- **Primary Maintainer:** Param Bande (`parambande2812@gmail.com`)
