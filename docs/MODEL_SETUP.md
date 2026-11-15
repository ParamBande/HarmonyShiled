# Model Architecture & Checkpoint Setup Guide

This guide details the three detection architectures used in **HarmonyShield**, their artifact locations, and how to execute or reproduce them.

---

## 1. Overview of Detection Methods

HarmonyShield implements a tripartite forensic evaluation framework:

```
                      Input Audio Track (MP3, WAV, FLAC, M4A)
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          ▼
    Method 1: Classical ML      Method 2: ResNet18        Method 3: Plan B AST
    (Spectral + Random Forest)  (Mel-Spectrogram CNN)     (Pretrained Transformer)
             │                          │                          │
             ▼                          ▼                          ▼
      AI Probability             AI Probability             AI Probability
             └──────────────────────────┬──────────────────────────┘
                                        ▼
                     Consensus & Uncertainty Analysis
                     (Strong / Mixed / Inconclusive)
```

---

## 2. Method 1: Classical ML (Feature Set B + Random Forest)

- **Input:** 10-second audio segment at 16 kHz mono.
- **Features Extracted:** Feature Set B (22 spectral features):
  - Spectral Centroid (mean, std)
  - Spectral Bandwidth (mean, std)
  - Spectral Rolloff (mean, std)
  - Spectral Flatness (mean, std)
  - Spectral Contrast across 7 octave bands (mean, std)
- **Model Pipeline:** `StandardScaler` + `RandomForestClassifier(n_estimators=100, random_state=42)`.
- **Status in Repository:** Tracked directly in Git at:
  ```
  data/pilot_dataset/best_rf_model.joblib  (Size: ~364 KB)
  ```
- **Execution:** Ready out-of-the-box with zero setup.

---

## 3. Method 2: ResNet18 Mel-Spectrogram Ensemble

- **Input:** 128-band Mel-spectrogram:
  - FFT size: 1024
  - Hop length: 512
  - Sampling rate: 16 kHz
  - Converted to dB scale (top 80 dB) and normalized to $[0, 1]$.
- **Architecture:** Torchvision ResNet18 adapted for single-channel Mel-spectrogram inputs (channel replicated to $3\times$ to leverage ImageNet pre-training), with a linear binary classification head trained using `BCEWithLogitsLoss`.
- **Ensemble Strategy:** 5 frozen models trained under Leave-One-Generator-Out (LOGO) cross-validation:
  1. `LOGO_audioldm2_model.pt` (Held-out: AudioLDM2)
  2. `LOGO_stable_audio_open_model.pt` (Held-out: Stable Audio Open)
  3. `LOGO_musicldm_model.pt` (Held-out: MusicLDM)
  4. `LOGO_mustango_model.pt` (Held-out: Mustango)
  5. `LOGO_MusicGen_medium_model.pt` (Held-out: MusicGen Medium)
- **Artifact Details:** Each checkpoint is ~42.7 MB (totaling ~214 MB).
- **Storage Location:**
  ```
  experiments/resnet18_generalization/checkpoints/
  ```
- **Reproducing from Scratch:**
  ```powershell
  python scripts/evaluate_resnet.py
  ```
  This trains all 5 folds on the 600-clip pilot dataset and writes the `.pt` checkpoints directly to the checkpoints folder.

---

## 4. Method 3: External Pretrained Baseline (AST Detector)

- **Architecture:** Audio Spectrogram Transformer (AST) based on `AI-Music-Detection/ai_music_detection_large_60s`.
- **Role:** Independent reference baseline that was not trained on any HarmonyShield project data.
- **Weight Download:** Weights (~340 MB) download automatically from the HuggingFace Hub on initial execution into the local PyTorch/HuggingFace cache.
- **Fallback:** If offline, the detector interface logs a warning and the application reports the AST state gracefully.

---

## 5. Running Unified Inference

### CLI Mode:
To evaluate an audio file across all three frozen methods simultaneously:
```powershell
python scripts/demo_inference.py --audio path/to/track.mp3
```

### Full Web Stack:
```powershell
# 1. Start the FastAPI backend
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 2. Start the React/Vite frontend (in a separate terminal)
cd frontend
npm run dev
```
Open your browser at `http://localhost:5173`.
