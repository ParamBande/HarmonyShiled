# HarmonyShield: Complete Technical Implementation Audit

**Document Date:** November 2026  
**Audit Type:** Read-Only Source Code & System Architecture Inspection  
**Repository Path:** `d:\PROJECTS\harmonyshield\harmonyshield`  
**Purpose:** End-Semester Comprehensive ML Project Review & Viva Defense  

---

## 1. Complete Repository Structure

### 1.1 Directory Tree & Purpose of Key Folders

```
D:\PROJECTS\harmonyshield\harmonyshield\
├── backend\                    # FastAPI REST server wrapping frozen ML inference
│   ├── main.py                 # Core API: /health, /analyze, model loaders, consensus
│   ├── test_analyze.py         # End-to-end integration test script for /analyze
│   └── __pycache__\
├── data\                       # Datasets, audio artifacts, and extracted features
│   ├── external_demo\          # Out-of-distribution external demo tracks (Hangova, Suno)
│   ├── musiccaps_human_pilot\  # Downloaded & trimmed 10s human audio tracks (MusicCaps)
│   ├── pilot_dataset\          # Master 600-clip pilot dataset & classical model
│   │   ├── ai\                 # 500 AI audio clips across 5 generators
│   │   ├── human\              # Symlinks / copies of human clips
│   │   ├── best_model.json     # Pilot 01 benchmark metrics (LR on Set B)
│   │   ├── best_rf_model.joblib# Deployed Classical Model artifact (Random Forest on Set B)
│   │   ├── dataset.csv         # Master manifest (600 rows: path, label, generator, ytid)
│   │   ├── features_A.csv      # Extracted MFCC features (40 features + ytid + label)
│   │   ├── features_B.csv      # Extracted Spectral features for Pilot 01 (22 features)
│   │   ├── features_B_all.csv  # Extracted Spectral features for 600 clips (22 features)
│   │   ├── features_C.csv      # Extracted MFCC + Spectral features (62 features)
│   │   ├── features_D.csv      # Extracted MFCC + Spectral + Harmonic (88 features)
│   │   ├── features_E.csv      # Complete feature set (96 numerical features)
│   │   └── pilot_results.csv   # Classical benchmark results across 7 algorithms
│   └── samples\                # Minimal audio samples for quick verification
├── docs\                       # Architectural and technical documentation
│   ├── DATA_SETUP.md           # Dataset acquisition, prompt grouping & DSP documentation
│   ├── MODEL_SETUP.md          # Model checkpoints, specifications & setup guide
│   ├── PLAN_A_PIPELINE.md      # Classical feature extraction & ML design notes
│   ├── PLAN_B_MODEL_NOTES.md   # Pretrained AST & HuggingFace integration notes
│   └── RESNET18_METHOD.md      # Deep learning Mel-spectrogram & CNN methodology
├── experiments\                # Empirical reports, logs, and checkpoint artifacts
│   ├── demo\                   # External black-box evaluation outputs
│   │   ├── DEMO_PROTOCOL.md    # Strict black-box evaluation constraints
│   │   ├── demo_report.md      # Results on Suno AI track and Human track
│   │   └── external_tests_output.txt # Raw console logs of demo runs
│   ├── generator_generalization\ # Classical Leave-One-Generator-Out (LOGO) study
│   │   ├── generator_generalization_report.md
│   │   ├── generator_generalization_results.csv
│   │   └── LEAKAGE_AUDIT.md    # Proof of disjoint ytid train/test sets
│   ├── pilot_01_report.md      # Initial AudioLDM2 in-domain feasibility study
│   └── resnet18_generalization\# Deep learning LOGO study & checkpoints
│       ├── checkpoints\        # 5 frozen PyTorch checkpoints (~42.7 MB each)
│       │   ├── LOGO_audioldm2_model.pt
│       │   ├── LOGO_MusicGen_medium_model.pt
│       │   ├── LOGO_musicldm_model.pt
│       │   ├── LOGO_mustango_model.pt
│       │   └── LOGO_stable_audio_open_model.pt
│       ├── resnet18_report.md  # GPU LOGO experimental report
│       └── resnet18_results.csv# Raw metrics per held-out generator
├── frontend\                   # React 19 + Vite 6 + Tailwind CSS v4 web UI
│   ├── public\                 # Static public assets
│   ├── src\
│   │   ├── api\analyze.js      # Fetch wrapper calling /api/analyze with timer staging
│   │   ├── assets\             # SVG / PNG assets
│   │   ├── components\         # Modular UI widgets
│   │   │   ├── ConsensusPanel.jsx        # Consensus status, message, and vote counter
│   │   │   ├── FeatureImportanceChart.jsx# Horizontal Recharts bar chart
│   │   │   ├── MelSpectrogramCanvas.jsx  # HTML5 Canvas spectrogram visualizer
│   │   │   ├── ModelCard.jsx             # Individual card per detection method
│   │   │   ├── Navbar.jsx                # Forensic navigation bar
│   │   │   └── ProcessingStages.jsx      # Animated forensic pipeline stage indicator
│   │   ├── pages\
│   │   │   ├── AnalyzePage.jsx           # Main analysis & forensic dashboard
│   │   │   ├── HowItWorksPage.jsx        # Educational architecture breakdown
│   │   │   └── ResearchPage.jsx          # Interactive charts of frozen LOGO metrics
│   │   ├── App.jsx             # React Router config (/analyze, /how-it-works, /research)
│   │   ├── index.css           # Tailwind v4 theme, fonts, custom glassmorphism
│   │   └── main.jsx            # React root mount
│   ├── package.json            # Node dependencies
│   └── vite.config.js          # Vite config with /api -> http://localhost:8000 proxy
├── models\                     # Empty legacy placeholder directory
├── scripts\                    # Standalone execution and evaluation CLI scripts
│   ├── demo_inference.py       # Standalone 3-method CLI inference script
│   ├── evaluate_resnet.py      # ResNet18 5-fold LOGO training & evaluation runner
│   ├── predict.py              # CLI runner for Plan B (AST)
│   ├── run_demo.py             # Batch runner for external demo tracks
│   ├── train_resnet.py         # ResNet18 dataset sanity check
│   └── verify_cuda.py          # PyTorch CUDA tensor verification script
├── src\                        # Core modular Python package
│   ├── audio\
│   │   ├── preprocessor.py     # Librosa/Soundfile loading, mono, resample, 10s chunking
│   │   └── __init__.py
│   ├── evaluation\
│   │   ├── evaluator.py        # Accuracy, F1, Precision, Recall, ROC-AUC, CM
│   │   └── __init__.py
│   ├── features\
│   │   ├── aggregator.py       # Segment-level feature aggregation (mean/std)
│   │   ├── extractor.py        # 96-dimensional acoustic feature extraction (Sets A-E)
│   │   ├── melspectrogram.py   # Torchaudio MelSpectrogram & AmplitudeToDB extractor
│   │   └── __init__.py
│   ├── models\
│   │   ├── resnet18_model.py   # AudioResNet18 class (ImageNet ResNet18 adapted for 1-ch)
│   │   ├── trainer.py          # ModelTrainer wrapper around 7 scikit-learn models
│   │   └── __init__.py
│   ├── pipeline\
│   │   ├── infer.py            # Plan A multi-segment classical inference pipeline
│   │   ├── resnet_infer.py     # Standalone ResNet single-model inference pipeline
│   │   └── __init__.py
│   ├── pretrained\
│   │   ├── ast_detector.py     # HuggingFace AST transformer integration (60s chunks)
│   │   ├── detector_interface.py# Abstract base class for pretrained detectors
│   │   ├── lightweight_detector.py # Stub for lofcz/ai-music-detector (deferred)
│   │   └── __init__.py
│   └── utils\
│       ├── splitter.py         # GroupShuffleSplit wrapper based on ytid
│       └── __init__.py
├── tests\                      # Automated test suite
│   ├── test_plan_a.py          # Unit tests for preprocessing, features, splitter, trainer
│   └── test_predict.py         # Integration test for AST detector schema & error handling
├── .gitignore                  # Git exclusion rules for large datasets, binaries, .pt files
├── harmonyshield.py            # Legacy/draft top-level CLI (partially implemented)
├── README.md                   # Primary project overview, research findings, reproduction
├── requirements.txt            # Production Python package requirements
└── requirements-dev.txt        # Development dependencies (pytest)
```

### 1.2 Obsolete, Duplicate, Experimental, Unused, or Disconnected Files

1. **`harmonyshield.py`:** A top-level CLI with subcommands (`extract`, `train`, `evaluate`, `predict`). The first three subcommands print `"Not fully implemented until dataset is available"` and exit. Only the `predict` subcommand calls `src.pipeline.infer.PipelineInference`. This file is superseded by `scripts/demo_inference.py` and `backend/main.py`.
2. **`scripts/train_resnet.py`:** Despite its name, this file does **not** train the 5 LOGO models. It only contains a `sanity_check()` function that runs 2 mini-epochs on 4 dummy samples. The actual training and saving of the 5 LOGO checkpoints is implemented in `scripts/evaluate_resnet.py`.
3. **`models/` directory:** Completely empty. Checkpoints are stored in `experiments/resnet18_generalization/checkpoints/` and `data/pilot_dataset/`.
4. **`src/pretrained/lightweight_detector.py`:** A non-functional stub. It inherits from `DetectorInterface` but raises `NotImplementedError("LightweightDetector is not fully implemented yet")` if executed.
5. **`src/pipeline/infer.py` and `src/pipeline/resnet_infer.py`:** Standalone single-model inference classes. They are disconnected from the live backend (`backend/main.py`), which implements its own streamlined multi-model inference functions directly.
6. **`backend/test_analyze.py`:** A manual script to test `/analyze` via `urllib`. It is an internal test utility, not part of the production application.

### 1.3 Actual Application Entry Points

- **Live Full-Stack Application:**
  - **Backend API:** `backend/main.py` (launched via `uvicorn backend.main:app --port 8000`).
  - **Frontend Web UI:** `frontend/src/main.jsx` (launched via `cd frontend && npm run dev` on port 5173).
- **Standalone CLI Unified Inference:** `scripts/demo_inference.py` (executes all 3 methods via command line).
- **Deep Learning Training & Evaluation Runner:** `scripts/evaluate_resnet.py`.
- **Automated Test Suite:** `tests/test_plan_a.py` and `tests/test_predict.py` (executed via `python -m pytest`).

### 1.4 Files Essential for Running the Live Application

To run the live website and generate predictions, only the following files are strictly necessary:
1. `backend/main.py`
2. `src/audio/preprocessor.py`
3. `src/features/extractor.py`
4. `src/features/melspectrogram.py`
5. `src/models/resnet18_model.py`
6. `src/pretrained/ast_detector.py` and `src/pretrained/detector_interface.py`
7. `data/pilot_dataset/best_rf_model.joblib` (Classical model)
8. `experiments/resnet18_generalization/checkpoints/*.pt` (ResNet ensemble checkpoints)
9. The entire `frontend/` directory (UI code, Vite config, package.json).

---

## 2. Complete Application Architecture & Execution Flow

```
+--------------------------------------------------------------------------------------------------+
|                                        FRONTEND (React 19)                                       |
|                                                                                                  |
|  [User drops audio] ---> [AnalyzePage.jsx: FilePreview] ---> [Click "ANALYZE AUDIO"]            |
|                                                                     |                            |
|                                                                     v                            |
|                                                  [api/analyze.js: analyzeAudio()]                |
|                                                  - Initiates fetch POST /api/analyze             |
|                                                  - Runs client-side timer staging:               |
|                                                    "Loading audio" -> "Extracting features"...   |
+--------------------------------------------------------------------------------------------------+
                                              |
                                              | HTTP POST /api/analyze (multipart/form-data)
                                              | [Vite Proxy: :5173/api/analyze -> :8000/analyze]
                                              v
+--------------------------------------------------------------------------------------------------+
|                                       BACKEND (FastAPI)                                          |
|                                                                                                  |
| 1. Endpoint: analyze() in backend/main.py                                                        |
|    - Extension validation (.mp3, .wav, .flac, .m4a, .ogg)                                        |
|    - Writes audio stream to temporary file: tempfile.mkdtemp(prefix="hs_")                       |
|                                                                                                  |
| 2. Parallel / Sequential Method Execution:                                                       |
|                                                                                                  |
|    ┌─────────────────────────────────┬───────────────────────────────┬────────────────────────┐  |
|    │ A. _run_classical()             │ B. _run_resnet()              │ C. _run_ast()          │  |
|    │ - Preprocessor(10s, sr=16000)   │ - Preprocessor(10s, sr=16000) │ - librosa.load(16000)  │  |
|    │ - Takes Segment 0 (220500/16000)│ - Takes Segment 0             │ - Chunks into 60s      │  |
|    │ - FeatureExtractor(Set 'B')     │ - MelSpectrogramExtractor()   │ - AST Transformer      │  |
|    │ - 22 spectral features          │ - 5x LOGO ResNet18 models     │ - Softmax & label map  │  |
|    │ - rf.predict_proba(X)[0][1]     │ - Sigmoid & average probs     │ - Mean AI probability  │  |
|    └─────────────────────────────────┴───────────────────────────────┴────────────────────────┘  |
|                                                                                                  |
| 3. Auxiliary Forensic Extractions:                                                               |
|    - _get_feature_importances(10): pulls rf.named_steps['classifier'].feature_importances_       |
|    - _get_mel_image(): generates downsampled (128 x <=200) 2D grid for canvas rendering         |
|                                                                                                  |
| 4. Forensic Consensus Engine:                                                                    |
|    - _consensus(c_prob, r_prob, ast_prob, c_pred, r_pred, ast_pred)                              |
|    - Variance calculation: np.var([probs])                                                      |
|    - Threshold checks: if variance > 0.05 or (0.3 < mean_prob < 0.7) -> INCONCLUSIVE             |
|                                                                                                  |
| 5. Return JSON: AnalysisResponse                                                                 |
| 6. Cleanup: shutil.rmtree(tmp_dir) in finally block                                              |
+--------------------------------------------------------------------------------------------------+
                                              |
                                              | HTTP 200 OK (JSON payload)
                                              v
+--------------------------------------------------------------------------------------------------+
|                                        FRONTEND RENDERING                                        |
|                                                                                                  |
| AnalyzePage.jsx parses JSON and renders:                                                         |
| 1. ConsensusPanel.jsx: Visual badge (STRONG AI / STRONG HUMAN / MIXED / INCONCLUSIVE) + counts   |
| 2. ModelCard.jsx (x3): AI/Human probability bars for Classical, ResNet18 (with 5 fold breakdowns)|
|    and AST Transformer                                                                           |
| 3. MelSpectrogramCanvas.jsx: HTML5 canvas painting the 2D Mel matrix using viridis color map     |
| 4. FeatureImportanceChart.jsx: Recharts horizontal bar chart of top 10 classical features        |
+--------------------------------------------------------------------------------------------------+
```

---

## 3. Audio Processing and Preprocessing Implementation

### 3.1 Audio Loading and Formats
- **Supported Formats:** `.mp3`, `.wav`, `.flac`, `.m4a`, `.ogg`. Enforced at the FastAPI route level via extension validation (`allowed = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}`).
- **Loading Libraries:** `librosa.load` with a fallback to `soundfile.read` (`sf.read`).
- **Source File:** `src/audio/preprocessor.py`, method `AudioPreprocessor.process(file_path)`.

```python
# Lines 21-35 in src/audio/preprocessor.py
try:
    y, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
except Exception as e:
    import soundfile as sf
    y, sr = sf.read(file_path)
    if len(y.shape) > 1:
        y = y.mean(axis=1) # Force mono
    if sr != self.target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
```

### 3.2 Sample-Rate, Mono/Stereo, and Normalization
- **Channel Handling:** Strictly downmixed to mono. If stereo is loaded via `soundfile`, `y.mean(axis=1)` collapses channels into a 1D signal.
- **Sample Rates:**
  - `src/audio/preprocessor.py` defaults to `target_sr=22050`.
  - In `backend/main.py` and `scripts/demo_inference.py`, `AudioPreprocessor(segment_duration=10.0, target_sr=16000)` overrides the sample rate to **16,000 Hz**.
- **Peak Normalization:**
  ```python
  # Line 38-39 in src/audio/preprocessor.py
  if np.max(np.abs(y)) > 0:
      y = y / np.max(np.abs(y))
  ```
  Every audio array is peak-normalized to $[-1.0, 1.0]$ by dividing by its maximum absolute value, removing volume/amplitude as an unintended predictive shortcut.

### 3.3 Segmentation, Padding, and Windowing
- **Segment Length:** Exactly `10.0` seconds (`segment_samples = int(10.0 * target_sr)` = 160,000 samples at 16 kHz, or 220,500 samples at 22.05 kHz).
- **Padding:** If `len(y) < segment_samples`, `np.zeros(segment_samples)` is allocated and the audio is placed at index 0.
- **Trimming / Chunking:** Non-overlapping windows (`step = segment_samples`). Any trailing remainder shorter than 10 seconds is dropped.

### 3.4 Representation Disparity Across the Three Detection Methods
**CRITICAL FINDING:** The three detection methods do **not** evaluate identical representations of the audio file:
1. **Classical ML (`_run_classical`):** Evaluates **only the first 10-second segment** (`segments[0]`) resampled to 16 kHz, converted into a 22-dimensional feature vector.
2. **ResNet18 Ensemble (`_run_resnet`):** Evaluates **only the first 10-second segment** (`segments[0]`) resampled to 16 kHz, converted into a 128-band Mel-spectrogram tensor.
3. **AST Baseline (`_run_ast`):** Evaluates the **entire audio file**, broken into **60-second chunks** via `src/pretrained/ast_detector.py`, running an attention model over all chunks and averaging the resulting probabilities.

---

## 4. Classical Machine Learning Pipeline

### 4.1 Feature Extraction Implementation (`src/features/extractor.py`)
Features are computed per 10-second audio array `y` via `librosa`. Five named configurations (A through E) are supported:

| Feature Set | Components Included | Actual Numerical Feature Count |
|---|---|---|
| **Set A** | MFCC Only (20 coefficients $\times$ [mean, std]) | 40 features |
| **Set B** | Spectral Only (Centroid, Bandwidth, Rolloff, Flatness, 7 Contrast bands $\times$ [mean, std]) | **22 features** |
| **Set C** | Set A + Set B (MFCC + Spectral) | 62 features |
| **Set D** | Set C + Chroma (12 $\times$ [mean, std]) + HPSS (Harmonic & Percussive energy) | 88 features |
| **Set E** | Set D + Temporal (Onset mean/std, Tempo) + Energy (RMS mean/std, ZCR mean/std, Low Energy Ratio) | **96 features** |

*Documentation Note:* Some documentation files refer to Set B as having "14 features" (counting only the 7 $\times$ 2 spectral contrast features) and Set E as having "77" or "85" features. In the actual code (`src/features/extractor.py`), Set B has **22** features and Set E has **96** features.

### 4.2 Group-Aware Leakage Prevention (`src/utils/splitter.py`)
- Slicing a track into multiple 10s segments or evaluating synthetic tracks generated from the same text prompt introduces **conceptual data leakage** if split randomly.
- HarmonyShield implements `DataSplitter.split_data()` using scikit-learn's `GroupShuffleSplit`:
  ```python
  gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
  train_val_idx, test_idx = next(gss_test.split(X, y, groups))
  ```
  The `groups` parameter is populated with the YouTube ID (`ytid`). All clips sharing the same `ytid` (the human source and all 5 AI generations derived from that prompt) are isolated exclusively into either the train or test set.

### 4.3 Implemented Classical Algorithms (`src/models/trainer.py`)
`ModelTrainer` builds a scikit-learn `Pipeline([('scaler', StandardScaler()), ('classifier', clf)])` supporting 7 model types:
- `'lr'`: `LogisticRegression(max_iter=1000, random_state=42)`
- `'svm'`: `SVC(probability=True, random_state=42)`
- `'rf'`: `RandomForestClassifier(n_estimators=100, random_state=42)`
- `'dt'`: `DecisionTreeClassifier(random_state=42)`
- `'knn'`: `KNeighborsClassifier(n_neighbors=5)`
- `'gnb'`: `GaussianNB()`
- `'gbdt'`: `HistGradientBoostingClassifier(random_state=42)`

### 4.4 Model Deployed in Live Application
- **Algorithm:** **Random Forest** (`RandomForestClassifier`, 100 trees).
- **Feature Set:** **Feature Set B (Spectral Features only, 22 dimensions)**.
- **Artifact Path:** `data/pilot_dataset/best_rf_model.joblib` (Size: 364 KB).
- **Probability Calculation:**
  ```python
  feats, _ = extractor.extract_features(seg, feature_set="B")
  X = feats.reshape(1, -1)
  prob = float(rf.predict_proba(X)[0][1])  # Class index 1 = AI
  pred = "AI" if prob > 0.5 else "Human"
  ```
- **Clarification:** `data/pilot_dataset/best_model.json` records that **Logistic Regression** achieved the highest score in the initial Pilot 01 benchmark (Accuracy: 0.975, F1: 0.9756, ROC-AUC: 0.9975). However, the deployed artifact is `best_rf_model.joblib` because Random Forest provides interpretable `feature_importances_` to power the forensic dashboard.

---

## 5. Deep Learning Pipeline — ResNet18

### 5.1 Mel-Spectrogram Extraction (`src/features/melspectrogram.py`)
- **Sampling Rate:** 16,000 Hz.
- **FFT Size (`n_fft`):** 1024 samples.
- **Hop Length (`hop_length`):** 512 samples.
- **Mel Frequency Bands (`n_mels`):** 128.
- **Dynamic Range Scaling:** `torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)`.
- **Normalization to $[0, 1]$:**
  ```python
  mel_db = (mel_db + 80.0) / 80.0
  mel_db = torch.clamp(mel_db, 0.0, 1.0)
  ```
- **Output Tensor Shape:** `(1, 128, T)` where $T \approx 313$ frames for a 10-second window.

### 5.2 Network Architecture (`src/models/resnet18_model.py`)
- **Base Architecture:** Standard `torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)`.
- **Channel Adaptation:** Spectrograms have 1 channel, but ImageNet weights require 3 channels (RGB). Rather than modifying the first convolutional kernel, the tensor is repeated across 3 channels in `forward()`:
  ```python
  if x.size(1) == 1:
      x = x.repeat(1, 3, 1, 1)
  logits = self.resnet(x)
  ```
- **Classification Head:** The 1000-class fully connected layer is replaced with a single linear output:
  ```python
  num_ftrs = self.resnet.fc.in_features  # 512
  self.resnet.fc = nn.Linear(num_ftrs, 1)
  ```

### 5.3 Training Procedure (`scripts/evaluate_resnet.py`)
- **Hardware Used:** NVIDIA GeForce GTX 1650 (CUDA 12.4).
- **Dataset:** 600 clips (100 Human + 500 AI across 5 generators).
- **Epochs:** 3 epochs per fold.
- **Batch Size:** 16.
- **Loss Function:** `torch.nn.BCEWithLogitsLoss()`.
- **Optimizer:** `torch.optim.Adam(model.parameters(), lr=1e-4)`.

### 5.4 Checkpoints & Ensemble Inference
The live application loads **all five frozen Leave-One-Generator-Out checkpoints**:
1. `experiments/resnet18_generalization/checkpoints/LOGO_audioldm2_model.pt`
2. `experiments/resnet18_generalization/checkpoints/LOGO_stable_audio_open_model.pt`
3. `experiments/resnet18_generalization/checkpoints/LOGO_musicldm_model.pt`
4. `experiments/resnet18_generalization/checkpoints/LOGO_mustango_model.pt`
5. `experiments/resnet18_generalization/checkpoints/LOGO_MusicGen_medium_model.pt`

**Inference Combining Rule:**
- Each model outputs a raw logit on the first 10-second segment Mel-spectrogram.
- Probabilities are extracted via `torch.sigmoid(logit).item()`.
- The final ResNet18 AI probability is the **unweighted arithmetic mean** of the 5 checkpoint probabilities:
  $$\text{ResNet AI Probability} = \frac{1}{5} \sum_{i=1}^{5} P_i(\text{AI})$$
- Prediction label: `"AI"` if mean probability $> 0.5$, else `"Human"`.

---

## 6. Third Detection Method — Pretrained AST (Plan B)

### 6.1 Model Specifications (`src/pretrained/ast_detector.py`)
- **Pretrained Identifier:** `AI-Music-Detection/ai_music_detection_large_60s` (HuggingFace Hub).
- **Architecture:** Audio Spectrogram Transformer (AST) with patch-based self-attention (~90.8M parameters).
- **Fallback Feature Extractor:** If config is missing, falls back to `MIT/ast-finetuned-audioset-10-10-0.4593`.
- **Target Sample Rate:** 16,000 Hz mono.
- **Input Window Duration:** 60.0 seconds.

### 6.2 Chunking & Inference Logic
1. Loads full audio file via `librosa.load(audio_path, sr=16000, mono=True)`.
2. Slices into 60-second non-overlapping windows.
3. For each window, extracts spectrogram patches using `AutoFeatureExtractor(padding="max_length", max_length=6000)`.
4. Executes model on GPU/CPU under `torch.no_grad()`.
5. Computes Softmax over the output logits.
6. Maps output classes dynamically by scanning `model.config.id2label`:
   - Labels containing `"ai"`, `"generated"`, or `"fake"` accumulate into `ai_prob`.
   - Labels containing `"human"`, `"real"`, or `"composed"` accumulate into `human_prob`.
7. Aggregates across all 60s windows by taking the arithmetic mean.
8. Normalizes: $P_{\text{AI}} = \frac{P_{\text{AI}}}{P_{\text{AI}} + P_{\text{Human}}}$.
9. Binary Prediction: `"AI"` if $P_{\text{AI}} > 0.5$, else `"Human"`.

---

## 7. Evaluation Experiments & Research Findings

### 7.1 Pilot 01 Benchmark (AudioLDM2 vs Human)
- Evaluated on 100 Human vs 100 AudioLDM2 clips (80 train ytids, 20 test ytids).
- **Key Finding:** In-domain separation was near-perfect (Logistic Regression on Set B achieved F1: 0.9756, Random Forest achieved F1: 0.9524).
- **Top Predictor:** High-frequency spectral contrast (`spectral_contrast_6_std`, `spectral_contrast_7_mean`).

### 7.2 Classical Leave-One-Generator-Out (LOGO) Experiment
To test whether these spectral contrast artifacts generalize to unseen generators, an 80/20 group split was run across all 5 generators, holding out one generator entirely for testing:

| Held-Out Generator | Classical RF Accuracy | Classical RF F1 | Classical RF ROC-AUC |
|---|---|---|---|
| `audioldm2` | 0.6000 | 0.6667 | 0.6225 |
| `stable_audio_open` | 0.6000 | 0.6667 | 0.6812 |
| `musicldm` | 0.6500 | 0.7407 | 0.7550 |
| `mustango` | 0.6250 | 0.7170 | 0.7212 |
| `MusicGen_medium` | 0.6750 | 0.7547 | 0.7963 |
| **Mean ± Std** | **0.6300 ± 0.0306** | **0.7092 ± 0.0367** | **0.7152 ± 0.0601** |

**Scientific Conclusion:** Performance on AudioLDM2 collapsed from **0.9524** (in-domain) to **0.6667** (held-out). Handcrafted classical features are substantially **generator-specific**.

### 7.3 ResNet18 Leave-One-Generator-Out (LOGO) Experiment
The 5-fold LOGO experiment was repeated on Mel-spectrograms with the GPU-accelerated ResNet18:

| Held-Out Generator | ResNet18 Accuracy | ResNet18 F1 | ResNet18 ROC-AUC |
|---|---|---|---|
| `audioldm2` | 0.6750 | 0.6977 | 0.6275 |
| `stable_audio_open` | 0.5000 | 0.6552 | 0.6075 |
| `musicldm` | 0.7250 | 0.7843 | 0.8475 |
| `mustango` | 0.5750 | 0.7018 | 0.6650 |
| `MusicGen_medium` | 0.7250 | 0.7843 | 0.8250 |
| **Mean ± Std** | **0.6400 ± 0.0894** | **0.7246 ± 0.0514** | **0.7145 ± 0.0967** |

**Scientific Comparison:**
- ResNet18 Mean LOGO F1 (**0.7246**) slightly exceeded Classical RF (**0.7092**), while ROC-AUC was virtually identical (**0.7145 vs 0.7152**).
- Both architectures hit a **generalization ceiling** around 71–72% F1 on unseen generators, scientifically validating that single-model architectures fail to learn a universal AI artifact boundary.

### 7.4 External Black-Box Demo Results (`experiments/demo/demo_report.md`)
Two external, out-of-distribution audio files were tested on the frozen three-method pipeline:

1. **Human Song (`Hangova (From DC ).mp3`):**
   - Classical RF: $P_{\text{AI}} = 0.1700$ (Predicts Human)
   - ResNet18 Ensemble: $P_{\text{AI}} = 0.9389$ (Predicts AI — False Positive)
   - Pretrained AST: $P_{\text{AI}} = 0.3802$ (Predicts Human)
   - **Consensus:** **`INCONCLUSIVE / LOW CONFIDENCE`** (Variance = 0.116 > 0.05).
2. **Suno AI Song (`धडधडतं_शहर.mp3`):**
   - Classical RF: $P_{\text{AI}} = 0.0000$ (Predicts Human — False Negative)
   - ResNet18 Ensemble: $P_{\text{AI}} = 0.7862$ (Predicts AI)
   - Pretrained AST: $P_{\text{AI}} = 0.9947$ (Predicts AI)
   - **Consensus:** **`INCONCLUSIVE / LOW CONFIDENCE`** (Variance = 0.190 > 0.05).

---

## 8. Three-Model Inference and Consensus Implementation

### 8.1 Summary of Individual Method Decisions

| Method | Feature / Input | Binary Threshold | Decision Formula |
|---|---|---|---|
| **Classical ML** | Segment 0, 22 Spectral features | 0.50 | $\text{AI if } P_{\text{AI}} > 0.5 \text{ else Human}$ |
| **ResNet18** | Segment 0, Mel-spectrogram | 0.50 | $\text{AI if } \frac{1}{5}\sum P_i > 0.5 \text{ else Human}$ |
| **AST Transformer** | Full file, 60s windows | 0.50 | $\text{AI if } \text{mean}(P_{\text{AI}}) > 0.5 \text{ else Human}$ |

### 8.2 Consensus Decision Logic
Implemented in `backend/main.py` (`_consensus`) and `scripts/demo_inference.py`:

```
FUNCTION compute_consensus(probs, preds):
    valid_pairs = filter(probs, preds where prob is not None)
    IF count(valid_pairs) == 0:
        RETURN status = "ERROR"

    variance = variance_of(valid_probs)
    mean_prob = mean_of(valid_probs)
    num_ai_votes = count(preds == "AI")
    num_human_votes = count(preds == "Human")

    # RULE 1: Disagreement or Borderline Confidence -> Inconclusive
    IF variance > 0.05 OR (0.3 < mean_prob < 0.7):
        RETURN status = "INCONCLUSIVE"
               message = "The detection methods disagree substantially on this audio.
                          Treat the result cautiously rather than as a definitive classification."

    # RULE 2: Unanimous AI
    IF num_ai_votes == count(valid_pairs):
        RETURN status = "STRONG_CONSENSUS_AI"
               message = "All methods indicate AI-generated patterns."

    # RULE 3: Unanimous Human
    IF num_human_votes == count(valid_pairs):
        RETURN status = "STRONG_CONSENSUS_HUMAN"
               message = "All methods indicate human-composed patterns."

    # RULE 4: Majority
    RETURN status = "MIXED"
           message = "Methods disagree. Majority suggests [AI/Human]."
```

### 8.3 Handling of Missing or Failed Models
- If a checkpoint fails to load (e.g. `.pt` file missing), the backend logs a warning and omits it.
- In `_consensus()`, `valid` only includes models returning non-null probabilities. If only 2 models succeed, consensus is computed across those 2. If 0 succeed, it returns `status="ERROR"`.

---

## 9. FastAPI Backend Implementation (`backend/main.py`)

### 9.1 Lifecycle & Model Caching
- Models are loaded into memory once at server startup using FastAPI's `@asynccontextmanager lifespan(app)`:
  - Sets PyTorch device (`cuda` or `cpu`).
  - Initializes `AudioPreprocessor(segment_duration=10.0, target_sr=16000)`.
  - Loads `best_rf_model.joblib` and extracts `feature_importances_`.
  - Iterates through the 5 ResNet `.pt` files, instantiating `AudioResNet18` and moving to device. Missing checkpoints are gracefully logged and skipped.
  - Instantiates `ASTDetector()`.
- Loaded instances are cached in a global dictionary `_models`.

### 9.2 Endpoints

#### 1. `GET /health`
Returns server operational status, active compute device (`cuda` or `cpu`), and list of loaded model keys.
```json
{
  "status": "ok",
  "device": "cuda",
  "models_loaded": ["device", "preprocessor", "rf", "classical_extractor", "rf_importances", "resnet_extractor", "resnet_models", "ast"]
}
```

#### 2. `POST /analyze`
Accepts `multipart/form-data` with an audio file under field name `file`.
- **Validation:** Rejects extensions not in `{".mp3", ".wav", ".flac", ".m4a", ".ogg"}` with HTTP 422.
- **Temporary Storage:** Writes file to an isolated temporary directory via `tempfile.mkdtemp(prefix="hs_")`.
- **Cleanup Guarantee:** Wrapped in a `try ... finally: shutil.rmtree(tmp_dir, ignore_errors=True)` block, ensuring zero leftover disk clutter.
- **Example Response Payload:**
```json
{
  "file": {
    "name": "sample.mp3",
    "size_bytes": 1048576,
    "duration": 30.0,
    "sample_rate": 16000,
    "num_segments": 3
  },
  "classical": {
    "prediction": "AI",
    "ai_probability": 0.8421,
    "human_probability": 0.1579,
    "error": null
  },
  "resnet18": {
    "prediction": "AI",
    "ai_probability": 0.7915,
    "human_probability": 0.2085,
    "ensemble_size": 5,
    "checkpoint_results": [
      {"name": "audioldm2", "ai_probability": 0.7214},
      {"name": "stable_audio_open", "ai_probability": 0.8105},
      {"name": "musicldm", "ai_probability": 0.8541},
      {"name": "mustango", "ai_probability": 0.7420},
      {"name": "MusicGen_medium", "ai_probability": 0.8295}
    ],
    "error": null
  },
  "ast": {
    "prediction": "AI",
    "ai_probability": 0.9124,
    "human_probability": 0.0876,
    "error": null
  },
  "consensus": {
    "status": "STRONG_CONSENSUS_AI",
    "message": "All 3 methods indicate AI-generated patterns...",
    "num_ai_votes": 3,
    "num_human_votes": 0
  },
  "features": [
    {"name": "spectral_contrast_7_mean", "importance": 0.1245, "family": "Spectral Contrast"},
    {"name": "spectral_contrast_6_std", "importance": 0.0981, "family": "Spectral Contrast"}
  ],
  "mel_spectrogram": [[0.12, 0.45], [0.89, 0.32]],
  "inference_time_s": 2.45
}
```

---

## 10. React Frontend Implementation

### 10.1 Tech Stack & Build System
- **Framework:** React 19.2 + Vite 8.3.
- **Styling:** Tailwind CSS v4.3 (`@tailwindcss/vite`).
- **Icons & Animation:** `lucide-react` (1.49) + `framer-motion` (13.4).
- **Visualization:** `recharts` (3.10) for bar charts + native HTML5 Canvas for the Mel-spectrogram.
- **Routing:** `react-router-dom` (v7.18) with 3 routes:
  - `/` or `/analyze`: `AnalyzePage.jsx`
  - `/how-it-works`: `HowItWorksPage.jsx`
  - `/research`: `ResearchPage.jsx`

### 10.2 Pages and Components
1. **`AnalyzePage.jsx`:**
   - Drag-and-drop audio upload zone with decorative SVG waveform bars.
   - Audio preview with native HTML5 `<audio>` player.
   - Connects to `api/analyze.js` via `analyzeAudio()`.
   - Displays animated forensic checklist (`ProcessingStages.jsx`).
   - Renders `ConsensusPanel`, 3 `ModelCard` components, `MelSpectrogramCanvas`, and `FeatureImportanceChart`.
2. **`ModelCard.jsx`:**
   - Displays method badge, title, subtitle, and prediction verdict.
   - Dual animated probability bars (AI vs Human).
   - If rendering ResNet18, displays an expandable breakdown of individual predictions across all 5 LOGO folds.
3. **`ConsensusPanel.jsx`:**
   - Dynamic coloring and iconography based on consensus status:
     - `STRONG_CONSENSUS_AI`: Red (`#f87171`) with `XCircle`.
     - `STRONG_CONSENSUS_HUMAN`: Green (`#4ade80`) with `CheckCircle`.
     - `MIXED`: Yellow (`#fbbf24`) with `Info`.
     - `INCONCLUSIVE`: Amber (`#fbbf24`) with `AlertTriangle`.
   - Vote counter showing AI votes vs Human votes.
4. **`MelSpectrogramCanvas.jsx`:**
   - Receives the 2D matrix (`128 x <=200`) from the backend.
   - Paints directly to an HTML5 `<canvas>` using an interpolated **Viridis** color palette (dark purple $\rightarrow$ teal $\rightarrow$ yellow).
5. **`FeatureImportanceChart.jsx`:**
   - Renders the top 10 classical features using a Recharts horizontal `BarChart`.
   - Features are color-coded by feature family (e.g. Spectral Contrast in `#38bdf8`, Flatness in `#818cf8`, Centroid in `#4ade80`).
6. **`ResearchPage.jsx`:**
   - Displays the frozen Leave-One-Generator-Out research results using Recharts radar and bar charts.
   - **Data Origin:** Data on this page is **static** (hardcoded constants `CLASSICAL_LOGO` and `RESNET_LOGO`), representing frozen historical research outputs.

---

## 11. Dependencies and Execution Environment

### 11.1 Python Environment & Tooling
- **Python Version:** 3.13.9 (in `venv`).
- **PyTorch & CUDA:** PyTorch `2.6.0+cu124` with CUDA 12.4 support. Verified on NVIDIA GeForce GTX 1650.
- **Torchaudio:** `2.6.0+cu124`.
- **Torchvision:** `0.21.0+cu124` (Installed in venv, but **missing from `requirements.txt`**).
- **Core Scientific Libraries:** `scikit-learn==1.6.1`, `numpy>=2.1.0`, `librosa==0.10.2`, `soundfile==0.13.1`, `transformers==4.49.0`, `joblib`.
- **API Framework:** `fastapi>=0.115.0`, `uvicorn>=0.30.0`, `python-multipart>=0.0.18`.

### 11.2 Commands to Run the Application

```powershell
# Terminal 1: Launch FastAPI Backend
cd D:\PROJECTS\harmonyshield\harmonyshield
.\venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Launch React Frontend
cd D:\PROJECTS\harmonyshield\harmonyshield\frontend
npm run dev
```

---

## 12. Data, Models, and Storage Audit

### 12.1 Git Tracking vs Exclusions (`.gitignore`)
- **Ignored from Git (via `.gitignore`):**
  - All raw audio formats: `*.wav`, `*.mp3`, `*.flac`, `*.m4a`, `*.ogg` (~406 MB).
  - Raw audio directories: `data/musiccaps_human_pilot/`, `data/pilot_dataset/ai/`, `data/external_demo/`.
  - Heavy PyTorch weights: `*.pt`, `*.pth`, `experiments/resnet18_generalization/checkpoints/` (~214 MB).
  - Virtual environments: `venv/`, `frontend/node_modules/`, `frontend/dist/`.
- **Tracked in Git:**
  - `data/pilot_dataset/best_rf_model.joblib` (364 KB, explicitly exempted via `!data/pilot_dataset/best_rf_model.joblib`).
  - All pre-extracted feature CSV files (`features_A.csv` through `features_E.csv`).
  - Master dataset manifest `dataset.csv`.
  - All source code, documentation, and frontend assets.

### 12.2 Missing Dependencies in Configuration Files
1. **`torchvision`:** Used in `src/models/resnet18_model.py` (`from torchvision.models import resnet18, ResNet18_Weights`). It is installed in the local virtual environment, but is **absent from `requirements.txt`**. A clean clone running `pip install -r requirements.txt` will fail on import unless `torchvision` is installed.
2. **`joblib`:** Used throughout `src/models/trainer.py` and `backend/main.py`. It is installed transitively via scikit-learn, but not explicitly declared.

---

## 13. Testing and Reliability Audit

### 13.1 Test Suite Status
HarmonyShield has two test files under `tests/`:
1. `tests/test_plan_a.py` (4 unit tests):
   - `test_audio_segmentation`: Validates 10s chunking and zero-padding of short files.
   - `test_feature_extraction`: Validates feature dimensions for Sets A and E, and checks NaN/Inf imputation.
   - `test_group_split`: Verifies disjoint group sets under `GroupShuffleSplit`.
   - `test_model_persistence`: Verifies model saving and loading with scikit-learn.
2. `tests/test_predict.py` (2 integration tests):
   - `test_predict_output_schema`: Mocks audio loading and checks AST detector response keys.
   - `test_invalid_audio`: Validates exception handling on nonexistent files.

**Test Execution Result:** Both test suites pass (`6 passed in 45s`).  
*Execution Note:* Running `pytest tests/test_plan_a.py` directly fails with `ModuleNotFoundError: No module named 'src'` because `sys.path` is not configured inside `test_plan_a.py`. It must be executed as `python -m pytest tests/test_plan_a.py`.

### 13.2 Warnings in Python 3.13
During test execution on Python 3.13.9, deprecated standard library warnings occur:
- `audioread` triggers `DeprecationWarning: aifc was removed in Python 3.13` and `sunau was removed in Python 3.13`. Handled via installed fallback packages (`standard-aifc`, `standard-sunau`).
- `librosa` triggers a deprecation warning on `__audioread_load`.

---

## 14. Scientific Validity & Research Limitations

### 14.1 Legitimate Claims for the Viva Defense
1. **Leakage Prevention is Proven:** Grouping by YouTube ID (`ytid`) successfully isolates musical prompts. The fact that held-out generator accuracy drops to ~60-70% proves the models did not cheat by memorizing prompt content.
2. **Generalization Ceiling Discovered:** Both classical models (F1: 0.7092) and deep CNNs (F1: 0.7246) suffer substantial degradation on unseen generators.
3. **Forensic Disagreement is Surfaced Honestly:** Rather than forcing a high-confidence guess, the ensemble measures probability variance and safely falls back to `INCONCLUSIVE / LOW CONFIDENCE` on unseen generators (such as Suno) and out-of-distribution human music.

### 14.2 Claims That Would Be Scientifically Misleading
1. **"Our system achieves 95% accuracy in the real world."** False. 95% was achieved only on in-domain AudioLDM2 data in Pilot 01. Real-world out-of-distribution accuracy on unseen generators is ~71-72%.
2. **"Deep learning solved the generator generalization problem."** False. ResNet18 achieved 0.7246 F1 vs Classical RF's 0.7092 F1 — a marginal improvement of only 1.5%.
3. **"The three models are completely independent."** Not fully. Classical ML and ResNet18 both operate on the first 10 seconds of the track and share frequency representations (Fourier/Mel domain), although one uses scalar summary statistics and the other uses 2D spatial convolutions.
4. **"The models detected Suno AI."** Only partially true. In the black-box demo, ResNet and AST flagged Suno as AI, but the Classical model failed completely ($P_{\text{AI}} = 0.00$), triggering the `INCONCLUSIVE` safety mechanism.

---

## 15. Final Technical Summary

### A. System Architecture Diagram

```
[Incoming Audio (.mp3/.wav/.flac/.m4a/.ogg)]
                    │
                    ▼
          FastAPI Backend (POST /analyze)
                    │
       ┌────────────┴────────────┐
       ▼                         ▼
AudioPreprocessor           AST Loading
(16 kHz Mono, Peak Norm)    (16 kHz Mono)
       │                         │
[10s Segment 0]             [60s Windows]
       ├─────────────────────────┐
       ▼                         ▼
FeatureExtractor(Set B)     MelSpectrogramExtractor
(22 Spectral Features)      (128 Mels, dB scale, [0,1])
       │                         │
       ▼                         ▼
best_rf_model.joblib        5x LOGO ResNet18 Checkpoints
(Random Forest)             (ImageNet Backbone, fc=1)
       │                         │
       ▼                         ▼
P_classical (AI Prob)       P_resnet (Sigmoid & Mean of 5 folds)
       │                         │
       └────────────┬────────────┘
                    │
                    ▼
          Pretrained AST Transformer
          (ai_music_detection_large_60s)
                    │
                    ▼
          P_ast (Softmax & Label Mapping)
                    │
                    ▼
        Forensic Consensus Engine
  Variance > 0.05 or 0.3 < Mean < 0.7?
       ├── YES ──> Status: "INCONCLUSIVE"
       └── NO  ──> Unanimous AI / Human / Majority
                    │
                    ▼
       JSON Response to React Frontend
 (ConsensusPanel + ModelCards + MelCanvas + Recharts)
```

### B. Summary of Three Detection Methods

| Specification | Method 1: Classical ML | Method 2: ResNet18 Ensemble | Method 3: Pretrained AST |
|---|---|---|---|
| **Underlying Algorithm** | Random Forest (100 estimators) | 5x ResNet18 (ImageNet backbone) | Audio Spectrogram Transformer (AST) |
| **Input Representation** | 22 Handcrafted Spectral features | 128-band Mel-spectrogram tensor | 60s Spectrogram patches |
| **Temporal Coverage** | First 10 seconds | First 10 seconds | Entire file (60s sliding windows) |
| **Model Weights** | `data/pilot_dataset/best_rf_model.joblib` | 5x `.pt` files in `experiments/.../checkpoints/` | Cached from HuggingFace Hub |
| **Output Metric** | $P(\text{AI}) \in [0, 1]$ | Mean $P(\text{AI})$ across 5 folds | Normalized Softmax $P(\text{AI})$ |
| **Decision Threshold** | 0.50 | 0.50 | 0.50 |
| **Primary Limitation** | Fails completely on unseen generators | Prone to false positives on human music | Computationally heavy (~90.8M params) |

### C. Research Experiments vs. Deployed Application

| Component | Research Experiment State | Deployed Application State |
|---|---|---|
| **Classical Model** | Evaluated 7 algorithms across Sets A-E. Best was Logistic Regression on Set B (F1: 0.9756). | Deploys **Random Forest on Set B** (`best_rf_model.joblib`) for feature importance transparency. |
| **ResNet18 Model** | Evaluated 5 separate LOGO models individually to measure generalization drop. | Deploys **all 5 checkpoints simultaneously as an ensemble**, averaging their probabilities. |
| **AST Model** | Explored as an independent external baseline (Plan B). | Deployed as the 3rd pillar in the live consensus engine. |
| **External Demo Tracks** | Both tracks yielded `INCONCLUSIVE` due to inter-model variance. | Accurately rendered on the UI as `INCONCLUSIVE / LOW CONFIDENCE`. |

### D. The 20 Most Critical Source Files for the Review
1. `backend/main.py`: Core server, model cache, `/analyze` route, consensus calculation.
2. `src/audio/preprocessor.py`: Audio loading, peak normalization, mono conversion, 10s chunking.
3. `src/features/extractor.py`: Implementation of 96 acoustic features and Sets A-E.
4. `src/features/melspectrogram.py`: Mel-spectrogram computation and dB normalization.
5. `src/models/resnet18_model.py`: AudioResNet18 class with channel repetition and binary head.
6. `src/models/trainer.py`: Scikit-learn Pipeline construction and model persistence.
7. `src/pretrained/ast_detector.py`: AST transformer inference, chunking, and label mapping.
8. `src/utils/splitter.py`: `GroupShuffleSplit` implementation enforcing `ytid` isolation.
9. `scripts/demo_inference.py`: Standalone CLI reproduction of the 3-method consensus engine.
10. `scripts/evaluate_resnet.py`: Complete training and evaluation runner for the 5 LOGO folds.
11. `frontend/src/api/analyze.js`: API client wrapper with simulated progress stages.
12. `frontend/src/pages/AnalyzePage.jsx`: Main UI dashboard connecting all forensic widgets.
13. `frontend/src/components/ConsensusPanel.jsx`: Visual representation of agreement and uncertainty.
14. `frontend/src/components/ModelCard.jsx`: Multi-model probability displays and fold breakdowns.
15. `frontend/src/components/MelSpectrogramCanvas.jsx`: Direct canvas rendering with Viridis palette.
16. `frontend/src/components/FeatureImportanceChart.jsx`: Recharts bar chart of top classical features.
17. `experiments/generator_generalization/generator_generalization_report.md`: Classical LOGO report.
18. `experiments/resnet18_generalization/resnet18_report.md`: ResNet18 GPU LOGO report.
19. `experiments/demo/demo_report.md`: Black-box evaluation report on Suno and human music.
20. `data/pilot_dataset/dataset.csv`: Master metadata manifest linking audio paths to `ytid`s.

### E. Discrepancies and Inconsistencies Discovered
1. **Feature Dimension Counts:** Documentation (`DATA_SETUP.md` and `MODEL_SETUP.md`) states Set B has "14 features" and Set E has "85 features". In the actual code (`src/features/extractor.py`), Set B has **22** features (14 contrast + 8 other spectral) and Set E has **96** features.
2. **Missing `torchvision` in `requirements.txt`:** `src/models/resnet18_model.py` imports `torchvision.models`, but `torchvision` is omitted from `requirements.txt`.
3. **`CLASSICAL_LOGO` in `ResearchPage.jsx`:** The static numbers in `frontend/src/pages/ResearchPage.jsx` for Classical LOGO on AudioLDM2 list `acc: 0.7000, f1: 0.6977`, which are the ResNet18 metrics rather than the classical metrics (`acc: 0.6000, f1: 0.6667`).
4. **Pytest Import Path in `tests/test_plan_a.py`:** Does not prepend `sys.path`, requiring execution via `python -m pytest` rather than bare `pytest`.
5. **Representation Inconsistency:** Classical ML and ResNet18 only inspect the first 10 seconds of a song, whereas the AST model inspects the full song in 60s increments.

### F. Working Components That Must NOT Be Disturbed
- The trained classical model `data/pilot_dataset/best_rf_model.joblib`.
- The five ResNet18 checkpoints in `experiments/resnet18_generalization/checkpoints/`.
- The consensus and variance mathematical logic (`variance > 0.05`).
- The peak normalization and audio chunking logic in `src/audio/preprocessor.py`.
- The Vite proxy configuration connecting React to FastAPI.

### G. Concise Viva Defense Script (Student Walkthrough)
> *"HarmonyShield addresses the core vulnerability in AI music forensics: generator overfitting. Single detectors easily learn generator-specific artifacts—such as high-frequency vocoder phase errors—that fail to generalize to unseen models. In our Leave-One-Generator-Out experiments across 5 generators, classical models and deep ResNets both experienced an F1 drop to ~71-72% on unseen models.
>
> To solve this, HarmonyShield implements a tripartite architecture:
> 1. A handcrafted Spectral Random Forest analyzing mathematical timbre statistics.
> 2. A 5-checkpoint ResNet18 ensemble analyzing visual Mel-spectrogram texture.
> 3. An external Audio Spectrogram Transformer providing an independent baseline.
>
> Rather than forcing a fragile majority vote, our consensus engine computes the statistical variance across these models. If the variance exceeds 0.05 or confidence is borderline, the system explicitly reports 'INCONCLUSIVE'. When tested on black-box external tracks (Suno AI and human music), the models diverged, and HarmonyShield safely reported Inconclusive rather than issuing a confident false accusation. This prioritizes forensic integrity over blind classification."*
