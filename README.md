# HarmonyShield

<div align="center">

### Machine Learning-Based Audio Forensics for Detection of AI-Generated Music

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2Bcu124-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg?logo=react)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 18 Passed](https://img.shields.io/badge/Tests-18%2F18%20Passed-brightgreen.svg)]()

*A multi-method forensic research prototype evaluating acoustic artifacts, deep spectrogram representations, and transformer baselines across seen and unseen generative models.*

[Key Capabilities](#key-capabilities) • [System Architecture](#system-architecture) • [Research Methodology](#methodology) • [Empirical Results](#research-results) • [Quickstart Guide](#installation-and-usage) • [Team Handover](TEAM_HANDOVER.md)

</div>

---

## 1. Overview & Research Motivation

The rapid advancement of text-to-music generative diffusion models (e.g., AudioLDM2, Stable Audio Open, MusicLDM, Mustango, MusicGen) has made it straightforward to generate high-fidelity music tracks with naturalistic timbral dynamics and structural phrasing. However, distinguishability between authentic human performance recordings and synthetic audio poses significant challenges for digital copyright verification, provenance tracking, and audio authenticity.

**HarmonyShield** is an audio forensics research project and interactive web application designed to systematically detect AI-generated music. Rather than relying on a single brittle classifier, HarmonyShield employs **three orthogonal detection approaches** and synthesizes their predictions through a transparent majority-voting decision rule:

1. **Classical Machine Learning (Acoustic Feature Forensics):** Extracts 22 engineered spectral features (Spectral Rolloff, Centroid, Flatness, Bandwidth, and Multiband Spectral Contrast) paired with a Random Forest classifier to expose timbral and frequency-distribution anomalies.
2. **Deep Representation Learning (ResNet-18 Ensemble):** Computes log-Mel spectrograms from 10-second segments and evaluates them across a 5-checkpoint Leave-One-Generator-Out (LOGO) ResNet-18 ensemble.
3. **Pretrained Foundation Model Baseline (AST Detector):** Utilizes an Audio Spectrogram Transformer fine-tuned on full-track 60-second audio windows to evaluate macroscopic structural artifacts.

> [!NOTE]
> **Research Prototype Scope:** HarmonyShield is designed as an investigative research tool and forensic diagnostic system. It does not claim 100% universal accuracy on arbitrary commercial audio, but provides interpretable diagnostic evidence, surfaces inter-model disagreements, and documents the fundamental generalization ceiling when encountering novel, unseen generative architectures.

---

## 2. Key Capabilities

- **Multi-Pipeline Audio Analysis:** Upload any standard audio file (`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`) for automatic sample-rate normalization (16 kHz mono) and multi-model parallel inference.
- **Dynamic Mel-Spectrogram Visualization:** Interactive HTML5 Canvas rendering time-frequency energy distributions in real time.
- **Model Feature Importance Transparency:** Explores the top Gini feature importances of the classical model (e.g., *Spectral Rolloff Mean*, *Spectral Contrast Band 1*).
- **Five-State Majority-Voting Consensus:** Replaces ambiguous threshold cutoffs with an operational consensus policy:
  - `AI-GENERATED` (Unanimous 3/3 AI)
  - `LIKELY AI-GENERATED` (Majority 2/3 AI)
  - `LIKELY HUMAN-CREATED` (Majority 2/3 Human)
  - `HUMAN-CREATED` (Unanimous 3/3 Human)
  - `UNABLE TO DETERMINE` (Fewer than 2 valid model outputs)
- **Disagreement & Uncertainty Diagnostics:** Reports mean predicted probability, inter-model variance, and surfaces qualitative confidence badges.
- **Complete Test & Verification Suite:** 18 automated unit and integration tests covering consensus logic, feature extraction, and API endpoints.

---

## 3. System Architecture

```
                    ┌────────────────────────────────────────┐
                    │          Input Audio File              │
                    │   (.wav, .mp3, .flac, .m4a, .ogg)      │
                    └───────────────────┬────────────────────┘
                                        │
                         [ AudioPreprocessor (16 kHz) ]
                                        │
            ┌───────────────────────────┼───────────────────────────┐
            │                           │                           │
   [ Window: First 10s ]       [ Window: First 10s ]       [ Window: 60s Chunks ]
            │                           │                           │
            ▼                           ▼                           ▼
  FeatureExtractor (Set B)    MelSpectrogramExtractor           AST Feature
  22 Spectral Descriptors     (128 Mels, Hop=512)            Extractor (16 kHz)
            │                           │                           │
            ▼                           ▼                           ▼
 ┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
 │  Method 1: Classical │    │  Method 2: ResNet-18 │    │  Method 3: AST Plan B│
 │    Random Forest     │    │  5-LOGO Ensemble     │    │  Audio Spectrogram   │
 │   (100 Estimators)   │    │  (Mel Spectrogram)   │    │     Transformer      │
 └──────────┬───────────┘    └──────────┬───────────┘    └──────────┬───────────┘
            │                           │                           │
            │ P(AI)                     │ P(AI)                     │ P(AI)
            └───────────────────────────┼───────────────────────────┘
                                        │
                                        ▼
                     ┌─────────────────────────────────────┐
                     │   Consensus Decision Engine         │
                     │   Majority Voting (Threshold = 0.5) │
                     └──────────────────┬──────────────────┘
                                        │
              ┌─────────────────────────┴─────────────────────────┐
              ▼                                                   ▼
     [ Final Classification ]                             [ Diagnostic Metrics ]
     • AI-GENERATED (3/3)                                 • Mean Probability
     • LIKELY AI-GENERATED (2/3)                          • Model Variance
     • LIKELY HUMAN-CREATED (2/3)                         • Disagreement Flag
     • HUMAN-CREATED (3/3)                                • Feature Importances
```

---

## 4. Methodology & Feature Engineering

### 4.1 Audio Preprocessing
All input audio is decoded and standardized using `librosa` and `soundfile` to 16,000 Hz single-channel (mono) floating-point audio. For the Classical and ResNet-18 pipelines, analysis is performed on the first 10 seconds of audio ($160,000$ samples) with automated zero-padding for shorter clips. For the AST model, sliding 60-second windows are evaluated across the full duration.

### 4.2 Feature Sets Evaluated
Five distinct acoustic feature sets were systematically developed and benchmarked:

| Feature Set | Group Description | Dimension | Components |
| :---: | :--- | :---: | :--- |
| **Set A** | MFCC Only | 40 | 20 MFCC means + 20 MFCC standard deviations |
| **Set B** | Spectral Only *(Deployed)* | **22** | Centroid (2), Bandwidth (2), Rolloff (2), Flatness (2), 7-Band Contrast (14) |
| **Set C** | MFCC + Spectral | 62 | Combined MFCCs and Spectral Descriptors |
| **Set D** | MFCC + Spectral + Harmonic | 88 | Set C + Chroma STFT (24) + Harmonic-Percussive Means (2) |
| **Set E** | All Features | 96 | Set D + RMS Energy, Onset Strength, and Tempo Features |

The production Classical model utilizes **Feature Set B (22 spectral features)**. Empirical analysis demonstrated that higher-dimensional sets (Sets D & E) were susceptible to acoustic overfitting on specific musical styles, whereas spectral shape features captured universal frequency distribution irregularities common to generative diffusion vocoders.

---

## 5. Research Results & Generalization Findings

### 5.1 Pilot 01 In-Domain Benchmark vs. Deployed Model

To maintain complete scientific honesty, HarmonyShield explicitly differentiates exploratory pilot benchmarks from the production deployed model:

- **Pilot 01 Benchmark (AudioLDM2 In-Domain):**
  - Evaluated on 100 human and 100 AudioLDM2 AI samples with strict grouped 80/20 train-test splits by YouTube ID to prevent leakage.
  - *Best exploratory model:* Logistic Regression on Spectral Features.
  - *Metrics:* **Accuracy: 97.5%**, **F1-Score: 0.9756**, **ROC-AUC: 0.9975**.
  - *Interpretation:* Confirms high discriminative capacity when testing on the *same* generator architecture seen during training.
- **Production Deployed Classical Model:**
  - Random Forest classifier trained on Feature Set B (22 spectral features, 100 trees).
  - Selected for production deployment to provide interpretable Gini feature importance rankings on the live dashboard.

### 5.2 Leave-One-Generator-Out (LOGO) Generalization

To test whether models generalize to completely unseen AI music generators, both Classical ML and ResNet-18 models were subjected to rigorous **Leave-One-Generator-Out (LOGO)** cross-validation across 5 open architectures (AudioLDM2, Stable Audio Open, MusicLDM, Mustango, MusicGen Medium):

| Evaluation Metric | Classical ML (LOGO) | Deep Learning ResNet-18 (LOGO) |
| :--- | :---: | :---: |
| **Mean LOGO F1-Score** | $0.7092 \pm 0.0367$ | $0.7246 \pm 0.0514$ |
| **Mean LOGO ROC-AUC** | $0.7152$ | $0.7145$ |

#### Generator-Specific LOGO Results (Classical ML):
- **MusicGen Medium (Unseen):** F1 = `0.7547` (Accuracy: 0.6750, ROC-AUC: 0.7963)
- **MusicLDM (Unseen):** F1 = `0.7407` (Accuracy: 0.6500, ROC-AUC: 0.7550)
- **Mustango (Unseen):** F1 = `0.7170` (Accuracy: 0.6250, ROC-AUC: 0.7212)
- **Stable Audio Open (Unseen):** F1 = `0.6667` (Accuracy: 0.6000, ROC-AUC: 0.6812)
- **AudioLDM2 (Unseen):** F1 = `0.6667` (Accuracy: 0.6000, ROC-AUC: 0.6225)

### 5.3 Research Findings & Scientific Limitations

1. **The Generalization Ceiling:** The performance drop from near-perfect in-domain scores ($F_1 > 0.97$) to cross-generator scores ($F_1 \approx 0.71$) demonstrates that models heavily exploit generator-specific acoustic artifacts rather than a universal invariant signature of synthetic audio.
2. **Correlation vs. Causation:** Feature importance metrics indicate statistical correlation with training labels, not causal physical acoustic defects.
3. **Audio Coverage Discrepancies:** The Classical and ResNet-18 models analyze the first 10 seconds of audio, whereas AST evaluates sliding 60-second windows, which can occasionally lead to justifiable inter-model disagreement.
4. **Majority Voting Policy:** Majority voting represents an operational consensus decision rule designed to balance precision and recall, not absolute forensic proof.

---

## 6. Installation and Usage

### Prerequisites
- Python 3.10 to 3.13
- Node.js 18+ and npm
- Git

### 6.1 Clone & Environment Setup
```bash
# Clone the repository
git clone -b review-ready https://github.com/ParamBande/HarmonyShiled.git
cd HarmonyShiled

# Create and activate Python virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 6.2 Retrieve & Verify Model Checkpoints
```bash
# Automatically download the 5 ResNet18 checkpoints from GitHub Releases
python scripts/download_checkpoints.py

# Verify SHA-256 checksums and file sizes
python scripts/verify_models.py
```

### 6.3 Launch the Application
Start the backend and frontend in two separate terminals:

```bash
# Terminal 1: FastAPI Backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: React Frontend
cd frontend
npm install
npm run dev
```

Open your browser to: **`http://localhost:5173`**.

### 6.4 Run Automated Tests
```bash
pytest tests/
```
All 18 tests will verify consensus policies, audio preprocessing, and API endpoints.

---

## 7. Model Manifest & Asset Integrity

| Model | Path | Size | SHA-256 Checksum | Source |
| :--- | :--- | :---: | :--- | :--- |
| **Random Forest** | `data/pilot_dataset/best_rf_model.joblib` | 364 KB | `36a6dc3f1f6e0a9c...bbe25c1dd9cc0b7f` | Git Repo |
| **ResNet18 (AudioLDM2)** | `experiments/.../LOGO_audioldm2_model.pt` | 42.7 MB | `6d6c2dbb1032ae3c...cad7763d195c251e` | Release Asset |
| **ResNet18 (MusicGen)** | `experiments/.../LOGO_MusicGen_medium_model.pt` | 42.7 MB | `fba9c0ca92ba1801...2da3c34ae3c885a1` | Release Asset |
| **ResNet18 (MusicLDM)** | `experiments/.../LOGO_musicldm_model.pt` | 42.7 MB | `27d3cd74adf6c74e...643c07957e5840ee` | Release Asset |
| **ResNet18 (Mustango)** | `experiments/.../LOGO_mustango_model.pt` | 42.7 MB | `6c21078d84e012e9...93bd02306785d8fa` | Release Asset |
| **ResNet18 (StableAudio)** | `experiments/.../LOGO_stable_audio_open_model.pt` | 42.7 MB | `d7fa7833d9150324...81595139503f016b` | Release Asset |
| **AST Detector (Plan B)** | Auto-cached by Hugging Face | ~340 MB | Managed by Transformers | Hugging Face Hub |

---

## 8. Team Handover & Reproducibility

For a complete step-by-step recovery guide designed for non-technical teammates or emergency laptop recovery, refer to:
👉 **[TEAM_HANDOVER.md](TEAM_HANDOVER.md)**

---

## 9. Citations & References

If you build upon this project or use our experimental benchmarks, please cite:

```bibtex
@software{bande2026harmonyshield,
  author = {Bande, Param},
  title = {HarmonyShield: Machine Learning-Based Audio Forensics for Detection of AI-Generated Music},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/ParamBande/HarmonyShiled}
}
```

### Acknowledgments & Libraries
- **MusicCaps:** Agostinelli et al., Google Research (2023).
- **FakeMusicCaps:** Synthetic audio generation benchmarks across open diffusion models.
- **AST Detector:** Gong et al., *AST: Audio Spectrogram Transformer*, Interspeech 2021. Pretrained weights provided via Hugging Face (`AI-Music-Detection`).
- **Core Libraries:** PyTorch, Librosa, Scikit-Learn, FastAPI, React, Vite, Tailwind CSS.

---

## 10. License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
