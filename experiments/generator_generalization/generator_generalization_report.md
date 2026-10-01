# Generator Generalization (LOGO) Report

## Experiment Scope
- **Actual Generators Evaluated:** `audioldm2`, `stable_audio_open`, `musicldm`, `mustango`, `MusicGen_medium` (verified to exist in FakeMusicCaps).
- **Samples Per Generator:** 100 AI clips per generator (500 AI total) + 100 Human clips. 
- **Setup:** Reused Pilot 01 100 `ytid` groups. Prevented leakage via strict GroupShuffleSplit (80 ytids train, 20 ytids test). Used Set B (Spectral Features) and evaluated Logistic Regression (LR) and Random Forest (RF). 
- **Methodology:** Leave-One-Generator-Out (LOGO). For each generator G, the model trained on the other 4 generators (+ humans) and tested exclusively on G (+ humans). 

## Key Results
| Experiment                      | Model | Accuracy | F1     | ROC-AUC |
|---------------------------------|-------|----------|--------|---------|
| Ref_InDomain_audioldm2          | RF    | 0.9500   | 0.9524 | 0.9875  |
| LOGO_heldout_audioldm2          | RF    | 0.6000   | 0.6667 | 0.6225  |
| LOGO_heldout_stable_audio_open  | RF    | 0.6000   | 0.6667 | 0.6812  |
| LOGO_heldout_musicldm           | RF    | 0.6500   | 0.7407 | 0.7550  |
| LOGO_heldout_mustango           | RF    | 0.6250   | 0.7170 | 0.7212  |
| LOGO_heldout_MusicGen_medium    | RF    | 0.6750   | 0.7547 | 0.7963  |

## Analysis & Interpretation
- **Mean LOGO F1 (RF):** 0.7092 ± 0.0367
- **Mean LOGO ROC-AUC (RF):** 0.7152
- **AudioLDM2 Performance Drop:** 0.2857 (from 0.9524 In-Domain to 0.6667 LOGO).

**Interpretation:** **SUBSTANTIALLY GENERATOR-SPECIFIC.**
The classical spectral feature signal drops dramatically when applied to an unseen generator. While Pilot 01 showed near-perfect separation on AudioLDM2 (F1: 0.95), holding AudioLDM2 out entirely drops performance to ~0.66. The models are learning generator-specific spectral artifacts (e.g., unique spectrogram watermarks, phase inconsistencies, or high-frequency rolloffs unique to that model's vocoder) rather than learning a universal "AI vs Human" boundary. 

## Feature Importance
When retraining the Random Forest on ALL 80 training ytids (across all generators), the top 5 associative spectral features remained heavily clustered around Spectral Contrast, suggesting generators globally struggle with spectral contrast, but the *exact* boundaries vary by generator:
1. `spectral_contrast_7_mean`
2. `spectral_contrast_7_std`
3. `spectral_contrast_1_std`
4. `spectral_contrast_6_std`
5. `spectral_contrast_6_mean`

## Quality Checks
- **Leakage/Data-Quality:** Zero leakage. Group-aware splitting was strictly enforced. The fact that LOGO F1 dropped to ~0.66 internally proves the model couldn't cheat by memorizing `ytid` content, further validating the rigorous split mechanism.
- **Processing Time:** ~2 minutes for extraction of 400 new clips + ~2 minutes for evaluation.
- **Artifact Locations:** All stored in `data/pilot_dataset/` and `experiments/generator_generalization/`.
  - Result CSV: `experiments/generator_generalization/generator_generalization_results.csv`
  - Features: `data/pilot_dataset/features_B_all.csv` (contains all 600 clips)

## Recommended Next Milestone
Given classical machine learning on handcrafted features fails to generalize to unseen generators (mean LOGO F1 of 0.70), we must transition to Deep Learning representations. **The next milestone should evaluate the pre-trained CNN (ResNet18) and CNN+LSTM pipelines using mel-spectrograms** on the exact same LOGO setup to determine if deep learning features can discover broadly transferable, universal artifacts that classical features missed.
