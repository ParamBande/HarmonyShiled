# Plan A Scientific Pilot 01 Report

## Dataset Composition
- **Human Samples:** 100
- **AI Samples:** 100 (generator: audioldm2)
- **Paired `ytid` Groups:** 100
- **Splits:** 80 Training Groups (160 samples), 20 Test Groups (40 samples). GroupShuffleSplit ensured no `ytid` leakage.

## Results & Comparison
We evaluated feature sets A (MFCC), B (Spectral), C (MFCC+Spectral), D (MFCC+Spectral+Harmonic), E (All).

- **Best Pilot Result:** Feature Set B (Spectral only) using Logistic Regression achieved **F1 0.9756** and **ROC-AUC 0.9975**.
- **Feature-Set Comparison:** Spectral features dramatically outperformed all others. Sets B, C, D, and E all hovered >0.90 F1 with RF/LR/SVM, while Set A (MFCC only) struggled at ~0.85 F1 max.
- **Model Comparison:** Logistic Regression and Random Forest dominated. Simple linear separation works exceptionally well on spectral features.

## Interpretability
**Top Important Features (Association, Not Causation):**
Using the strongest Random Forest model (F1: 0.952), the primary predictors were:
1. `spectral_contrast_6_std` (0.1375)
2. `spectral_contrast_7_mean` (0.1027)
3. `spectral_contrast_7_std` (0.1009)
4. `spectral_contrast_2_std` (0.0919)
5. `spectral_contrast_3_std` (0.0565)
6. `spectral_contrast_5_std` (0.0465)
7. `spectral_contrast_5_mean` (0.0424)

*Observation:* AudioLDM2 leaves distinct spectral contrast artifacts across multiple bands that cleanly separate it from human audio.

## Quality & Execution
- **Leakage/Data-Quality Issues:** None observed during training. The strict 10s trimming and GroupShuffleSplit guarantee held.
- **Processing Time:** ~6 minutes for feature extraction and ML pipeline execution.
- **Storage Used:** ~24 MB for audio pairs + ~2 MB for extracted features and models.
- **Confusion Matrix Location:** Available directly within `pilot_results.csv` (CM column) under the dataset directory.

## Artifacts Location
- **Dataset / Audio:** `data/pilot_dataset/`
- **Features CSVs:** `data/pilot_dataset/features_*.csv`
- **Results CSV:** `data/pilot_dataset/pilot_results.csv`
- **Best Model:** `data/pilot_dataset/best_rf_model.joblib`

## Recommended Next Milestone
Expand the classical ML evaluation (specifically Logistic Regression and Random Forest on Spectral Features) across **all 5 AI generators** (MusicGen, AudioGen, Riffusion, Mustango, AudioLDM2) using a "leave-one-generator-out" validation scheme to test if these spectral contrast artifacts generalize to other models, or if they are unique to AudioLDM2.
