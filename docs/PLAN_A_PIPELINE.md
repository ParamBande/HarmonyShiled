# HarmonyShield Plan A Pipeline

This document explains the architecture and design of the classical Machine Learning pipeline (Plan A) for HarmonyShield.

## Architecture & CPU-First Design
Plan A is designed to be highly efficient, interpretable, and run entirely on the CPU without requiring deep learning frameworks like PyTorch or expensive GPUs. It leverages established audio signal processing (via `librosa`) and robust classical machine learning algorithms (via `scikit-learn`).

## Preprocessing
The pipeline standardizes all incoming audio:
1. Loads audio (supporting various formats).
2. Converts to Mono.
3. Resamples to 22.05 kHz.
4. Normalizes amplitude to prevent scaling artifacts.
5. Divides the track into non-overlapping 10-second segments. If an audio file is shorter than 10 seconds, it is padded with zeros.

## Feature Extraction & Families
Each 10-second segment is reduced to a fixed-length numerical feature vector summarizing its characteristics. We use the following feature families (extracting the mean and standard deviation across the segment's frames):
- **MFCC (Mel-Frequency Cepstral Coefficients):** 20 coefficients summarizing the spectral envelope.
- **Spectral:** Centroid, bandwidth, rolloff, flatness, and contrast.
- **Harmonic:** Chroma features and harmonic/percussive energy ratio.
- **Temporal & Energy:** RMS energy, zero-crossing rate (ZCR), low-energy ratio, onset envelope, and tempo.

## Feature Sets A-E
The pipeline supports ablation studies through named configurations:
- **A:** MFCC only
- **B:** Spectral only
- **C:** MFCC + Spectral
- **D:** MFCC + Spectral + Harmonic
- **E:** MFCC + Spectral + Harmonic + Temporal

## Leakage Prevention
Dataset splitting strictly employs a group-wise strategy (`GroupShuffleSplit`). By grouping segments by their source `track_id` or `group_id`, we guarantee that no segment from a track in the training set accidentally bleeds into the validation or test sets. The default split is Train: 70%, Validation: 15%, Test: 15%.

## Model Families
The pipeline currently supports multiple classical algorithms suitable for our extracted features:
- Logistic Regression (LR)
- Support Vector Machines (SVM)
- Random Forest (RF)
- Decision Tree (DT)
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes (GNB)
- Histogram-based Gradient Boosting (GBDT)

We implement scaling (StandardScaler) automatically within an sklearn `Pipeline` to ensure robust convergence. Random Forest is the default due to its capability for non-linear relationships and inherent feature importance reporting.

## Song-level Inference
Inference runs iteratively:
1. The track is divided into 10-second segments.
2. Features are extracted per segment.
3. The model predicts probabilities per segment.
4. Segment-level probabilities are aggregated (e.g., via mean) to produce the final track-level prediction. 
Alternative strategies include aggregating features across segments before executing a single prediction.

## Evaluation Metrics
When real labelled data becomes available (e.g., FakeMusicCaps), models will be evaluated using:
- Accuracy
- Precision (Macro)
- Recall (Macro)
- F1-Score (Macro)
- ROC-AUC (where class probabilities are supported)
- Confusion Matrix (saved as plot and JSON)

*Note: All feature importance reports will explicitly note that the importance scores indicate model association, not biological or causal truth.*
