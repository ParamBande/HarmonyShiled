# HarmonyShield Live Demonstration Protocol

This document defines the strict requirements, cases, and criteria for the final HarmonyShield real-world live demonstration.

## A. Scientific Rules

1. **Zero Leakage:** Demo inputs must NEVER be used for training, hyperparameter tuning, feature selection, threshold calibration, or model selection.
2. **Black-Box Testing:** The demo constitutes a genuine black-box external test representing real-world usage. 
3. **External Generator Evaluation:** Suno is not included in the FakeMusicCaps training dataset (which only contains AudioLDM2, Stable Audio Open, MusicLDM, Mustango, and MusicGen). Therefore, a Suno test explicitly evaluates cross-architecture generalizability to an external state-of-the-art generator.
4. **External Human Evaluation:** A random human song constitutes a completely external out-of-distribution human test, independent of the MusicCaps training distribution.
5. **No Guarantees:** We must never guarantee that an arbitrary external song will be classified correctly. The system is empirical and must reflect realistic limitations.
6. **Uncertainty Reporting:** The system must not force a binary prediction when confidence is low. If probabilities hover around 50% or if different models disagree heavily, the system must explicitly report "Inconclusive / Low confidence."

## B. Demo Cases

The final demo requires two untouched, external source files:

1. **DEMO A (AI MUSIC):** A completely AI-generated song from an external state-of-the-art generator (e.g., Suno). Downloaded and uploaded untouched.
2. **DEMO B (HUMAN MUSIC):** A genuinely human-created song from an external source, verifiably absent from any HarmonyShield training or validation datasets. Uploaded untouched.

## C. Required Result Display

For every input file submitted to the system, the final demo output must display:

1. **Final Ensemble Prediction** (AI, Human, or Inconclusive)
2. **Overall Confidence/Probability**
3. **Classical ML Result** (Prediction & Probability)
4. **ResNet18 Result** (Prediction & Probability)
5. **Plan B Result** (Prediction & Probability)
6. **Ensemble Agreement** (Whether the three independent pipelines agree or disagree)
7. **Segment-Level Predictions** (Time-series probabilities across the song)
8. **Explainability Information** (e.g., highlighting classical feature anomalies like spectral contrast)

## D. Controlled Validation vs. Live Demo

- **Scientific Validation Datasets:** Used exclusively by developers during building to benchmark F1, accuracy, and LOGO generalization.
- **Live External Demonstration Inputs:** Handled exclusively by the inference pipeline at the end of the project. Absolutely no overlap or cross-contamination between validation datasets and live demo inputs is permitted.

## E. Demo Readiness Criteria

HarmonyShield will be considered "Demo Ready" only when the following conditions are met:

1. **Frozen Models:** All three underlying models (Classical ML, ResNet18, Plan B) are fully trained, frozen, and serialized. No further parameter updates are possible.
2. **No Leakage:** Demo audio files have never been processed by any script outside of the read-only inference pipeline.
3. **Reproducible Inference:** The exact inference code (`predict.py` or similar) can ingest a raw audio file and deterministically output predictions without requiring any developer intervention.
4. **Uncertainty Handling:** Thresholds for "Inconclusive" predictions are defined and implemented in the aggregation logic.
5. **No Hardcoding:** Absolutely no hard-coded demo answers. The prediction must be dynamically generated from the raw audio via the actual inference pipeline.
6. **Successful Execution:** Both Demo A and Demo B inputs are successfully processed from end-to-end without crashing.
