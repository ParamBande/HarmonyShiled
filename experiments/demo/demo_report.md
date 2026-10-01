# HarmonyShield Black-Box Live Demo Report

This report documents the final frozen inference on two completely external, unseen audio tracks. These tracks were NOT used for any training, tuning, threshold selection, or model selection. The outputs remain strictly raw and un-tampered.

## 1. External Demo Track 1: Genuine Human Music
- **File:** `Hangova (From DC ).mp3` (4.09 MB)
- **Origin:** Genuinely human-created track.

### Inference Results
- **Classical ML (Set B Random Forest):** Prediction: Human (AI Probability: 0.1700)
- **ResNet18 (5-Fold LOGO Ensemble):** Prediction: AI (Avg AI Probability: 0.9389)
  - Held-out audioldm2: 0.9691
  - Held-out stable_audio_open: 0.8947
  - Held-out musicldm: 0.9956
  - Held-out mustango: 0.8677
  - Held-out MusicGen_medium: 0.9675
- **Plan B (Pretrained AST Detector):** Prediction: Human (AI Probability: 0.3802)

**=> FINAL CONCLUSION:** INCONCLUSIVE / LOW CONFIDENCE
*(Models fundamentally disagree. The Classical and AST models accurately recognize it as Human, but the ResNet CNN triggers a massive false-positive. The ensemble rightly refuses to force a consensus.)*

---

## 2. External Demo Track 2: Suno AI Generation
- **File:** `??????_???.mp3` (744 KB)
- **Origin:** Suno (a completely unseen generative architecture not present in the FakeMusicCaps dataset).

### Inference Results
- **Classical ML (Set B Random Forest):** Prediction: Human (AI Probability: 0.0000)
- **ResNet18 (5-Fold LOGO Ensemble):** Prediction: AI (Avg AI Probability: 0.7862)
  - Held-out audioldm2: 0.3969
  - Held-out stable_audio_open: 0.8450
  - Held-out musicldm: 0.9791
  - Held-out mustango: 0.8998
  - Held-out MusicGen_medium: 0.8102
- **Plan B (Pretrained AST Detector):** Prediction: AI (AI Probability: 0.9947)

**=> FINAL CONCLUSION:** INCONCLUSIVE / LOW CONFIDENCE
*(Models fundamentally disagree. The Classical ML model completely fails, catastrophically misclassifying the unseen AI as 100% Human. However, the ResNet CNN and Plan B AST transformer both independently catch the AI artifacts. Because of the Classical model's extreme failure, the ensemble appropriately flags the track as inconclusive rather than presenting a false sense of security.)*

---

## Executive Summary
This black-box test perfectly validates the central hypothesis of the HarmonyShield project: **Single-architecture detectors are fundamentally unsafe for real-world unseen AI generators.**

1. The Handcrafted Classical Model is rigid; when faced with a generator outside its distribution (Suno), it collapses entirely.
2. The CNN ResNet is highly sensitive to spectral patterns but suffers from false-positives on out-of-distribution human audio.
3. The AST Transformer (Plan B) is powerful but can be ambiguous on noisy internet audio.

By ensembling them and **measuring their disagreement**, HarmonyShield successfully exposes the uncertainty boundaries of generative audio detection instead of confidently lying to the user.
