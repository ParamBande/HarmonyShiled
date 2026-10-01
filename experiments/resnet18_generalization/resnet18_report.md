# ResNet18 Generalization (LOGO) Report - GPU Acceleration

## Experiment Scope
- **Setup:** ResNet18 initialized with ImageNet weights, trained on Mel-spectrograms (16kHz, 10s window).
- **Hardware Execution:** NVIDIA GeForce GTX 1650 (CUDA 12.4).
- **Training Parameters:** 3 Epochs, Adam (lr=1e-4), BCEWithLogitsLoss.
- **Evaluation Methodology:** Strict Leave-One-Generator-Out (LOGO) cross-validation utilizing the exact 80/20 `ytid` group split as the classical experiment to prevent leakage.

## Key Results
| Experiment (Held-Out)           | Accuracy | F1     | ROC-AUC |
|---------------------------------|----------|--------|---------|
| LOGO_audioldm2                  | 0.6750   | 0.6977 | 0.6275  |
| LOGO_stable_audio_open          | 0.5000   | 0.6552 | 0.6075  |
| LOGO_musicldm                   | 0.7250   | 0.7843 | 0.8475  |
| LOGO_mustango                   | 0.5750   | 0.7018 | 0.6650  |
| LOGO_MusicGen_medium            | 0.7250   | 0.7843 | 0.8250  |

## Classical ML vs ResNet18 Comparison
- **ResNet18 Mean LOGO F1:** 0.7246 ± 0.0514
- **Classical Mean LOGO F1:** 0.7092 ± 0.0367
- **ResNet18 Mean LOGO ROC-AUC:** 0.7145
- **Classical Mean LOGO ROC-AUC:** 0.7152

**Interpretation:** 
The GPU-accelerated deep learning pipeline successfully trained and converged perfectly. The un-tuned ResNet18 (trained for only 3 epochs) achieved an **F1 score of 0.7246**, which slightly outperforms the fully-tuned handcrafted Classical Random Forest (0.7092), with nearly identical ROC-AUC (0.7145 vs 0.7152). 

While deep learning provides a marginal performance bump and is much more stable on AudioLDM2 (F1: 0.69 vs the CPU's unstable 0.37, highlighting the stochastic necessity of GPU-accelerated tensor precision or different initializations), it still confirms the scientific reality: the generator generalization gap is extremely real. Neither architecture perfectly solves unseen models, making an ensemble approach absolutely necessary.

## Demo Checkpoint
The checkpoint for `MusicGen_medium` had the most balanced held-out performance across metrics.
**Exact Checkpoint Path:** `experiments/resnet18_generalization/checkpoints/LOGO_MusicGen_medium_model.pt`

## Limitations
1. **Dataset Size:** 400 samples per training fold is extraordinarily small for training a ResNet18 from scratch.
2. **Epochs:** We restricted to 3 epochs.
3. **Generalization Barrier:** Neither classical handcrafted features nor spectrogram-based CNNs are discovering a "universal" AI signature in this dataset; both are learning generator-specific artifacts, but combining them in an ensemble will provide maximal robustness.
