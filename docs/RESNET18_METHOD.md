# HarmonyShield: Deep Learning Pipeline (ResNet18)

## 1. Why Mel-Spectrograms?
Audio waveforms are high-dimensional and complex. A Mel-spectrogram represents audio in the frequency domain, scaled logarithmically (both in frequency and amplitude) to match human auditory perception. For generative AI music detection, Mel-spectrograms visually expose artifacts such as unnatural high-frequency roll-offs, phase distortions, and harmonic inconsistencies that might be difficult to capture in purely handcrafted classical features.

## 2. Why CNNs / ResNet?
Convolutional Neural Networks (CNNs) excel at discovering hierarchical spatial patterns in 2D data. By treating the Mel-spectrogram as a 2D image (Time x Frequency), a CNN can learn localized textural anomalies (e.g., a "smudged" transient) in early layers, and semantic structural anomalies (e.g., repetitive, synthetic looping) in deeper layers.

ResNet18 specifically provides an excellent balance:
- **Lightweight:** With only ~11 million parameters, it comfortably fits on consumer GPUs (e.g., GTX 1650 4GB VRAM).
- **Residual Connections:** Prevents vanishing gradients, allowing deep feature extraction without requiring massive datasets to converge.

## 3. Transfer Learning Approach
To overcome our relatively small dataset size (600 clips), we utilize transfer learning from ImageNet:
- The model is initialized with `ResNet18_Weights.DEFAULT`.
- Although ImageNet consists of natural images (RGB), the lower-level Gabor filters learned by the network (edge detectors, texture detectors) are universally highly effective for finding textures in spectrograms.
- The 1-channel Mel-spectrogram is repeated across 3 channels to interface with the pre-trained first convolutional layer.
- The final fully connected layer is replaced with a single neuron (binary classification) optimized via Binary Cross Entropy (`BCEWithLogitsLoss`).

## 4. What the Model Learns Conceptually
Unlike the classical ML pipeline which explicitly calculates global statistics (mean/std of spectral contrast, MFCCs, flatness), the ResNet:
- Learns the *local distribution* of energy across time and frequency.
- Can detect localized generator "watermarks" (specific frequency bands where the vocoder struggles to reconstruct phase).
- Potentially discovers broadly transferable features (e.g., how AI models handle drum transients vs. sustained chords) that classical global averages smooth over.

## 5. Differences from Handcrafted Pipeline
- **Representation:** Handcrafted (Global scalar statistics) vs. Deep Learning (Dense 2D time-frequency grids).
- **Flexibility:** The CNN learns the optimal filters directly from the data, whereas classical ML relies on predetermined human audio engineering assumptions.
- **Compute:** CNNs require significantly more compute for inference and training, but can capture more complex non-linear relationships.

## 6. Limitations
- CNNs can overfit to specific spectrogram "textures" (e.g., MP3 compression artifacts) if the dataset is not carefully controlled.
- ResNet18 lacks a recurrent mechanism to explicitly model long-term temporal dependencies across the entire 10-second sequence (which is why a CNN+LSTM is sometimes preferred for music).
