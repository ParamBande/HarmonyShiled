# Plan B Pretrained Model Notes

## 1. AI-Music-Detection/ai_music_detection_large_60s

### Published/model-card facts
- **Model Identifier**: `AI-Music-Detection/ai_music_detection_large_60s`
- **URL**: [Hugging Face](https://huggingface.co/AI-Music-Detection/ai_music_detection_large_60s)
- **Model Architecture**: Audio Spectrogram Transformer (AST) based on `MIT/ast-finetuned-audioset-10-10-0.4593`
- **Stated Training Dataset**: `SleepyJesse/ai_music_large`
- **Stated Training Procedure**: Fine-tuned using Hugging Face Transformers
- **Input Duration**: up to 60 seconds
- **Input Sample Rate**: 16000 Hz
- **Preprocessing**: Requires `AutoFeatureExtractor` from the model ID
- **Output Labels**: Used for classifying music as AI-generated or human-composed
- **Known Limitations**: Unknown generalization to novel AI generation architectures, codecs, and mixing conditions.
- **Model Size**: ~90.8M parameters
- **License**: Refer to the Hugging Face repository
- **Date Accessed**: 2026-09-30

### Our observations
- The model successfully downloads and loads within the 4GB VRAM constraint (or on CPU).
- Hugging Face interface is straightforward to adapt for long-file sliding-window inference.
- Probabilities reflect window-level confidence which we aggregate by simple averaging. 

### Our future experiments
- Test accuracy on our own evaluation dataset once collected.
- Compare calibration of probabilities.
- Evaluate robustness against MP3 compression artifacts and audio filtering.
- Compare with Plan A classical ML models (SVM, Random Forest, etc.) over MFCC features.

## 2. lofcz/ai-music-detector (Lightweight Fallback)

### Published/model-card facts
- **Model Identifier**: `lofcz/ai-music-detector`
- **URL**: [Hugging Face](https://huggingface.co/lofcz/ai-music-detector), [GitHub](https://github.com/lofcz/ai-music-detector)
- **Model Architecture**: Logistic Regression over custom features
- **Stated Training Dataset**: N/A
- **License**: Refer to repository

### Our observations
- Relies heavily on custom scikit-learn pipelines and specific GitHub repo structures that aren't available as a standalone pip package.
- Implementation deferred to avoid polluting the core prototype with complex dependencies, prioritizing AST.

### Our future experiments
- Potentially rewrite the feature extraction from scratch in Plan A instead of adapting this specific repo, since Plan A explicitly aims to investigate Logistic Regression, SVM, and classical ML features.
