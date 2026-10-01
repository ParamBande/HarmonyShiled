import torch
import numpy as np
from src.audio.preprocessor import AudioPreprocessor
from src.features.melspectrogram import MelSpectrogramExtractor
from src.models.resnet18_model import AudioResNet18

class ResNetInference:
    """
    Handles end-to-end inference for the ResNet18 pipeline.
    """
    def __init__(self, model_path, segment_duration=10.0, target_sr=16000):
        self.preprocessor = AudioPreprocessor(target_sr=target_sr, segment_duration=segment_duration)
        self.extractor = MelSpectrogramExtractor(sr=target_sr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = AudioResNet18(pretrained=False)
        # Load weights safely
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, audio_path):
        segments, metadata = self.preprocessor.process(audio_path)
        
        if not segments:
            raise ValueError(f"No valid segments extracted from {audio_path}")
            
        segment_probs = []
        
        with torch.no_grad():
            for seg in segments:
                mel = self.extractor.extract(seg)
                mel = mel.unsqueeze(0).to(self.device) # Add batch dim: (1, 1, n_mels, time)
                
                logits = self.model(mel)
                prob = torch.sigmoid(logits).item()
                segment_probs.append(prob)
                
        # Aggregate
        mean_prob = np.mean(segment_probs)
        final_pred = 1 if mean_prob > 0.5 else 0
        
        return {
            "prediction": final_pred,
            "probabilities": [1.0 - mean_prob, mean_prob], # [Human, AI]
            "metadata": metadata,
            "segment_predictions": segment_probs
        }
