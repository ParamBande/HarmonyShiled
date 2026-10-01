import time
import torch
import librosa
import numpy as np
from typing import Dict, Any
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import warnings
from .detector_interface import DetectorInterface

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

class ASTDetector(DetectorInterface):
    def __init__(self, model_id: str = "AI-Music-Detection/ai_music_detection_large_60s"):
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model {model_id} on {self.device}...")
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
        except Exception as e:
            print(f"Preprocessor config not found in {self.model_id}. Falling back to MIT/ast-finetuned-audioset-10-10-0.4593 feature extractor...")
            self.feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
            
        self.model = AutoModelForAudioClassification.from_pretrained(self.model_id).to(self.device)
        self.model.eval()
        
        # Override feature extractor max_length to match model's expected positional embedding length
        max_length = getattr(self.model.config, "max_length", 6000)
        self.feature_extractor.max_length = max_length
        self.feature_extractor.n_frames = max_length # sometimes AST relies on n_frames

        
        # Determine sampling rate from feature extractor
        self.target_sr = getattr(self.feature_extractor, "sampling_rate", 16000)
        # AST 60s model typically expects 60s max, but let's extract exactly the length it prefers
        # from the config if available, otherwise default to 60.
        # Actually AST can take arbitrary length (with max_length truncation), but let's chunk to 60s windows
        self.window_duration = 60.0 # seconds

    def predict(self, audio_path: str) -> Dict[str, Any]:
        start_time = time.time()
        
        # Load audio using librosa (handles WAV, MP3, FLAC, M4A and converts to mono)
        audio, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
        audio_duration = len(audio) / sr
        
        # Segment into windows
        window_samples = int(self.window_duration * sr)
        
        # If audio is very short, pad it slightly or just pass it
        if len(audio) == 0:
            raise ValueError("Audio file is empty or could not be loaded.")
            
        chunks = []
        for i in range(0, len(audio), window_samples):
            chunk = audio[i:i + window_samples]
            # Ensure minimum chunk length if needed, but feature extractor usually handles short arrays by padding
            chunks.append(chunk)
            
        ai_probs = []
        human_probs = []
        
        with torch.no_grad():
            for chunk in chunks:
                # Prepare inputs
                # Extract max_length from model config, default to 6000 for 60s AST
                max_length = getattr(self.model.config, "max_length", 6000)
                
                inputs = self.feature_extractor(
                    chunk, 
                    sampling_rate=sr, 
                    return_tensors="pt", 
                    padding="max_length",
                    truncation=True,
                    max_length=max_length
                )
                
                input_values = inputs.input_values.to(self.device)
                outputs = self.model(input_values)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().cpu().numpy()
                
                # Get label mapping
                id2label = self.model.config.id2label
                
                # We need to map which probability is AI and which is human.
                # Assuming labels contain "ai" or "generated" vs "human" or "real"
                ai_prob = 0.0
                human_prob = 0.0
                
                for idx, prob in enumerate(probs):
                    label = id2label[idx].lower()
                    if "ai" in label or "generated" in label or "fake" in label:
                        ai_prob += float(prob)
                    elif "human" in label or "real" in label or "composed" in label:
                        human_prob += float(prob)
                        
                ai_probs.append(ai_prob)
                human_probs.append(human_prob)

        # Aggregate predictions (average probabilities across all windows)
        avg_ai_prob = np.mean(ai_probs)
        avg_human_prob = np.mean(human_probs)
        
        # Normalize just in case
        total_prob = avg_ai_prob + avg_human_prob
        if total_prob > 0:
            avg_ai_prob /= total_prob
            avg_human_prob /= total_prob
        else:
            avg_ai_prob = 0.5
            avg_human_prob = 0.5

        predicted_label = "AI-generated" if avg_ai_prob > avg_human_prob else "Human-composed"

        processing_time = time.time() - start_time
        
        return {
            "predicted_label": predicted_label,
            "ai_probability": float(avg_ai_prob),
            "human_probability": float(avg_human_prob),
            "model_name": self.model_id,
            "audio_duration": audio_duration,
            "processing_time": processing_time,
            "windows_processed": len(chunks)
        }
