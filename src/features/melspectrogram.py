import torch
import torchaudio
import numpy as np

class MelSpectrogramExtractor:
    """
    Extracts Mel-spectrograms from audio segments for the ResNet pipeline.
    """
    def __init__(self, sr=16000, n_mels=128, n_fft=1024, hop_length=512):
        self.sr = sr
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)

    def extract(self, y):
        """
        y: numpy array of shape (samples,)
        Returns: torch.Tensor of shape (1, n_mels, time)
        """
        if isinstance(y, np.ndarray):
            y_tensor = torch.tensor(y, dtype=torch.float32)
        else:
            y_tensor = y.float()
            
        # Ensure 1D
        if y_tensor.ndim > 1:
            y_tensor = y_tensor.mean(dim=0)
            
        mel = self.transform(y_tensor)
        mel_db = self.amplitude_to_db(mel)
        
        # Add channel dimension (1, H, W)
        mel_db = mel_db.unsqueeze(0)
        
        # Normalize to [0, 1] approximately for CNN
        # Typically mel_db is in range [-80, 0] if top_db=80 and max is 0
        mel_db = (mel_db + 80.0) / 80.0
        mel_db = torch.clamp(mel_db, 0.0, 1.0)
        
        return mel_db
