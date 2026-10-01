import librosa
import numpy as np
import warnings

class AudioPreprocessor:
    """
    Handles loading and chunking audio for the classical ML pipeline (Plan A).
    """
    def __init__(self, target_sr=22050, segment_duration=10.0, overlap=0.0):
        self.target_sr = target_sr
        self.segment_duration = segment_duration
        self.overlap = overlap
        
    def process(self, file_path):
        """
        Loads an audio file, converts to mono, resamples, and segments it.
        Shorter audio is padded with zeros.
        Returns metadata and list of audio segments.
        """
        # Load audio safely
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Convert to mono by default, standardize sample rate
            try:
                y, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
            except Exception as e:
                try:
                    import soundfile as sf
                    y, sr = sf.read(file_path)
                    if len(y.shape) > 1:
                        y = y.mean(axis=1)
                    if sr != self.target_sr:
                        y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
                except Exception as e2:
                    raise RuntimeError(f"Could not load audio file {file_path}: {e2}")

        # Normalize audio safely
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
            
        duration = len(y) / self.target_sr
        segment_samples = int(self.segment_duration * self.target_sr)
        
        segments = []
        if len(y) < segment_samples:
            # Pad shorter audio
            y_padded = np.zeros(segment_samples)
            y_padded[:len(y)] = y
            segments.append(y_padded)
        else:
            step = int((self.segment_duration - self.overlap) * self.target_sr)
            if step <= 0: step = segment_samples
            
            for start_idx in range(0, len(y) - segment_samples + 1, step):
                segments.append(y[start_idx:start_idx + segment_samples])
                
        metadata = {
            "duration": duration,
            "sample_rate": self.target_sr,
            "num_segments": len(segments),
            "segment_duration": self.segment_duration
        }
        
        return segments, metadata
