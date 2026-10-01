from abc import ABC, abstractmethod
from typing import Dict, Any

class DetectorInterface(ABC):
    @abstractmethod
    def predict(self, audio_path: str) -> Dict[str, Any]:
        """
        Predict whether an audio file is AI-generated or human-composed.
        
        Returns:
            Dict containing:
                - predicted_label (str)
                - ai_probability (float)
                - human_probability (float)
                - model_name (str)
                - audio_duration (float)
                - processing_time (float)
        """
        pass
