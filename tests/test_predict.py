import unittest
import os
import sys
from unittest.mock import patch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pretrained import ASTDetector

class TestHarmonyShieldPlanB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # We don't download/instantiate the actual model in CI tests 
        # to save time, but we can test the structure if we mock the model.
        # But for this Plan B, we do want to verify it loads locally if possible.
        # We will instantiate it here if possible.
        try:
            cls.detector = ASTDetector()
        except Exception as e:
            cls.detector = None
            print(f"Failed to initialize model for tests: {e}")

    @patch('librosa.load')
    def test_predict_output_schema(self, mock_load):
        if self.detector is None:
            self.skipTest("Detector failed to load.")
            
        # Mock librosa load to return 3 seconds of dummy audio at 16000Hz
        sr = 16000
        dummy_audio = np.zeros(sr * 3, dtype=np.float32)
        mock_load.return_value = (dummy_audio, sr)
        
        # Pass a dummy path, librosa.load is mocked anyway
        result = self.detector.predict("dummy_path.wav")
        
        self.assertIn("predicted_label", result)
        self.assertIn("ai_probability", result)
        self.assertIn("human_probability", result)
        self.assertIn("model_name", result)
        self.assertIn("audio_duration", result)
        self.assertIn("processing_time", result)
        
        self.assertIsInstance(result["ai_probability"], float)
        self.assertIsInstance(result["human_probability"], float)
        self.assertAlmostEqual(result["ai_probability"] + result["human_probability"], 1.0, places=4)

    def test_invalid_audio(self):
        if self.detector is None:
            self.skipTest("Detector failed to load.")
            
        with self.assertRaises(Exception):
            self.detector.predict("non_existent_file_that_really_does_not_exist.mp3")

if __name__ == "__main__":
    unittest.main()
