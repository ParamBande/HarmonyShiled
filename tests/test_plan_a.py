import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import numpy as np
from src.audio.preprocessor import AudioPreprocessor
from src.features.extractor import FeatureExtractor
from src.utils.splitter import DataSplitter
from src.models.trainer import ModelTrainer

@pytest.fixture
def dummy_audio():
    # 3 seconds of dummy audio at 22050 Hz
    return np.random.randn(22050 * 3)

def test_audio_segmentation(dummy_audio, tmp_path):
    import soundfile as sf
    test_file = tmp_path / "test.wav"
    sf.write(test_file, dummy_audio, 22050)
    
    # 1. Test short audio padding
    preprocessor = AudioPreprocessor(target_sr=22050, segment_duration=10.0)
    segments, metadata = preprocessor.process(str(test_file))
    assert metadata['num_segments'] == 1
    assert len(segments[0]) == 10 * 22050
    
    # 2. Test chunking
    long_audio = np.random.randn(22050 * 25)
    sf.write(test_file, long_audio, 22050)
    segments, metadata = preprocessor.process(str(test_file))
    assert metadata['num_segments'] == 2 # 0-10, 10-20 (remainder ignored)
    assert metadata['num_segments'] == 2
    assert len(segments) == 2
    assert len(segments[0]) == 10 * 22050

def test_feature_extraction(dummy_audio):
    extractor = FeatureExtractor(sr=22050)
    
    # Test fixed-length vectors and naming
    feats, names = extractor.extract_features(dummy_audio, feature_set='A')
    assert len(feats) == len(names)
    assert len(feats) == 40 # 20 mfccs * 2 (mean, std)
    
    feats_E, names_E = extractor.extract_features(dummy_audio, feature_set='E')
    assert len(feats_E) == len(names_E)
    assert len(feats_E) > len(feats)
    
    # Test NaN/inf handling
    nan_audio = dummy_audio.copy()
    nan_audio[0] = np.nan
    nan_audio[1] = np.inf
    feats_nan, names_nan = extractor.extract_features(nan_audio, feature_set='E')
    assert not np.any(np.isnan(feats_nan))
    assert not np.any(np.isinf(feats_nan))

def test_group_split():
    splitter = DataSplitter(random_state=42)
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    # 10 groups, each has 10 samples
    groups = np.repeat(np.arange(10), 10)
    
    X_train, X_val, X_test, y_train, y_val, y_test, g_train, g_val, g_test = splitter.split_data(
        X, y, groups, test_size=0.2, val_size=0.2
    )
    
    # Check that groups don't overlap
    assert len(set(g_train).intersection(set(g_val))) == 0
    assert len(set(g_train).intersection(set(g_test))) == 0
    assert len(set(g_val).intersection(set(g_test))) == 0

def test_model_persistence(tmp_path):
    X = np.random.randn(50, 10)
    y = np.random.randint(0, 2, 50)
    
    trainer = ModelTrainer(model_type='rf', random_state=42)
    trainer.train(X, y)
    
    model_path = tmp_path / "model.pkl"
    trainer.save(str(model_path))
    
    assert os.path.exists(model_path)
    
    loaded_trainer = ModelTrainer.load(str(model_path))
    preds = loaded_trainer.predict(X)
    
    assert len(preds) == 50
