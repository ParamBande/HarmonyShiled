import numpy as np
from src.audio.preprocessor import AudioPreprocessor
from src.features.extractor import FeatureExtractor
from src.features.aggregator import FeatureAggregator
from src.models.trainer import ModelTrainer

class PipelineInference:
    """
    Handles end-to-end inference for Plan A models.
    """
    def __init__(self, model_path, feature_set='E', segment_duration=10.0, target_sr=22050):
        self.preprocessor = AudioPreprocessor(target_sr=target_sr, segment_duration=segment_duration)
        self.extractor = FeatureExtractor(sr=target_sr)
        self.model = ModelTrainer.load(model_path)
        self.feature_set = feature_set
        
    def predict(self, audio_path, aggregation_mode='prediction'):
        """
        Runs the full inference pipeline.
        aggregation_mode: 
            'prediction' - predicts on each segment and aggregates the probabilities/labels.
            'feature' - aggregates segment features into track features and predicts once.
        """
        segments, metadata = self.preprocessor.process(audio_path)
        
        segment_features = []
        for seg in segments:
            feats, _ = self.extractor.extract_features(seg, feature_set=self.feature_set)
            segment_features.append(feats)
            
        if aggregation_mode == 'feature':
            # Aggregate features then predict
            track_features = FeatureAggregator.aggregate(segment_features)
            # Reshape for sklearn
            track_features = track_features.reshape(1, -1)
            
            pred = self.model.predict(track_features)[0]
            try:
                prob = self.model.predict_proba(track_features)[0]
            except NotImplementedError:
                prob = None
                
            return {
                "prediction": pred,
                "probabilities": prob,
                "metadata": metadata
            }
            
        elif aggregation_mode == 'prediction':
            # Predict per segment then aggregate
            seg_feats_array = np.array(segment_features)
            preds = self.model.predict(seg_feats_array)
            
            try:
                probs = self.model.predict_proba(seg_feats_array)
                mean_prob = np.mean(probs, axis=0)
                final_pred = np.argmax(mean_prob)
            except NotImplementedError:
                mean_prob = None
                # Majority voting
                counts = np.bincount(preds)
                final_pred = np.argmax(counts)
                
            return {
                "prediction": final_pred,
                "probabilities": mean_prob,
                "metadata": metadata,
                "segment_predictions": preds.tolist()
            }
            
        else:
            raise ValueError(f"Unknown aggregation mode: {aggregation_mode}")
