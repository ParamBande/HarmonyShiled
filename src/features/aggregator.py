import numpy as np

class FeatureAggregator:
    """
    Aggregates segment-level features into track-level features.
    """
    @staticmethod
    def aggregate(segment_features):
        """
        segment_features: list of 1D numpy arrays (segments x features) or a 2D array.
        Returns track-level features (mean, std, median) concatenated.
        """
        if len(segment_features) == 0:
            return np.array([])
            
        seg_array = np.array(segment_features)
        
        # Calculate statistics across the segment axis (axis=0)
        mean_feats = np.mean(seg_array, axis=0)
        std_feats = np.std(seg_array, axis=0)
        median_feats = np.median(seg_array, axis=0)
        
        # Concatenate them for the final track-level representation
        track_features = np.concatenate([mean_feats, std_feats, median_feats])
        return track_features

    @staticmethod
    def get_aggregated_feature_names(base_names):
        """
        Returns the names of the aggregated features.
        """
        names = []
        names.extend([f"{name}_agg_mean" for name in base_names])
        names.extend([f"{name}_agg_std" for name in base_names])
        names.extend([f"{name}_agg_median" for name in base_names])
        return names
