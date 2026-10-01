import numpy as np
import librosa

class FeatureExtractor:
    """
    Extracts classical ML features from audio segments.
    """
    def __init__(self, sr=22050, n_mfcc=20):
        self.sr = sr
        self.n_mfcc = n_mfcc

    def extract_features(self, y, feature_set='E'):
        """
        Extracts features based on the requested feature set.
        A: MFCC only
        B: Spectral only
        C: MFCC + Spectral
        D: MFCC + Spectral + Harmonic
        E: MFCC + Spectral + Harmonic + Temporal
        """
        features = {}
        
        # Clean audio array before passing to librosa
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        if feature_set in ['A', 'C', 'D', 'E']:
            features.update(self._extract_mfcc(y))
            
        if feature_set in ['B', 'C', 'D', 'E']:
            features.update(self._extract_spectral(y))
            
        if feature_set in ['D', 'E']:
            features.update(self._extract_harmonic(y))
            
        if feature_set in ['E']:
            features.update(self._extract_temporal(y))
            features.update(self._extract_energy(y)) # Treat energy as part of the full set or temporal
            
        # Convert dictionary to fixed-length numerical vector (and keep names)
        feature_names = list(features.keys())
        feature_values = np.array(list(features.values()))
        
        # Handle NaN/Inf safely
        feature_values = np.nan_to_num(feature_values, nan=0.0, posinf=0.0, neginf=0.0)
        
        return feature_values, feature_names
        
    def _extract_mfcc(self, y):
        mfccs = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        feats = {}
        for i in range(self.n_mfcc):
            feats[f'mfcc_{i+1}_mean'] = mfcc_mean[i]
            feats[f'mfcc_{i+1}_std'] = mfcc_std[i]
        return feats

    def _extract_spectral(self, y):
        feats = {}
        # Spectral Centroid
        cent = librosa.feature.spectral_centroid(y=y, sr=self.sr)
        feats['spectral_centroid_mean'] = np.mean(cent)
        feats['spectral_centroid_std'] = np.std(cent)
        
        # Spectral Bandwidth
        bw = librosa.feature.spectral_bandwidth(y=y, sr=self.sr)
        feats['spectral_bandwidth_mean'] = np.mean(bw)
        feats['spectral_bandwidth_std'] = np.std(bw)
        
        # Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr)
        feats['spectral_rolloff_mean'] = np.mean(rolloff)
        feats['spectral_rolloff_std'] = np.std(rolloff)
        
        # Spectral Flatness
        flatness = librosa.feature.spectral_flatness(y=y)
        feats['spectral_flatness_mean'] = np.mean(flatness)
        feats['spectral_flatness_std'] = np.std(flatness)
        
        # Spectral Contrast
        S = np.abs(librosa.stft(y))
        contrast = librosa.feature.spectral_contrast(S=S, sr=self.sr)
        for i in range(contrast.shape[0]):
            feats[f'spectral_contrast_{i+1}_mean'] = np.mean(contrast[i])
            feats[f'spectral_contrast_{i+1}_std'] = np.std(contrast[i])
            
        return feats
        
    def _extract_energy(self, y):
        feats = {}
        rms = librosa.feature.rms(y=y)
        feats['rms_energy_mean'] = np.mean(rms)
        feats['rms_energy_std'] = np.std(rms)
        
        zcr = librosa.feature.zero_crossing_rate(y)
        feats['zcr_mean'] = np.mean(zcr)
        feats['zcr_std'] = np.std(zcr)
        
        # Silence/low-energy ratio
        low_energy_frame_ratio = np.sum(rms < (np.mean(rms) * 0.5)) / float(rms.shape[1])
        feats['low_energy_ratio'] = low_energy_frame_ratio
        
        return feats

    def _extract_harmonic(self, y):
        feats = {}
        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr)
        for i in range(chroma.shape[0]):
            feats[f'chroma_{i+1}_mean'] = np.mean(chroma[i])
            feats[f'chroma_{i+1}_std'] = np.std(chroma[i])
            
        # Harmonic/Percussive
        y_h, y_p = librosa.effects.hpss(y)
        feats['harmonic_energy'] = np.sum(y_h**2)
        feats['percussive_energy'] = np.sum(y_p**2)
        
        return feats

    def _extract_temporal(self, y):
        feats = {}
        # Onset envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr)
        feats['onset_mean'] = np.mean(onset_env)
        feats['onset_std'] = np.std(onset_env)
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sr)
        # Handle tempo shape difference between versions
        if isinstance(tempo, np.ndarray):
            tempo = tempo[0]
        feats['tempo'] = float(tempo)
        
        return feats
