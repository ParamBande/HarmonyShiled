import os
import sys
import numpy as np
import torch
import joblib
import argparse
import time

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.audio.preprocessor import AudioPreprocessor
from src.features.extractor import FeatureExtractor
from src.features.melspectrogram import MelSpectrogramExtractor
from src.models.resnet18_model import AudioResNet18
from src.pretrained import ASTDetector

def run_classical(audio_path, preprocessor, extractor, model):
    try:
        segments, _ = preprocessor.process(audio_path)
        if not segments:
            return None, "No valid audio segments"
        
        # We just use the first 10s segment for inference
        segment = segments[0]
        feats, feat_names = extractor.extract_features(segment, feature_set='B')
        
        # Reshape to 2D array for sklearn
        X = feats.reshape(1, -1)
        
        prob = model.predict_proba(X)[0][1] # Probability of class 1 (AI)
        pred = "AI" if prob > 0.5 else "Human"
        
        return prob, pred
    except Exception as e:
        return None, str(e)

def run_resnet_ensemble(audio_path, preprocessor, extractor, models, device):
    try:
        segments, _ = preprocessor.process(audio_path)
        if not segments:
            return None, {}, "No valid audio segments"
            
        segment = segments[0]
        mel = extractor.extract(segment).unsqueeze(0).to(device) # Shape: (1, 1, 128, time)
        
        probs = {}
        for name, model in models.items():
            with torch.no_grad():
                out = model(mel)
                prob = torch.sigmoid(out).item()
                probs[name] = prob
                
        avg_prob = np.mean(list(probs.values()))
        pred = "AI" if avg_prob > 0.5 else "Human"
        
        return avg_prob, probs, pred
    except Exception as e:
        return None, {}, str(e)

def main():
    parser = argparse.ArgumentParser(description="HarmonyShield Unified Inference Pipeline")
    parser.add_argument("--audio", type=str, required=True, help="Path to the audio file")
    args = parser.parse_args()
    
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found at {args.audio}")
        sys.exit(1)
        
    print(f"Analyzing audio file...")
    print("-" * 50)
    
    # 1. Initialization
    print("Loading preprocessing pipeline...")
    preprocessor = AudioPreprocessor(segment_duration=10.0, target_sr=16000)
    
    print("Loading Classical ML Pipeline (Set B)...")
    classical_extractor = FeatureExtractor(sr=16000)
    classical_model_path = os.path.join(project_root, "data", "pilot_dataset", "best_rf_model.joblib")
    classical_model = joblib.load(classical_model_path)
    
    print("Loading ResNet18 LOGO Ensemble (5 models)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet_extractor = MelSpectrogramExtractor(sr=16000)
    
    resnet_dir = os.path.join(project_root, "experiments", "resnet18_generalization", "checkpoints")
    resnet_checkpoints = [
        "LOGO_audioldm2_model.pt",
        "LOGO_stable_audio_open_model.pt",
        "LOGO_musicldm_model.pt",
        "LOGO_mustango_model.pt",
        "LOGO_MusicGen_medium_model.pt"
    ]
    
    resnet_models = {}
    for ckpt in resnet_checkpoints:
        model = AudioResNet18(pretrained=False).to(device)
        model.load_state_dict(torch.load(os.path.join(resnet_dir, ckpt), map_location=device, weights_only=True))
        model.eval()
        resnet_models[ckpt.replace("_model.pt", "").replace("LOGO_", "")] = model
        
    print("Loading Plan B Pretrained Pipeline (AST)...")
    ast_detector = ASTDetector()
    
    print("-" * 50)
    print("Executing Inference...")
    print("-" * 50)
    
    # 2. Execution
    start_time = time.time()
    
    # Classical
    c_prob, c_pred = run_classical(args.audio, preprocessor, classical_extractor, classical_model)
    
    # ResNet Ensemble
    r_avg_prob, r_probs, r_pred = run_resnet_ensemble(args.audio, preprocessor, resnet_extractor, resnet_models, device)
    
    # Plan B (AST)
    try:
        ast_result = ast_detector.predict(args.audio)
        ast_prob = ast_result['ai_probability']
        ast_pred = "AI" if ast_prob > 0.5 else "Human"
    except Exception as e:
        ast_prob, ast_pred = None, str(e)
        
    end_time = time.time()
    
    # 3. Reporting
    print(f"Classical ML (Set B Random Forest):")
    if c_prob is not None:
        print(f"  Prediction: {c_pred} (AI Probability: {c_prob:.4f})")
    else:
        print(f"  Error: {c_pred}")
        
    print(f"\nResNet18 (5-Fold LOGO Ensemble):")
    if r_avg_prob is not None:
        print(f"  Ensemble Prediction: {r_pred} (Avg AI Probability: {r_avg_prob:.4f})")
        print(f"  Individual Fold Probabilities:")
        for name, prob in r_probs.items():
            print(f"    - Held-out {name}: {prob:.4f}")
    else:
        print(f"  Error: {r_pred}")
        
    print(f"\nPlan B (Pretrained AST Detector):")
    if ast_prob is not None:
        print(f"  Prediction: {ast_pred} (AI Probability: {ast_prob:.4f})")
    else:
        print(f"  Error: {ast_pred}")
        
    print("-" * 50)
    
    # 4. Consensus Logic
    valid_probs = [p for p in [c_prob, r_avg_prob, ast_prob] if p is not None]
    if len(valid_probs) == 3:
        preds = [c_pred, r_pred, ast_pred]
        
        # Calculate variance
        variance = np.var(valid_probs)
        mean_prob = np.mean(valid_probs)
        
        num_ai = preds.count("AI")
        num_human = preds.count("Human")
        
        print("Final Analysis:")
        
        if variance > 0.05 or (mean_prob > 0.3 and mean_prob < 0.7):
            print("=> CONCLUSION: INCONCLUSIVE / LOW CONFIDENCE")
            print("   The models fundamentally disagree or are highly uncertain about this audio.")
            print("   Do not trust this classification.")
        else:
            if num_ai == 3:
                print("=> CONCLUSION: STRONG CONSENSUS -> AI-GENERATED")
            elif num_human == 3:
                print("=> CONCLUSION: STRONG CONSENSUS -> HUMAN-GENERATED")
            else:
                print("=> CONCLUSION: MIXED CONSENSUS")
                print(f"   Majority vote: {'AI' if num_ai > num_human else 'Human'} ({max(num_ai, num_human)}/3 models)")
                print("   Note: Agreement does not guarantee correctness, especially for external unknown generators.")
                
    print(f"\nTotal Inference Time: {end_time - start_time:.2f} sec")

if __name__ == "__main__":
    main()
