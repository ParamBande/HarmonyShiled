import argparse
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pretrained import ASTDetector, LightweightDetector

def main():
    parser = argparse.ArgumentParser(description="HarmonyShield Plan B Pretrained Inference CLI")
    parser.add_argument("--audio", type=str, required=True, help="Path to the audio file (WAV, MP3, FLAC, M4A)")
    parser.add_argument("--model", type=str, choices=["ast", "lightweight", "all"], default="ast", 
                        help="Model to use: 'ast' (default), 'lightweight', or 'all'")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found at {args.audio}")
        sys.exit(1)
        
    models_to_run = []
    if args.model in ["ast", "all"]:
        print("Initializing AST Detector...")
        models_to_run.append(ASTDetector())
    if args.model in ["lightweight", "all"]:
        print("Initializing Lightweight Detector...")
        models_to_run.append(LightweightDetector())
        
    print("\nHarmonyShield")
    print("-" * 30)
    print(f"Audio: {os.path.basename(args.audio)}")
    
    for detector in models_to_run:
        try:
            result = detector.predict(args.audio)
            
            print(f"Duration: {result['audio_duration']:.2f} sec")
            print(f"\nModel: {result['model_name']}")
            print(f"AI probability: {result['ai_probability']:.4f}")
            print(f"Human probability: {result['human_probability']:.4f}")
            print(f"Prediction: {result['predicted_label']}")
            print(f"\nProcessing time: {result['processing_time']:.2f} sec")
            if 'windows_processed' in result:
                 print(f"Windows processed: {result['windows_processed']}")
            print("-" * 30)
            
        except Exception as e:
            print(f"\nError running model {detector.__class__.__name__}: {e}")
            print("-" * 30)

if __name__ == "__main__":
    main()
