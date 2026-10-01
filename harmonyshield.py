import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="HarmonyShield Plan A CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Feature extraction command
    extract_parser = subparsers.add_parser("extract", help="Extract features from audio")
    extract_parser.add_argument("--input", type=str, required=True, help="Input audio file or directory")
    extract_parser.add_argument("--output", type=str, required=True, help="Output file/directory for features")
    extract_parser.add_argument("--feature-set", type=str, choices=['A', 'B', 'C', 'D', 'E'], default='E', help="Feature set configuration")
    
    # Training command
    train_parser = subparsers.add_parser("train", help="Train a classical ML model")
    train_parser.add_argument("--features", type=str, required=True, help="Path to extracted features dataset")
    train_parser.add_argument("--model-type", type=str, default="rf", choices=["lr", "svm", "rf", "dt", "knn", "gnb", "gbdt"], help="Type of classical model to train")
    train_parser.add_argument("--output", type=str, required=True, help="Path to save the trained model")
    
    # Evaluation command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    eval_parser.add_argument("--test-data", type=str, required=True, help="Path to test features")
    eval_parser.add_argument("--output-dir", type=str, required=True, help="Directory to save evaluation metrics")
    
    # Prediction command
    predict_parser = subparsers.add_parser("predict", help="Run inference on an audio file")
    predict_parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    predict_parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    predict_parser.add_argument("--feature-set", type=str, choices=['A', 'B', 'C', 'D', 'E'], default='E', help="Feature set configuration")
    
    args = parser.parse_args()
    
    if args.command == "extract":
        print(f"Feature extraction requested for {args.input} using set {args.feature_set}")
        # Not fully implemented until dataset is available
        
    elif args.command == "train":
        print(f"Training requested using {args.model_type} model on {args.features}")
        # Not fully implemented until dataset is available
        
    elif args.command == "evaluate":
        print(f"Evaluation requested for model {args.model} on {args.test_data}")
        # Not fully implemented until dataset is available
        
    elif args.command == "predict":
        print(f"Prediction requested for {args.audio} using model {args.model}")
        if not os.path.exists(args.audio):
            print("Audio file does not exist.")
            sys.exit(1)
        if not os.path.exists(args.model):
            print("Model file does not exist.")
            sys.exit(1)
            
        from src.pipeline.infer import PipelineInference
        try:
            pipeline = PipelineInference(model_path=args.model, feature_set=args.feature_set)
            result = pipeline.predict(args.audio, aggregation_mode='prediction')
            print(f"Prediction: {result['prediction']}")
            if result['probabilities'] is not None:
                print(f"Probabilities: {result['probabilities']}")
        except Exception as e:
            print(f"Error during prediction: {e}")
            
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
