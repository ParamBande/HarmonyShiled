import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import sys

sys.path.insert(0, r"D:\PROJECTS\harmonyshield\harmonyshield")
from src.features.melspectrogram import MelSpectrogramExtractor
from src.models.resnet18_model import AudioResNet18
from src.audio.preprocessor import AudioPreprocessor

class AudioDataset(Dataset):
    def __init__(self, df, extractor, preprocessor):
        self.df = df
        self.extractor = extractor
        self.preprocessor = preprocessor
        
        # Cache to speed up multiple folds
        self.cache = {}
        print(f"Initializing dataset with {len(df)} samples...")
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df[idx]
        path = row['path']
        label = 1.0 if row['label'] == 'ai' else 0.0
        
        if path in self.cache:
            mel = self.cache[path]
        else:
            segments, _ = self.preprocessor.process(path)
            if not segments:
                segment = np.zeros(int(10.0 * 16000))
            else:
                segment = segments[0]
            mel = self.extractor.extract(segment)
            self.cache[path] = mel
            
        return mel, torch.tensor([label], dtype=torch.float32)

def train_and_eval(train_df, test_df, extractor, preprocessor, device, exp_name, checkpoint_dir):
    train_ds = AudioDataset(train_df, extractor, preprocessor)
    test_ds = AudioDataset(test_df, extractor, preprocessor)
    
    # Pre-populate cache so we don't duplicate work in loops if possible
    
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    model = AudioResNet18(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    epochs = 3 # Lightweight training
    for epoch in range(epochs):
        model.train()
        for mels, labels in train_dl:
            mels, labels = mels.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(mels)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            
    # Eval
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for mels, labels in test_dl:
            mels = mels.to(device)
            out = model(mels)
            probs = torch.sigmoid(out).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.numpy().flatten())
            
    y_test = np.array(all_labels)
    preds = np.array(all_preds)
    probs = np.array(all_probs)
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_test, probs)
    except:
        auc = 0.0
    cm = confusion_matrix(y_test, preds).tolist()
    
    # Save checkpoint
    ckpt_path = os.path.join(checkpoint_dir, f"{exp_name}_model.pt")
    torch.save(model.state_dict(), ckpt_path)
    
    return {
        'Experiment': exp_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'ROC-AUC': auc,
        'CM': cm,
        'Test_Human': int(np.sum(y_test == 0)),
        'Test_AI': int(np.sum(y_test == 1)),
        'Train_Size': len(train_df),
        'Checkpoint': ckpt_path
    }

def main():
    dataset_dir = r"D:\PROJECTS\harmonyshield\harmonyshield\data\pilot_dataset"
    csv_path = os.path.join(dataset_dir, "dataset.csv") # The master CSV containing 600 rows
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
        
    # We must use exactly the same ytid splits!
    # Let's extract unique ytids and use GroupShuffleSplit with random_state=42 
    # to perfectly match the classical pipeline!
    ytids_all = np.array([r['ytid'] for r in rows])
    unique_ytids = np.unique(ytids_all)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_ytid_idx, test_ytid_idx = next(gss.split(unique_ytids, groups=unique_ytids))
    train_ytids = set(unique_ytids[train_ytid_idx])
    test_ytids = set(unique_ytids[test_ytid_idx])
    
    generators = ['audioldm2', 'stable_audio_open', 'musicldm', 'mustango', 'MusicGen_medium']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running LOGO on {device}")
    
    preprocessor = AudioPreprocessor(segment_duration=10.0, target_sr=16000)
    extractor = MelSpectrogramExtractor(sr=16000)
    
    out_dir = r"D:\PROJECTS\harmonyshield\harmonyshield\experiments\resnet18_generalization"
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    results = []
    
    for g in generators:
        print(f"\n--- LOGO: Held-out {g} ---")
        
        train_df = [
            r for r in rows 
            if r['ytid'] in train_ytids and (r['label'] == 'human' or (r['label'] == 'ai' and r.get('generator', '') != g))
        ]
        
        test_df = [
            r for r in rows
            if r['ytid'] in test_ytids and (r['label'] == 'human' or (r['label'] == 'ai' and r.get('generator', '') == g))
        ]
        
        res = train_and_eval(train_df, test_df, extractor, preprocessor, device, f"LOGO_{g}", ckpt_dir)
        results.append(res)
        
    # Save CSV
    csv_out = os.path.join(out_dir, "resnet18_results.csv")
    with open(csv_out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print("\n--- RESULTS ---")
    for r in results:
        print(f"{r['Experiment']:<25} | Acc: {r['Accuracy']:.4f} | F1: {r['F1']:.4f} | ROC-AUC: {r['ROC-AUC']:.4f}")
        
    # Write analysis
    f1s = [r['F1'] for r in results]
    rocs = [r['ROC-AUC'] for r in results]
    mean_f1 = np.mean(f1s)
    std_f1 = np.std(f1s)
    mean_roc = np.mean(rocs)
    
    print("\n--- COMPARISON ---")
    print(f"ResNet18 Mean LOGO F1: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Classical Mean LOGO F1: 0.7092 ± 0.0367")
    print(f"ResNet18 Mean LOGO ROC-AUC: {mean_roc:.4f}")
    print(f"Classical Mean LOGO ROC-AUC: 0.7152")
    
if __name__ == "__main__":
    main()
