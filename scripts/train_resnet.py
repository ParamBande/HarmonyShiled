import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df[idx]
        path = row['path']
        label = 1.0 if row['label'] == 'ai' else 0.0
        
        segments, _ = self.preprocessor.process(path)
        if not segments:
            # Fallback for empty
            segment = np.zeros(int(10.0 * 16000))
        else:
            segment = segments[0]
            
        mel = self.extractor.extract(segment)
        return mel, torch.tensor([label], dtype=torch.float32)

def sanity_check():
    dataset_dir = r"D:\PROJECTS\harmonyshield\harmonyshield\data\pilot_dataset"
    csv_path = os.path.join(dataset_dir, "dataset.csv")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
        
    # Just take 4 samples
    mini_df = rows[:4]
    
    preprocessor = AudioPreprocessor(segment_duration=10.0, target_sr=16000)
    extractor = MelSpectrogramExtractor(sr=16000)
    
    ds = AudioDataset(mini_df, extractor, preprocessor)
    dl = DataLoader(ds, batch_size=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sanity check on {device}")
    
    model = AudioResNet18(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(2):
        for mels, labels in dl:
            mels, labels = mels.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(mels)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
    print("Sanity check passed!")
    
if __name__ == "__main__":
    sanity_check()
