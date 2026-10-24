# scripts/02_train_cnn.py
import sys
from pathlib import Path

# Automatically add the root project directory to Python's path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.config import PREPROCESSING_CONFIG, MODEL_CONFIG, PROCESSED_DATA_DIR
from src.dataset import EEGChunkWindowDataset
from src.models.cnn_1d import EEG1DCNN
from src.evaluation import evaluate_model
import pandas as pd

def train_lopo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    metadata_path = PROCESSED_DATA_DIR / "all_metadata.csv"
    try:
        chunk_index_df = pd.read_csv(metadata_path)
    except FileNotFoundError:
        print(f"Error: Could not find {metadata_path}. Please download it from Google Drive.")
        return

    patients = MODEL_CONFIG.valid_patients

    for test_patient in patients:
        print(f"\n{'='*40}\nStarting LOPO fold: Test Patient {test_patient}\n{'='*40}")
        
        train_patients = [p for p in patients if p != test_patient]
        
        train_dataset = EEGChunkWindowDataset(
            chunk_dir=PROCESSED_DATA_DIR / "chunks",
            chunk_index_df=chunk_index_df,
            patients=train_patients
        )
        test_dataset = EEGChunkWindowDataset(
            chunk_dir=PROCESSED_DATA_DIR / "chunks",
            chunk_index_df=chunk_index_df,
            patients=[test_patient]
        )

        train_loader = DataLoader(train_dataset, batch_size=MODEL_CONFIG.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=MODEL_CONFIG.batch_size, shuffle=False)

        model = EEG1DCNN(n_channels=MODEL_CONFIG.expected_channels, dropout=MODEL_CONFIG.dropout).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=MODEL_CONFIG.lr, weight_decay=MODEL_CONFIG.weight_decay)

        for epoch in range(MODEL_CONFIG.epochs):
            model.train()
            running_loss = 0.0
            
            for X_batch, y_batch, _ in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
            if len(train_loader) > 0:
                print(f"Epoch {epoch+1}/1 - Loss: {running_loss/len(train_loader):.4f}")
            else:
                print(f"Epoch {epoch+1}/1 - No training data found for {train_patients}")

        if len(test_loader) > 0:
            metrics, _, _, _, _ = evaluate_model(model, test_loader, device, threshold=MODEL_CONFIG.rf_threshold)
            print(f"Results for {test_patient}: AUC = {metrics['auc']:.4f}, Recall = {metrics['recall']:.4f}")
        else:
            print(f"No test data found for {test_patient} in this dry run.")

if __name__ == "__main__":
    train_lopo()