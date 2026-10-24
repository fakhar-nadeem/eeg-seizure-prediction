# src/dataset.py
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import List, Tuple

class EEGChunkWindowDataset(Dataset):
    """
    PyTorch Dataset for loading pre-chunked EEG data windows from disk.
    Handles memory-efficient loading via `.npz` caching and chunk mapping.
    """
    def __init__(
        self, 
        chunk_dir: str, 
        chunk_index_df: pd.DataFrame, 
        patients: List[str], 
        normalize_per_window: bool = False, 
        cache_chunks_in_memory: bool = True
    ):
        self.chunk_dir = chunk_dir
        self.normalize_per_window = normalize_per_window
        self.cache_chunks_in_memory = cache_chunks_in_memory

        self.records = []
        self.chunk_cache = {}

        selected = chunk_index_df[chunk_index_df["patient_id"].isin(patients)].copy()
        valid_chunk_rows = []

        for _, row in selected.iterrows():
            chunk_fname = row["chunk_file"]
            chunk_path = os.path.join(chunk_dir, chunk_fname)

            if not os.path.exists(chunk_path):
                print(f"Warning: Skipping missing chunk {chunk_fname}")
                continue

            try:
                d = np.load(chunk_path, allow_pickle=True)
                y = d["y"]
                d.close()

                valid_chunk_rows.append(row)
                for i in range(len(y)):
                    self.records.append((chunk_fname, i, int(y[i]), row["patient_id"]))

            except Exception as e:
                print(f"Error: Skipping unreadable chunk {chunk_fname} | {e}")

        self.valid_chunk_index_df = pd.DataFrame(valid_chunk_rows)

    def __len__(self) -> int:
        return len(self.records)

    def _load_chunk(self, chunk_fname: str) -> np.ndarray:
        """Retrieves chunk from memory cache or loads from disk if missing."""
        if self.cache_chunks_in_memory and chunk_fname in self.chunk_cache:
            return self.chunk_cache[chunk_fname]

        chunk_path = os.path.join(self.chunk_dir, chunk_fname)
        d = np.load(chunk_path, allow_pickle=True)
        X = d["X"].astype(np.float32)
        d.close()

        if self.cache_chunks_in_memory:
            self.chunk_cache[chunk_fname] = X

        return X

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        chunk_fname, window_idx, label, patient_id = self.records[idx]
        
        # Load the whole chunk, then extract the specific window
        X_all = self._load_chunk(chunk_fname)
        X = X_all[window_idx]

        # Optional secondary normalization
        if self.normalize_per_window:
            mean = X.mean(axis=1, keepdims=True)
            std = X.std(axis=1, keepdims=True) + 1e-8
            X = (X - mean) / std

        return (
            torch.tensor(X, dtype=torch.float32), 
            torch.tensor(label, dtype=torch.float32), 
            patient_id
        )