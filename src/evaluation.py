# src/evaluation.py
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from typing import Dict, Tuple

def evaluate_model(
    model: torch.nn.Module, 
    loader: torch.utils.data.DataLoader, 
    device: torch.device, 
    threshold: float = 0.5
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Evaluates a PyTorch model and computes classification metrics.
    Handles dynamic batching and extracts patient-level metadata.
    """
    model.eval()

    all_probs, all_preds, all_labels, all_patients = [], [], [], []

    with torch.no_grad():
        for X, y, patients in loader:
            X = X.to(device)
            logits = model(X)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(int)

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(y.numpy().tolist())
            all_patients.extend(list(patients))

    y_true = np.array(all_labels).astype(int)
    y_prob = np.array(all_probs)
    y_pred = np.array(all_preds)

    # AUC requires at least one instance of both classes
    auc_val = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "auc": auc_val
    }

    return metrics, y_true, y_prob, y_pred, all_patients