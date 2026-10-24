# src/models/cnn_1d.py
import torch
import torch.nn as nn

class EEG1DCNN(nn.Module):
    """
    1D Convolutional Neural Network for Seizure Prediction.
    Expects input shape: (batch_size, n_channels, sequence_length).
    """
    def __init__(self, n_channels: int = 16, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 3
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 4 (Dimensionality reduction via Adaptive Pooling)
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1) # Sigmoid applied during evaluation/loss calculation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass outputting raw logits."""
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(1)