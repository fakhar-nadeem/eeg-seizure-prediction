# src/config.py
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# Base Project Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

@dataclass
class PreprocessingConfig:
    """EEG Signal Preprocessing & Windowing Parameters"""
    patients_to_use: List[str] = field(default_factory=lambda: [
        "chb01", "chb02", "chb03", "chb04", "chb05",
        "chb11", "chb12", "chb13", "chb14", "chb15"
    ])
    target_sfreq: int = 256
    bandpass_low: float = 0.5
    bandpass_high: float = 40.0
    window_sec: int = 30
    overlap: float = 0.5
    
    # Seizure prediction horizon settings
    preictal_start_min: int = 30
    preictal_end_min: int = 5
    interictal_min_distance_hr: float = 2.0
    max_interictal_windows_per_file: int = 80
    
    # Channel harmonization constraints
    min_common_channels: int = 10
    expected_n_channels: int = 16

@dataclass
class ModelConfig:
    """PyTorch 1D-CNN Training & Validation Parameters"""
    valid_patients: List[str] = field(default_factory=lambda: [
        "chb01", "chb02", "chb03", "chb04", "chb05"
    ])
    expected_channels: int = 16
    window_sec: int = 30
    sfreq: int = 256
    batch_size: int = 32
    epochs: int = 15
    lr: float = 0.001
    weight_decay: float = 0.0001
    dropout: float = 0.3
    early_stopping_patience: int = 4
    num_workers: int = 0
    rf_threshold: float = 0.1

    @property
    def sequence_length(self) -> int:
        """Calculates input time steps per window"""
        return self.window_sec * self.sfreq

PREPROCESSING_CONFIG = PreprocessingConfig()
MODEL_CONFIG = ModelConfig()