# src/data_utils.py
import numpy as np
import mne
from typing import List, Tuple, Optional
from src.config import PREPROCESSING_CONFIG

def standardize_ch_name(name: str) -> str:
    """Helper function to standardize channel strings."""
    return name.upper()

def butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 4) -> np.ndarray:
    """Applies a Butterworth bandpass filter."""
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

def load_and_preprocess_edf(
    edf_path: str, 
    common_channels: List[str], 
    target_sfreq: int = 256, 
    l_freq: float = 0.5, 
    h_freq: float = 40.0
) -> Tuple[Optional[np.ndarray], Optional[int], Optional[List[str]]]:
    """
    Loads an EDF file, filters for common channels, resamples, applies bandpass, 
    and z-score standardizes the EEG signals.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.pick_types(eeg=True)
    raw.rename_channels(lambda x: standardize_ch_name(x))

    available = set(raw.ch_names)
    keep = [ch for ch in common_channels if ch in available]

    if len(keep) < PREPROCESSING_CONFIG.min_common_channels:
        return None, None, None

    raw.pick(keep)

    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq)

    data = raw.get_data()
    data = butter_bandpass_filter(data, l_freq, h_freq, raw.info["sfreq"])

    # Z-score normalization per channel
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True) + 1e-8
    data = (data - mean) / std

    return data.astype(np.float32), int(raw.info["sfreq"]), keep

def generate_windows(
    data: np.ndarray, 
    sfreq: int, 
    window_sec: int = 30, 
    overlap: float = 0.5
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Slices continuous EEG data into overlapping windows.
    Returns tensor of shape (num_windows, channels, time_steps).
    """
    window_size = int(window_sec * sfreq)
    stride = int(window_size * (1 - overlap))
    windows = []
    starts_sec = []

    for start in range(0, data.shape[1] - window_size + 1, stride):
        end = start + window_size
        windows.append(data[:, start:end])
        starts_sec.append(start / sfreq)

    if len(windows) == 0:
        return None, None

    return np.stack(windows).astype(np.float32), np.array(starts_sec)