import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from components.audio import wav_to_mel
from components.tokenizer import CharTokenizer


class LJSpeechDataset(Dataset):
    def __init__(self, root_dir="data/LJSpeech-1.1", max_samples=None):
        self.metadata = pd.read_csv(
            os.path.join(root_dir, "metadata.csv"), sep="|", header=None
        )
        self.root_dir = root_dir
        if max_samples:
            self.metadata = self.metadata.iloc[:max_samples]
        self.tokenizer = CharTokenizer()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        wav_file = self.metadata.iloc[idx, 0]
        text = self.metadata.iloc[idx, 1]
        mel_path = os.path.join(self.root_dir, "mel_cache", f"{wav_file}.npy")
        
        # Load pre-computed mel spectrogram
        mel = np.load(mel_path)
        
        # Handle different cached formats
        if mel.ndim == 3:  # Shape: (1, n_mels, time)
            mel = mel.squeeze(0)  # Convert to (n_mels, time)
        elif mel.ndim == 2:  # Shape: (n_mels, time) - already correct
            pass
        else:
            raise ValueError(f"Unexpected mel spectrogram shape: {mel.shape}")
            
        tokens = self.tokenizer.encode(text)
        return torch.tensor(tokens), torch.tensor(mel)
