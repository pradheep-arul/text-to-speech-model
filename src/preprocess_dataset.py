import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from components.audio import wav_to_mel

def preprocess_dataset(root_dir="data/LJSpeech-1.1"):
    # Create cache directory
    cache_dir = os.path.join(root_dir, "mel_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Read metadata
    metadata = pd.read_csv(os.path.join(root_dir, "metadata.csv"), sep="|", header=None)
    
    print(f"Pre-computing mel spectrograms for {len(metadata)} files...")
    for idx in tqdm(range(len(metadata))):
        wav_file = metadata.iloc[idx, 0]
        wav_path = os.path.join(root_dir, "wavs", f"{wav_file}.wav")
        cache_path = os.path.join(cache_dir, f"{wav_file}.npy")
        
        if not os.path.exists(cache_path):
            mel = wav_to_mel(wav_path)
            np.save(cache_path, mel)

if __name__ == "__main__":
    preprocess_dataset()
