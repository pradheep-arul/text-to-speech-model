import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from components.audio import wav_to_mel
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

def batch_preprocess(file_list, root_dir, cache_dir, device):
    """Process a batch of files"""
    for wav_file in file_list:
        wav_path = os.path.join(root_dir, "wavs", f"{wav_file}.wav")
        cache_path = os.path.join(cache_dir, f"{wav_file}.npy")
        
        if not os.path.exists(cache_path):
            mel = wav_to_mel(wav_path, device=device)
            np.save(cache_path, mel[0])  # Remove batch dimension

def preprocess_dataset(root_dir="data/LJSpeech-1.1", batch_size=32):
    # Create cache directory
    cache_dir = os.path.join(root_dir, "mel_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Read metadata
    metadata = pd.read_csv(os.path.join(root_dir, "metadata.csv"), sep="|", header=None)
    all_files = metadata.iloc[:, 0].tolist()
    
    # Filter out already processed files
    to_process = [f for f in all_files if not os.path.exists(os.path.join(cache_dir, f"{f}.npy"))]
    
    if not to_process:
        print("All files already processed!")
        return
    
    print(f"Pre-computing mel spectrograms for {len(to_process)} files...")
    
    # Process in batches
    for i in tqdm(range(0, len(to_process), batch_size)):
        batch_files = to_process[i:i + batch_size]
        batch_preprocess(batch_files, root_dir, cache_dir, device)
        
        # Clear CUDA cache periodically
        if device.type == "cuda":
            torch.cuda.empty_cache()

if __name__ == "__main__":
    preprocess_dataset()
