import librosa
import numpy as np
import torch
from torchaudio.transforms import GriffinLim, InverseMelScale


def wav_to_mel(wav_path, sr=22050, n_mels=80, hop_length=256, win_length=1024):
    y, _ = librosa.load(wav_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, win_length=win_length
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_db  # shape: (n_mels, time)


def mel_to_audio(mel_spec, sr=22050, n_fft=1024, hop_length=256, win_length=1024):
    # -------- Griffin-Lim Reconstruction -------- #
    mel_tensor = torch.tensor(mel_spec)

    # Convert from dB scale (training format) to linear power scale (Griffin-Lim format)
    mel_tensor = torch.pow(10.0, mel_tensor / 10.0)  # dB to power scale
    mel_tensor = torch.clamp(mel_tensor, min=1e-8)  # Avoid zeros

    print("Converting mel spectrogram to audio...")
    inverse_mel = InverseMelScale(n_stft=512 + 1, n_mels=80, sample_rate=sr)(mel_tensor)
    print("Applying Griffin-Lim...")
    waveform = GriffinLim(n_fft=n_fft, hop_length=hop_length, win_length=win_length)(
        inverse_mel
    )

    print(
        f"Final audio - min: {waveform.min():.6f}, max: {waveform.max():.6f}, RMS: {torch.sqrt(torch.mean(waveform**2)):.6f}"
    )

    print("Griffin-Lim completed.")
    return waveform
