import numpy as np
import torch
import torchaudio
from torchaudio.transforms import GriffinLim, InverseMelScale, MelSpectrogram


def wav_to_mel(
    wav_path, sr=22050, n_mels=80, hop_length=256, win_length=1024, device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load audio using torchaudio (faster than librosa)
    waveform, sample_rate = torchaudio.load(wav_path)

    # Resample if needed
    if sample_rate != sr:
        resampler = torchaudio.transforms.Resample(sample_rate, sr).to(device)
        waveform = resampler(waveform.to(device))
    else:
        waveform = waveform.to(device)

    # Create mel spectrogram transform on GPU
    mel_transform = MelSpectrogram(
        sample_rate=sr,
        n_fft=win_length,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    ).to(device)

    # Compute mel spectrogram
    mel_spec = mel_transform(waveform)

    # Convert to dB scale to match librosa.power_to_db(ref=np.max)
    # This matches the working CPU version more closely
    mel_spec = 10.0 * torch.log10(torch.clamp(mel_spec, min=1e-10))
    # Normalize by max like librosa does with ref=np.max
    mel_spec = mel_spec - torch.max(mel_spec)

    return (
        mel_spec.squeeze(0).cpu().numpy()
    )  # shape: (n_mels, time) - match librosa output


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
