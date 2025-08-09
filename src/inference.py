import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

from components.audio import mel_to_audio
from components.tokenizer import CharTokenizer
from nn_models.transformer_tts import TransformerTTS

# -------- Setup -------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"Using GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
else:
    torch.set_num_threads(14)  # Use all CPU cores
    print("Using CPU threads:", torch.get_num_threads())
print("Using device:", device)

# -------- Load Tokenizer and Model -------- #
tokenizer = CharTokenizer()
vocab_size = len(tokenizer.vocab)

model = TransformerTTS(vocab_size=vocab_size).to(device)
model.load_state_dict(
    torch.load("model/tts_transformer_latest.pth", map_location=device)
)
model.eval()

# -------- Prepare Input -------- #
text = "Speech synthesis is amazing."
tokens = tokenizer.encode(text)
tokens = torch.tensor(tokens).unsqueeze(0).to(device)  # [1, T_text]

# -------- Inference Parameters -------- #
max_len = 300  # Reduced for faster inference
mel_dim = 80

# -------- Auto-regressive Inference -------- #
print(f"Starting inference for {max_len} frames...")
start_time = time.time()

with torch.no_grad():
    generated_mels = []

    # Start with training mean initialization (from debug analysis: mean = -57.26)
    training_mean = -57.26  # Average value from training mel spectrograms
    decoder_input = torch.full((1, 1, mel_dim), training_mean).to(device)
    print(f"Starting with training mean initialization: {training_mean}")

    for t in range(max_len):
        if t % 20 == 0 or t == max_len - 1:
            print(f"Progress: {t+1}/{max_len} frames ({(t+1)/max_len*100:.1f}%)")

        output = model(tokens, decoder_input)  # [1, t+1, mel_dim]
        next_frame = output[:, -1:, :]  # [1, 1, mel_dim]
        generated_mels.append(next_frame.cpu())

        decoder_input = torch.cat(
            [decoder_input, next_frame], dim=1
        )  # Append for next step

    mel = torch.cat(generated_mels, dim=1).squeeze(0).cpu().numpy()  # [mel_dim, T]
    mel = mel.T  # [80, T] for Griffin-Lim

inference_time = time.time() - start_time
print(
    f"✅ Inference completed in {inference_time:.2f}s ({inference_time/max_len*1000:.1f}ms per frame)"
)

# Debug mel spectrogram
print(f"Mel shape: {mel.shape}")
print(f"Mel min: {mel.min():.3f}, max: {mel.max():.3f}, mean: {mel.mean():.3f}")
print(f"Mel std: {mel.std():.3f}")
print(
    f"Non-zero values: {np.count_nonzero(mel)}/{mel.size} ({np.count_nonzero(mel)/mel.size*100:.1f}%)"
)


# -------- Plot Mel -------- #
plt.figure(figsize=(10, 4))
plt.imshow(mel, aspect="auto", origin="lower", interpolation="none")
plt.title("Generated Mel Spectrogram")
plt.colorbar()
plt.tight_layout()
plt.savefig("output/mel_spectrogram.png", dpi=150, bbox_inches="tight")
plt.close()  # Close the figure to free memory
print("📊 Mel spectrogram saved as output/mel_spectrogram.png")


# Save waveform
audio = mel_to_audio(mel)
torchaudio.save("output/audio.wav", audio.unsqueeze(0), 22050)
amplified = audio * 100  # Amplify significantly
torchaudio.save("output/audio_amplified.wav", amplified.unsqueeze(0), 22050)

print("✅ Audio saved as output/audio.wav")
print("✅ Amplified audio saved as output/audio_amplified.wav")
