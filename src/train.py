import glob
import os
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from components.dataset import LJSpeechDataset
from components.tokenizer import CharTokenizer
from nn_models.transformer_tts import TransformerTTS
from utils.collate import collate_fn



# -------- Setup -------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"Using GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
else:
    torch.set_num_threads(14)  # Use all CPU cores
    print("Using CPU threads:", torch.get_num_threads())
print("Using device:", device)

# -------- Hyperparameters -------- #
vocab = CharTokenizer()
vocab_size = len(vocab.vocab)
batch_size = 32  # Match the working CPU version exactly
num_epochs = 50
lr = 1e-4  # Match the working CPU version

# Enable CUDA optimizations
torch.backends.cudnn.benchmark = True  # Enable auto-tuner
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere
torch.backends.cudnn.allow_tf32 = True

# Memory optimization
if device.type == "cuda":
    torch.cuda.empty_cache()  # Clear any residual memory
    # Note: Modern PyTorch manages memory automatically
    torch.cuda.set_device(0)  # Use first GPU

# -------- Checkpoint Configuration -------- #
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
save_every = 1  # Save checkpoint every N epochs

# -------- Dataset & Loader -------- #
dataset = LJSpeechDataset()

loader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    collate_fn=collate_fn, 
    num_workers=0  # Match working CPU version - simple loading
)

# -------- Model, Loss, Optimizer -------- #
model = TransformerTTS(vocab_size=vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Match working CPU version
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)
loss_fn = torch.nn.L1Loss()


# -------- Checkpoint Loading -------- #
def load_checkpoint():
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
    if not checkpoint_files:
        return 0, []

    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    print(f"Loading checkpoint: {latest_checkpoint}")

    checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=True)
    
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    start_epoch = checkpoint["epoch"] + 1
    loss_history = checkpoint.get("loss_history", [])

    print(f"Resumed from epoch {checkpoint['epoch']} (starting epoch {start_epoch})")
    return start_epoch, loss_history


def save_checkpoint(epoch, loss_history):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    temp_path = checkpoint_path + '.tmp'
    
    try:
        # Save to temporary file first
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss_history": loss_history,
            "vocab_size": vocab_size,
            "batch_size": batch_size,
            "lr": lr,
        }
        torch.save(checkpoint, temp_path)
        
        # If save was successful, rename to final path
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        os.rename(temp_path, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    except Exception as e:
        print(f"Warning: Failed to save checkpoint: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
    print(f"Checkpoint saved: {checkpoint_path}")

    old_checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
    old_checkpoints = [f for f in old_checkpoints if f != checkpoint_path]
    if len(old_checkpoints) > 2:
        oldest = min(old_checkpoints, key=os.path.getctime)
        os.remove(oldest)
        print(f"Removed old checkpoint: {oldest}")


start_epoch, loss_history = load_checkpoint()

# -------- Training Loop -------- #
for epoch in range(start_epoch, num_epochs):
    epoch_start = time.time()
    print(f"Epoch {epoch+1}/{num_epochs}")
    model.train()
    print("Training...")
    total_loss = 0
    min_loss = float('inf')
    max_loss = float('-inf')
    running_loss = 0  # For last N iterations
    grad_norm = 0
    data_loading_time = 0
    compute_time = 0
    window_size = 50  # For running average

    iteration = 0
    batch_start = time.time()
    last_log_time = time.time()
    
    # Simple data loading like working CPU version
    for tokens, mels in loader:
        tokens = tokens.to(device, non_blocking=True)  # [B, T_text]
        mels = mels.to(device, non_blocking=True)  # [B, 80, T_mel]

        # Shift mel for teacher forcing
        decoder_input = mels[:, :, :-1].transpose(1, 2)  # [B, T_mel-1, 80]
        target = mels[:, :, 1:].transpose(1, 2)  # [B, T_mel-1, 80]

        # Forward pass
        pred_mels = model(tokens, decoder_input)  # [B, T_mel-1, 80]
        loss = loss_fn(pred_mels, target)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update metrics
        loss_val = loss.item()
        total_loss += loss_val
        min_loss = min(min_loss, loss_val)
        max_loss = max(max_loss, loss_val)
        running_loss = 0.95 * running_loss + 0.05 * loss_val if iteration > 0 else loss_val

        if iteration % 25 == 0:
            batch_time = time.time() - batch_start
            print(
                f"Iteration {iteration} - loss: {loss.item():.4f} - batch_time: {batch_time/25:.2f}s/batch"
            )
            batch_start = time.time()
        iteration += 1

    epoch_time = time.time() - epoch_start
    avg = total_loss / len(loader)
    loss_history.append(avg)
    
    # Calculate detailed metrics
    samples_per_sec = (len(loader) * batch_size) / epoch_time
    gpu_mem_used = torch.cuda.memory_allocated() / 1024**3
    gpu_mem_cached = torch.cuda.memory_reserved() / 1024**3
    current_lr = optimizer.param_groups[0]['lr']
    
    print(
        f"[Epoch {epoch+1}/{num_epochs}]"
        f"\n  Loss:"
        f"\n    Current: {avg:.4f}"
        f"\n    Running: {running_loss:.4f}"
        f"\n    Min: {min_loss:.4f}"
        f"\n    Max: {max_loss:.4f}"
        f"\n  Gradient norm: {grad_norm:.2f}"
        f"\n  Learning rate: {current_lr:.2e}"
        f"\n  Time: {epoch_time:.1f}s ({epoch_time/60:.1f}min)"
        f"\n  Throughput: {len(loader)/epoch_time:.1f} batches/sec ({samples_per_sec:.1f} samples/sec)"
        f"\n  GPU Memory: {gpu_mem_used:.1f}GB used, {gpu_mem_cached:.1f}GB cached"
        f"\n  ETA: {(num_epochs - epoch - 1) * epoch_time/60:.1f}min remaining"
    )

    # Step the learning rate scheduler
    scheduler.step(avg)

    if (epoch + 1) % save_every == 0 or epoch + 1 == num_epochs:
        save_checkpoint(epoch, loss_history)

    if epoch + 1 == num_epochs:
        model_path = "model/tts_transformer_latest.pth"
        if os.path.exists(model_path):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            backup_path = f"model/tts_transformer_{timestamp}.pth"
            os.rename(model_path, backup_path)
            print(f"📦 Existing model backed up as {backup_path}")

        torch.save(model.state_dict(), model_path)
        print(f"✅ Model saved as {model_path}")
