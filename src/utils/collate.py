import torch
import torch.nn.functional as F


def pad_mel(mel, max_len):
    # mel: (n_mels, T)
    T = mel.shape[1]
    if T < max_len:
        pad_amt = max_len - T
        mel = F.pad(mel, (0, pad_amt), mode="constant", value=0.0)
    return mel


def pad_1d(x, max_len):
    return F.pad(x, (0, max_len - x.shape[0]), value=0)


def collate_fn(batch):
    tokens, mels = zip(*batch)

    max_token_len = max([t.shape[0] for t in tokens])
    max_mel_len = max([m.shape[1] for m in mels])

    padded_tokens = torch.stack([pad_1d(t, max_token_len) for t in tokens])
    padded_mels = torch.stack([pad_mel(m, max_mel_len) for m in mels])

    return padded_tokens, padded_mels
