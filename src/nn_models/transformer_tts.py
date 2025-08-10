import math

import torch.nn as nn

from .positional_encoding import PositionalEncoding


class TransformerTTS(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        nhead=4,
        num_layers=4,
        dim_feedforward=512,
        mel_bins=80,
    ):
        super().__init__()
        self.d_model = d_model
        self.mel_bins = mel_bins

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.mel_embed = nn.Linear(mel_bins, d_model)

        self.pos_enc = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True,
        )

        self.fc_out = nn.Linear(d_model, mel_bins)

    def forward(self, src, tgt_mel):
        # Embedding + Positional Encoding
        src_embed = self.token_embed(src) * math.sqrt(self.d_model)
        src_embed = self.pos_enc(src_embed)

        tgt_embed = self.mel_embed(tgt_mel)
        tgt_embed = self.pos_enc(tgt_embed)

        # Auto-regressive masking
        tgt_len = tgt_embed.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(
            tgt_embed.device
        )

        output = self.transformer(
            src=src_embed,
            tgt=tgt_embed,
            tgt_mask=tgt_mask,
        )

        return self.fc_out(output)  # [B, T_mel-1, mel_bins]
