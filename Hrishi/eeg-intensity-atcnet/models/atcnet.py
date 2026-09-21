"""
PyTorch re-implementation of ATCNet (Altaheri, Muhammad & Alsulaiman, 2022 --
"Physics-informed attention temporal convolutional network for EEG-based motor imagery
classification", IEEE TII), adapted for **regression** (predicting a continuous/ordinal
stimulus intensity instead of classifying a discrete motor-imagery class).

Two variants are provided:

  ATCNetRegressor
      The real thing: consumes raw multi-channel EEG epochs, shape (batch, channels, time).
      Conv block (temporal -> depthwise spatial -> separable) turns the raw signal into a
      short temporal sequence of embeddings; that sequence is split into overlapping sliding
      windows; each window goes through multi-head self-attention (MSA) then a small
      temporal convolutional network (TCN); each window produces a scalar regression
      output via its own dense head; final prediction is the mean over windows (as in the
      original classifier, which averages per-window softmax logits instead of concatenating).

  TabularATCNetRegressor
      Same block structure (temporal conv -> "spatial" conv over channels -> MSA -> TCN),
      but the input is Aditya's per-trial hand-crafted feature matrix reshaped to
      (batch, n_channels, n_features_per_channel) -- i.e. the "sequence" axis is features,
      not time. This is a stand-in for when raw EEG isn't available; it is not a substitute
      for the raw-signal model and should be described as such in any writeup.

Loss: use nn.SmoothL1Loss (Huber) or nn.MSELoss with the outputs of either model; both
return a single scalar per trial (batch,) -- no output activation, so pair with a loss that
expects raw regression targets (normalize targets to comparable scale across datasets, e.g.
the `y_relative` column that build_epochs.py writes out).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBlockRaw(nn.Module):
    """Temporal -> depthwise spatial -> separable conv block, operating on raw EEG (B,1,C,T)."""

    def __init__(self, n_channels: int, F1: int = 16, D: int = 2, F2: int | None = None,
                 kern_length: int = 32, pool1: int = 4, pool2: int = 4, dropout: float = 0.3):
        super().__init__()
        F2 = F2 or F1 * D
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kern_length), padding=(0, kern_length // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, pool1)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, pool2)),
            nn.Dropout(dropout),
        )
        self.out_channels = F2

    def forward(self, x):  # x: (B, 1, C, T)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        return x  # (B, F2, 1, T')


class _ConvBlockTabular(nn.Module):
    """Analogous block for tabular features: (B, 1, C, feat) -> (B, F2, 1, feat')."""

    def __init__(self, n_channels: int, F1: int = 16, D: int = 2, F2: int | None = None,
                 kern_length: int = 5, pool1: int = 2, pool2: int = 2, dropout: float = 0.3):
        super().__init__()
        F2 = F2 or F1 * D
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kern_length), padding=(0, kern_length // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, pool1)) if pool1 > 1 else nn.Identity(),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 3), padding=(0, 1), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, pool2)) if pool2 > 1 else nn.Identity(),
            nn.Dropout(dropout),
        )
        self.out_channels = F2

    def forward(self, x):  # x: (B, 1, C, feat)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        return x


class _TemporalConvLayer(nn.Module):
    """One dilated causal conv layer of a TCN, with residual connection."""

    def __init__(self, channels: int, kernel_size: int = 4, dilation: int = 1, dropout: float = 0.3):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.ELU()
        self.drop = nn.Dropout(dropout)
        self.pad = pad

    def forward(self, x):  # x: (B, C, T)
        out = self.conv(x)
        if self.pad > 0:
            out = out[:, :, :-self.pad]  # causal: trim right-side padding
        out = self.act(self.bn(out))
        out = self.drop(out)
        return x + out


class _TCNBlock(nn.Module):
    def __init__(self, channels: int, n_layers: int = 2, kernel_size: int = 4, dropout: float = 0.3):
        super().__init__()
        self.layers = nn.Sequential(*[
            _TemporalConvLayer(channels, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(n_layers)
        ])

    def forward(self, x):  # x: (B, C, T)
        return self.layers(x)


class _AttentionBranch(nn.Module):
    """MSA over the window's time axis, followed by a TCN, followed by a scalar regression head."""

    def __init__(self, embed_dim: int, n_heads: int = 4, tcn_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.msa = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)
        self.tcn = _TCNBlock(embed_dim, n_layers=tcn_layers, dropout=dropout)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x):  # x: (B, T_window, E)
        attn_out, _ = self.msa(x, x, x, need_weights=False)
        x = self.ln(x + attn_out)
        x = x.transpose(1, 2)          # (B, E, T_window) for conv1d
        x = self.tcn(x)
        last = x[:, :, -1]             # (B, E) -- last timestep, as in the original ATCNet
        return self.head(last).squeeze(-1)  # (B,)


class _ATCNetCore(nn.Module):
    def __init__(self, conv_block: nn.Module, n_windows: int = 3, n_heads: int = 4,
                 tcn_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.conv_block = conv_block
        self.n_windows = n_windows
        embed_dim = conv_block.out_channels
        self.branches = nn.ModuleList([
            _AttentionBranch(embed_dim, n_heads=n_heads, tcn_layers=tcn_layers, dropout=dropout)
            for _ in range(n_windows)
        ])

    def forward(self, x):  # x: (B, 1, C, L)
        feat = self.conv_block(x)               # (B, E, 1, T')
        feat = feat.squeeze(2).transpose(1, 2)   # (B, T', E)
        T = feat.shape[1]
        if T <= self.n_windows:
            windows = [feat for _ in range(self.n_windows)]
        else:
            win_len = T - self.n_windows + 1
            windows = [feat[:, i:i + win_len, :] for i in range(self.n_windows)]

        preds = torch.stack([branch(w) for branch, w in zip(self.branches, windows)], dim=0)
        return preds.mean(dim=0)  # (B,) -- average over sliding-window branches


class ATCNetRegressor(nn.Module):
    """Raw-EEG ATCNet adapted to regression. Input: (batch, n_channels, n_samples)."""

    def __init__(self, n_channels: int, n_samples: int, F1: int = 16, D: int = 2,
                 n_windows: int = 3, n_heads: int = 4, tcn_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        conv_block = _ConvBlockRaw(n_channels, F1=F1, D=D, dropout=dropout)
        self.core = _ATCNetCore(conv_block, n_windows=n_windows, n_heads=n_heads,
                                 tcn_layers=tcn_layers, dropout=dropout)

    def forward(self, x):  # x: (B, C, T)
        x = x.unsqueeze(1)  # (B, 1, C, T)
        return self.core(x)


class TabularATCNetRegressor(nn.Module):
    """
    ATCNet-inspired CNN for pre-extracted tabular features, reshaped to
    (batch, n_channels, n_features_per_channel) -- e.g. 3 channels (C3, Cz, C4) x ~45 features
    each, as produced by Aditya's feature-extraction pipeline. NOT a substitute for running
    ATCNet on raw EEG; use only when raw epochs aren't available.
    """

    def __init__(self, n_channels: int, n_features: int, F1: int = 16, D: int = 2,
                 n_windows: int = 2, n_heads: int = 4, tcn_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        conv_block = _ConvBlockTabular(n_channels, F1=F1, D=D, dropout=dropout)
        self.core = _ATCNetCore(conv_block, n_windows=n_windows, n_heads=n_heads,
                                 tcn_layers=tcn_layers, dropout=dropout)

    def forward(self, x):  # x: (B, n_channels, n_features)
        x = x.unsqueeze(1)  # (B, 1, C, feat)
        return self.core(x)


if __name__ == "__main__":
    # quick shape sanity check (not a real test -- run this once torch is available)
    raw_model = ATCNetRegressor(n_channels=9, n_samples=256)
    x = torch.randn(4, 9, 256)
    print("raw output shape:", raw_model(x).shape)  # expect (4,)

    tab_model = TabularATCNetRegressor(n_channels=3, n_features=45)
    x2 = torch.randn(4, 3, 45)
    print("tabular output shape:", tab_model(x2).shape)  # expect (4,)
