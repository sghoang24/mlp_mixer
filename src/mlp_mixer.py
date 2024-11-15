"""Mixer model."""

import torch
from enum import Enum
import torch.nn as nn

from utils.model_info import MODEL_INFO


class MLPBlock(nn.Module):
    """MLP Block."""

    def __init__(self, hidden_dim, mlp_dim):
        super(MLPBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        """Forward."""
        return self.mlp(x)


class MixerBlock(nn.Module):
    """Mixer Block."""

    def __init__(self, num_tokens, hidden_dim, tokens_mlp_dim, channels_mlp_dim):
        super(MixerBlock, self).__init__()
        # Token mixer
        self.ln_token = nn.LayerNorm(hidden_dim)
        self.token_mix = MLPBlock(num_tokens, tokens_mlp_dim)

        # Channel mixer
        self.ln_channel = nn.LayerNorm(hidden_dim)
        self.channel_mix = MLPBlock(hidden_dim, channels_mlp_dim)

    def forward(self, x):
        """Forward."""
        out = self.ln_token(x).transpose(1, 2)
        x = x + self.token_mix(out).transpose(1, 2)
        out = self.ln_channel(x)
        x = x + self.channel_mix(out)
        return x


class MLPMixer(nn.Module):
    """MLP Mixer."""
    def __init__(
        self,
        num_classes,
        num_blocks,
        patch_size,
        hidden_dim,
        tokens_mlp_dim,
        channels_mlp_dim,
        image_size: int = 224,
    ):
        super(MLPMixer, self).__init__()
        num_tokens = (image_size // patch_size)**2
        self.num_channels = num_channels

        self.patch_emb = nn.Conv2D(num_channels, hidden_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.mlp = nn.Sequential(
            *[
                MixerBlock(num_tokens, hidden_dim, tokens_mlp_dim, channels_mlp_dim)
                for _ in range(num_blocks)
            ]
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self):
        """Forward."""
        # x shape = (3, image_size, image_size) 
        x = self.patch_emb(x)               # (batch_size, hidden_dim, patch_size, patch_size)
        x = x.flatten(2).transpose(1, 2)    # (batch_size, num_tokens, hidden_dim)
        x = self.mlp(x)                     # (batch_size, num_tokens, hidden_dim)
        x = self.ln(x)                      # (batch_size, hidden_dim)
        x = x.mean(dim=1)                   # (batch_size, hidden_dim)
        x = self.fc(x)                      # (batch_size, num_classes)


def mixer(model: str, num_classes: int = 1000, **kwargs):
    """Mixer model."""
    model_info = MODEL_INFO.get(model, None)
    image_size = kwargs.get('image_size', 224)

    if not model_info:
        return MLPMixer(num_classes=num_classes, **kwargs)
    return MLPMixer(
        num_classes=num_classes,
        image_size=image_size,
        **model_info,
    )
