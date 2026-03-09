"""
Energy-Based Score Generation Model
Predicts a single scalar energy given RGBD(4ch) + optimizing_rgb(3ch) + diopter
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Re-use utilities from model.py
from model import SinusoidalEncoding, calCoC

def conv_block(in_ch, out_ch, kernel_size=3, stride=1, padding=1, use_sn=False):
    layers = []
    conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
    if use_sn:
        conv = nn.utils.spectral_norm(conv)
    layers.append(conv)
    # BatchNorm removed for better stability in EBM gradient matching context
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

class EnergyNet(nn.Module):
    def __init__(self, input_channels=7, use_spectral_norm=False,
                 diopter_mode='spatial', sin_freqs=8):
        """
        Args:
            input_channels: RGBD (4) + current focal plane (3) = 7
            use_spectral_norm: If True, apply spectral normalization
            diopter_mode: 'spatial' | 'coc' | 'sinusoidal'
            sin_freqs: Frequencies for sinusoidal encoding
        """
        super().__init__()
        self.diopter_mode = diopter_mode
        self.use_spectral_norm = use_spectral_norm

        if diopter_mode == 'spatial':
            in_ch = input_channels + 1
        elif diopter_mode == 'coc':
            in_ch = input_channels + 1 # RGBD(4) + OptimRGB(3) + CoC(1) = 8
        elif diopter_mode == 'sinusoidal':
            in_ch = input_channels
            self.diopter_enc = SinusoidalEncoding(freqs=sin_freqs)
        else:
            raise ValueError(f"Unknown diopter mode {diopter_mode}")

        # Convolutional Backbone (3 Layers)
        # Reverted to stride=1 as requested by the user to match the successful notebook version.
        self.layer1 = conv_block(in_ch, 64, stride=1, use_sn=use_spectral_norm)
        self.layer2 = conv_block(64, 128, stride=1, use_sn=use_spectral_norm)
        self.layer3 = conv_block(128, 256, stride=1, use_sn=use_spectral_norm)

        # No pooling to maintain spatial resolution for gradient matching.
        # With 256x256 input and all stride 1, the output is (256, 256, 256).
        fc_in = 256 * 512 * 512
        if diopter_mode == 'sinusoidal':
            fc_in += self.diopter_enc.out_dim

        self.fc_energy = nn.Linear(fc_in, 1)

    def forward(self, x, diopter):
        """
        Args:
            x: (N, 7, H, W) - RGBD + current optimizing focal plane
            diopter: (N,) - scalar diopter values
        Returns:
            energy: (N, 1) scalar energy
        """
        N, C, H, W = x.shape

        if self.diopter_mode == 'spatial':
            diopter_map = diopter.view(N, 1, 1, 1).expand(N, 1, H, W)
            x = torch.cat([x, diopter_map], dim=1)
        elif self.diopter_mode == 'coc':
            pass # CoC is already in x from train_score.py

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        feat = x.view(N, -1)

        if self.diopter_mode == 'sinusoidal':
            d_enc = self.diopter_enc(diopter)
            feat = torch.cat([feat, d_enc], dim=1)

        energy = self.fc_energy(feat)  # (N, 1)

        return energy
