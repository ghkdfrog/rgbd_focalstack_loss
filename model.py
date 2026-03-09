"""
Loss Estimation Model Architecture

Input:
  - RGBD: (4, H, W) - clean_pass_rgb (3ch) + depth (1ch)
  - Predicted focal plane: (3, H, W) - single focal plane from DeepFocus
  - Diopter: scalar float

Output:
  - PSNR prediction: scalar
  - SSIM prediction: scalar
  - LPIPS prediction: scalar

diopter_mode:
  'spatial'    (default) - diopter broadcast to 512x512, added as 8th conv channel
  'sinusoidal' (new)     - sinusoidal positional encoding, concat after GAP -> FC
  'coc'        (new)     - Circle of Confusion map from depth+diopter, 8th conv channel
                           (follows original DeepFocus LVF architecture)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as sn
import numpy as np


def calCoC(depth_map, dp_focal,
           film_len_dist=0.017, fov=0.7854,
           D=0.004, cocScale=30.0):
    """Compute Circle of Confusion map (port of deepfocus_pytorch.py calCoC).

    Args:
        depth_map: (N, 1, H, W) torch tensor, depth in meters (normalized 0-1 from /12.0)
        dp_focal:  (N,) torch tensor, query diopter value
        film_len_dist: focal length of virtual lens in meters
        fov:       horizontal field of view in radians
        D:         pupil diameter in meters
        cocScale:  normalization scale for output

    Returns:
        coc: (N, 1, H, W) CoC map, approximate range [0, 1]
    """
    # Recover approximate metric depth (reverse of depth/12.0 normalization)
    depth_m = depth_map * 12.0 + 1e-6          # (N, 1, H, W) in meters
    dp_dm = 1.0 / depth_m                       # depth diopter map

    dp_focal = dp_focal.view(-1, 1, 1, 1)       # (N, 1, 1, 1)
    dp_fl = 1.0 / film_len_dist + dp_focal       # effective focal diopter

    # Physical CoC formula
    coc_physical = D * torch.abs(
        (dp_fl - dp_dm) / (dp_fl - dp_focal + 1e-8) - 1.0
    )

    # Convert to pixels and normalize
    film_width = 2.0 * film_len_dist * np.tan(fov / 2.0)
    # Use a fixed W=512 reference (matches training data)
    coc = coc_physical / film_width * 512.0 / cocScale
    return coc.clamp(0.0, 1.0)                  # (N, 1, H, W)


class SinusoidalEncoding(nn.Module):
    """Encode a scalar value into a sinusoidal positional encoding vector.

    Given diopter d (shape: (N,)), returns a vector of shape (N, 2*freqs)
    containing [sin(2^0 * pi * d), cos(2^0 * pi * d), ...,
                sin(2^(L-1) * pi * d), cos(2^(L-1) * pi * d)]
    """
    def __init__(self, freqs=8, max_d=4.0):
        super().__init__()
        self.freqs = freqs
        self.max_d = max_d
        # Register frequency bands as buffer (no gradient)
        freq_bands = 2.0 ** torch.arange(freqs).float() * np.pi
        self.register_buffer('freq_bands', freq_bands)  # (L,)

    @property
    def out_dim(self):
        return 2 * self.freqs

    def forward(self, d):
        """d: (N,) -> (N, 2*freqs)"""
        d_norm = d / self.max_d  # normalize to ~[0, 1]
        # (N, 1) * (1, L) -> (N, L)
        angles = d_norm.unsqueeze(1) * self.freq_bands.unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # (N, 2L)

class LossEstimationNet(nn.Module):
    def __init__(self, input_channels=7, use_spectral_norm=False,
                 diopter_mode='spatial', sin_freqs=8, version='v2'):
        """
        Args:
            input_channels: RGBD (4) + predicted focal plane (3) = 7
            use_spectral_norm: If True, apply spectral normalization
            diopter_mode: 'spatial' (broadcast, backward-compat) |
                          'sinusoidal' (encoding concat at FC stage)
            sin_freqs: Number of sinusoidal frequency bands (L).
                       Only used when diopter_mode='sinusoidal'.
        """
        super().__init__()
        self.use_spectral_norm = use_spectral_norm
        self.diopter_mode = diopter_mode
        self.version = version
        _wrap = sn if use_spectral_norm else lambda x: x

        if diopter_mode == 'spatial':
            # Original: diopter broadcast as 8th conv channel
            conv1_in = input_channels + 1  # 8
            fc_in = 512
        elif diopter_mode == 'sinusoidal':
            # New: only image goes through conv (7ch), diopter added at FC
            conv1_in = input_channels      # 7
            self.diopter_enc = SinusoidalEncoding(freqs=sin_freqs)
            fc_in = 512 + self.diopter_enc.out_dim  # 512 + 2*L
        elif diopter_mode == 'coc':
            # CoC map computed from depth+diopter, added as 8th conv channel
            # (follows original DeepFocus LVF architecture)
            conv1_in = input_channels + 1  # 8
            fc_in = 512
        else:
            raise ValueError(f"Unknown diopter_mode: {diopter_mode}")

        # Feature extraction backbone (ResNet-like)
        self.conv1 = _wrap(nn.Conv2d(conv1_in, 64, 7, stride=2, padding=3))
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 128, num_blocks=2, stride=2,
                                       use_sn=use_spectral_norm)
        self.layer2 = self._make_layer(128, 256, num_blocks=2, stride=2,
                                       use_sn=use_spectral_norm)
        self.layer3 = self._make_layer(256, 512, num_blocks=2, stride=2,
                                       use_sn=use_spectral_norm)

        # Global pooling to handle arbitrary input sizes
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Multi-head outputs
        # Multi-head outputs
        if self.version == 'v1':
            self.fc_psnr = _wrap(nn.Linear(fc_in, 1))
            self.fc_ssim = _wrap(nn.Linear(fc_in, 1))
            self.fc_lpips = _wrap(nn.Linear(fc_in, 1))
        elif self.version == 'v2':
            # Multi-head outputs (Upgraded to 2-layer MLP for sharper non-linear capacity)
            self.fc_psnr = nn.Sequential(
                _wrap(nn.Linear(fc_in, 128)),
                nn.LeakyReLU(0.2, inplace=True),
                _wrap(nn.Linear(128, 1))
            )
            self.fc_ssim = nn.Sequential(
                _wrap(nn.Linear(fc_in, 128)),
                nn.LeakyReLU(0.2, inplace=True),
                _wrap(nn.Linear(128, 1))
            )
            self.fc_lpips = nn.Sequential(
                _wrap(nn.Linear(fc_in, 128)),
                nn.LeakyReLU(0.2, inplace=True),
                _wrap(nn.Linear(128, 1))
            )
        else:
            raise ValueError(f"Unknown version: {self.version}")
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride,
                    use_sn=False):
        """Create a layer with residual blocks"""
        layers = []
        
        # First block with downsampling
        layers.append(ResidualBlock(in_channels, out_channels, stride,
                                    use_spectral_norm=use_sn))
        
        # Remaining blocks
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels, 1,
                                        use_spectral_norm=use_sn))
        
        return nn.Sequential(*layers)
    
    def forward(self, x, diopter):
        """
        Args:
            x: (N, 7, H, W) - RGBD + predicted focal plane
            diopter: (N,) - scalar diopter values

        Returns:
            dict with 'psnr', 'ssim', 'lpips' predictions
        """
        N, _, H, W = x.shape

        if self.diopter_mode == 'spatial':
            # Original: broadcast diopter as 8th channel
            diopter_map = diopter.view(N, 1, 1, 1).expand(N, 1, H, W)
            x = torch.cat([x, diopter_map], dim=1)  # (N, 8, H, W)
        # coc mode: CoC already pre-computed by dataset as 8th channel → x is (N, 8, H, W)
        # sinusoidal mode: x stays (N, 7, H, W), diopter added after GAP


        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Global pooling
        feat = self.gap(x).view(N, -1)  # (N, 512)

        if self.diopter_mode == 'sinusoidal':
            # Encode diopter and concat to features
            d_enc = self.diopter_enc(diopter)   # (N, 2L)
            feat = torch.cat([feat, d_enc], dim=1)  # (N, 512+2L)

        # Multi-head predictions
        # PSNR: normalized to [0, 1] (multiply by 100 to get dB)
        psnr_pred = torch.sigmoid(self.fc_psnr(feat))  # (N, 1)

        # SSIM: [0, 1]
        ssim_pred = torch.sigmoid(self.fc_ssim(feat))  # (N, 1)

        # LPIPS: [0, 1]
        lpips_pred = torch.sigmoid(self.fc_lpips(feat))  # (N, 1)

        return {
            'psnr': psnr_pred,
            'ssim': ssim_pred,
            'lpips': lpips_pred
        }

class ResidualBlock(nn.Module):
    """Basic residual block"""
    def __init__(self, in_channels, out_channels, stride,
                 use_spectral_norm=False):
        super().__init__()
        _wrap = sn if use_spectral_norm else lambda x: x
        
        self.conv1 = _wrap(nn.Conv2d(in_channels, out_channels, 3,
                                     stride=stride, padding=1))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = _wrap(nn.Conv2d(out_channels, out_channels, 3,
                                     stride=1, padding=1))
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                _wrap(nn.Conv2d(in_channels, out_channels, 1, stride=stride)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        
        return out

if __name__ == "__main__":
    batch_size = 4
    x = torch.randn(batch_size, 7, 512, 512)
    diopter = torch.tensor([0.5, 1.0, 2.0, 3.5])

    print("=== Spatial mode (backward-compat) ===")
    model_spatial = LossEstimationNet(diopter_mode='spatial')
    out = model_spatial(x, diopter)
    print(f"Input shape: {x.shape}")
    print(f"PSNR: {out['psnr']}, SSIM: {out['ssim']}, LPIPS: {out['lpips']}")
    total = sum(p.numel() for p in model_spatial.parameters())
    print(f"Parameters: {total:,}")

    print("\n=== Sinusoidal mode ===")
    model_sin = LossEstimationNet(diopter_mode='sinusoidal', sin_freqs=8)
    out2 = model_sin(x, diopter)
    print(f"PSNR: {out2['psnr']}, SSIM: {out2['ssim']}, LPIPS: {out2['lpips']}")
    total2 = sum(p.numel() for p in model_sin.parameters())
    print(f"Parameters: {total2:,}")

    print("\n=== CoC mode ===")
    model_coc = LossEstimationNet(diopter_mode='coc')
    out3 = model_coc(x, diopter)
    print(f"PSNR: {out3['psnr']}, SSIM: {out3['ssim']}, LPIPS: {out3['lpips']}")
    total3 = sum(p.numel() for p in model_coc.parameters())
    print(f"Parameters: {total3:,}")
