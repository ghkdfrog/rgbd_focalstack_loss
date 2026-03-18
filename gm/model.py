"""
SimpleCNN Energy-Based Model for Gradient Matching

모델 구조는 gradient_matching_focal.ipynb 에서 그대로 가져옴.
- 5-layer CNN, stride=1 전부, ReLU 활성화
- energy_head:
    'fc'     : Linear(256*512*512, 1) — 512x512 해상도 전용 (기존)
    'conv1x1': Conv2d(256,1,1) + spatial sum — 해상도 독립, 파라미터 극소
- diopter_mode: 'spatial' | 'coc'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint  # ✅ 1. 체크포인트 임포트


class ResidualBlock(nn.Module):
    """
    기본적인 2-Conv Residual Block. 
    공간 해상도와 채널 수를 유지합니다 (stride=1).
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return F.relu(out)



class SimpleCNN(nn.Module):
    def __init__(self, input_channels=7, diopter_mode='spatial', energy_head='fc'):
        super(SimpleCNN, self).__init__()
        self.diopter_mode = diopter_mode
        self.energy_head = energy_head

        # diopter_mode에 따라 입력 채널 수 결정
        if diopter_mode == 'spatial':
            in_ch = input_channels + 1  # RGBD(4) + Optim(3) + diopter(1) = 8
        elif diopter_mode == 'coc':
            in_ch = input_channels + 1  # RGBD(4) + Optim(3) + CoC(1) = 8
        else:
            in_ch = input_channels

        # Conv Layers: 공간 해상도 유지 (stride=1)
        self.conv1 = nn.Conv2d(in_ch, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        # Energy output head
        if energy_head == 'conv1x1':
            self.conv_energy = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        else:  # 'fc'
            self.fc = nn.Linear(256 * 512 * 512, 1)

    def forward(self, x, diopter):
        N, C, H, W = x.shape

        if self.diopter_mode == 'spatial':
            diopter_map = diopter.view(N, 1, 1, 1).expand(N, 1, H, W)
            x = torch.cat([x, diopter_map], dim=1)
        elif self.diopter_mode == 'coc':
            pass  # CoC는 이미 x에 포함되어 있음

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        if self.energy_head == 'conv1x1':
            x = self.conv_energy(x)           # (N, 1, H, W)
            x = torch.sum(x, dim=(2, 3))      # (N, 1) — 공간 에너지 합산
        else:  # 'fc'
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x


class SimpleCNNDeep(nn.Module):
    def __init__(self, input_channels=7, diopter_mode='spatial', energy_head='fc'):
        super(SimpleCNNDeep, self).__init__()
        self.diopter_mode = diopter_mode
        self.energy_head = energy_head

        # diopter_mode에 따라 입력 채널 수 결정
        if diopter_mode == 'spatial':
            in_ch = input_channels + 1
        elif diopter_mode == 'coc':
            in_ch = input_channels + 1
        else:
            in_ch = input_channels

        # Conv Layers: 공간 해상도 유지 (stride=1), 깊이 10레이어
        self.conv1 = nn.Conv2d(in_ch, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        # 256 채널 유지하며 깊이 추가
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Energy output head
        if energy_head == 'conv1x1':
            self.conv_energy = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        else:  # 'fc'
            self.fc = nn.Linear(256 * 512 * 512, 1)

    def forward(self, x, diopter):
        N, C, H, W = x.shape

        if self.diopter_mode == 'spatial':
            diopter_map = diopter.view(N, 1, 1, 1).expand(N, 1, H, W)
            x = torch.cat([x, diopter_map], dim=1)
        elif self.diopter_mode == 'coc':
            pass

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))

        if self.energy_head == 'conv1x1':
            x = self.conv_energy(x)           # (N, 1, H, W)
            x = torch.sum(x, dim=(2, 3))      # (N, 1)
        else:  # 'fc'
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x


class SimpleCNNStride(nn.Module):
    def __init__(self, input_channels=7, diopter_mode='spatial', energy_head='fc'):
        super(SimpleCNNStride, self).__init__()
        self.diopter_mode = diopter_mode
        self.energy_head = energy_head

        # diopter_mode에 따라 입력 채널 수 결정
        if diopter_mode == 'spatial':
            in_ch = input_channels + 1
        elif diopter_mode == 'coc':
            in_ch = input_channels + 1
        else:
            in_ch = input_channels

        # Conv Layers: 초반 1~3번째 레이어는 해상도(stride=1)를 유지하여 세밀한 로컬 텍스처(엣지 등) 특징 추출
        self.conv1 = nn.Conv2d(in_ch, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # 중후반 4~5번째 레이어에서 stride=2 를 사용하여 Receptive Field를 확장하고 글로벌 컨텍스트(광대역 블러 등)를 파악
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # 1/2 downsample
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # 1/4 downsample

        # Energy output head
        if energy_head == 'conv1x1':
            self.conv_energy = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        else:  # 'fc'
            # 입력 512x512 해상도 가정 시 1/4 로 줄어들어 128x128 이 됨
            self.fc = nn.Linear(256 * 128 * 128, 1)

    def forward(self, x, diopter):
        N, C, H, W = x.shape

        if self.diopter_mode == 'spatial':
            diopter_map = diopter.view(N, 1, 1, 1).expand(N, 1, H, W)
            x = torch.cat([x, diopter_map], dim=1)
        elif self.diopter_mode == 'coc':
            pass

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        if self.energy_head == 'conv1x1':
            x = self.conv_energy(x)           # (N, 1, H/4, W/4)
            x = torch.sum(x, dim=(2, 3))      # (N, 1) — 공간 에너지 합산
        else:  # 'fc'
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x


class SimpleResNet(nn.Module):
    """
    stride=1 을 유지하면서 깊이를 쌓기 위한 ResNet 구조.
    - 입력 채널을 128 (또는 256)까지 확장한 후 Residual Block 반복
    - fc 에너지 헤드 유지 가능
    """
    def __init__(self, input_channels=7, diopter_mode='spatial', energy_head='fc', num_blocks=4, channels=256):
        super(SimpleResNet, self).__init__()
        self.diopter_mode = diopter_mode
        self.energy_head = energy_head

        # diopter_mode에 따라 입력 채널 수 결정
        if diopter_mode == 'spatial':
            in_ch = input_channels + 1
        elif diopter_mode == 'coc':
            in_ch = input_channels + 1
        else:
            in_ch = input_channels

        # 초기 특징 추출 (해상도 유지)
        self.conv_in = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1)
        self.conv_expand = nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1)
        
        # Residual Blocks 쌓기 (stride=1 유지, 채널 256)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

        # Energy output head
        if energy_head == 'conv1x1':
            self.conv_energy = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0)
        else:  # 'fc'
            # 512x512 해상도 유지 시 파라미터 맞추기 위해 256 채널로 FC 레이어 구성 (기존 SimpleCNN과 동일: 약 67M)
            self.fc = nn.Linear(channels * 512 * 512, 1)

    def forward(self, x, diopter):
        N, C, H, W = x.shape

        if self.diopter_mode == 'spatial':
            diopter_map = diopter.view(N, 1, 1, 1).expand(N, 1, H, W)
            x = torch.cat([x, diopter_map], dim=1)
        elif self.diopter_mode == 'coc':
            pass

        x = F.relu(self.conv_in(x))
        x = F.relu(self.conv_expand(x))
        
        x = self.res_blocks(x)

        if self.energy_head == 'conv1x1':
            x = self.conv_energy(x)           # (N, 1, H, W)
            x = torch.sum(x, dim=(2, 3))      # (N, 1) 스칼라 합산
        else:  # 'fc'
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block (A ConvNet for the 2020s, CVPR 2022).
    7x7 Depthwise Conv → LayerNorm → Pointwise Expand → GELU → Pointwise Project.
    
    expansion=4 일 때 channels=256 기준 약 0.54M 파라미터.
    (기존 expansion=9는 OOM을 유발하므로 정석인 4로 원복하고 블록 수를 늘립니다)
    """
    def __init__(self, channels, expansion=4):
        super(ConvNeXtBlock, self).__init__()
        hidden_dim = expansion * channels  # 256 * 4 = 1024
        
        # 7x7 Depthwise Conv: 넓은 Receptive Field 확보 (groups=channels)
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        # LayerNorm (channel-last 방식)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        # Pointwise 확장 → GELU → Pointwise 축소 (Inverted Bottleneck)
        self.pwconv1 = nn.Linear(channels, hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, channels)

    def forward(self, x):
        residual = x
        # Depthwise Conv (NCHW)
        x = self.dwconv(x)
        # Channel-last로 변환하여 LayerNorm + Pointwise 적용
        x = x.permute(0, 2, 3, 1)   # (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)   # (N, C, H, W)
        return residual + x


class SimpleConvNeXt(nn.Module):
    """
    SimpleResNet과 동일한 구조에서 ResidualBlock → ConvNeXtBlock 교체.
    - Stem: 8ch → 64ch → 256ch (conv_in + conv_expand)
    - Body: ConvNeXtBlock × num_blocks (stride=1, 해상도 유지)
    - Head: fc (512×512 전용) 또는 conv1x1 (해상도 무관)
    
    expansion=4, num_blocks=9 기준:
    - ConvNeXt blocks: 9 × ~0.54M ≈ 4.86M
    - 총 파라미터: SimpleResNet(4블록, ~4.87M)과 거의 동일하게 일치!
    """
    def __init__(self, input_channels=7, diopter_mode='spatial', energy_head='fc', num_blocks=9):
        super(SimpleConvNeXt, self).__init__()
        self.diopter_mode = diopter_mode
        self.energy_head = energy_head

        if diopter_mode == 'spatial' or diopter_mode == 'coc':
            in_ch = input_channels + 1
        else:
            in_ch = input_channels

        # Stem: 입력 채널 → 64 → 256 (SimpleResNet과 동일)
        self.conv_in = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1)
        self.conv_expand = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)

        # ConvNeXt Blocks (stride=1 유지, 채널 256, expansion=4)
        self.blocks = nn.Sequential(
            *[ConvNeXtBlock(channels=256, expansion=4) for _ in range(num_blocks)]
        )

        # Energy output head
        if energy_head == 'conv1x1':
            self.conv_energy = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        else:  # 'fc'
            self.fc = nn.Linear(256 * 512 * 512, 1)

    def forward(self, x, diopter):
        N, C, H, W = x.shape

        if self.diopter_mode == 'spatial':
            diopter_map = diopter.view(N, 1, 1, 1).expand(N, 1, H, W)
            x = torch.cat([x, diopter_map], dim=1)
        elif self.diopter_mode == 'coc':
            pass

        x = F.relu(self.conv_in(x))
        x = F.relu(self.conv_expand(x))

        # Gradient Checkpointing: VRAM 절약을 위해 블록 단위로 체크포인트 적용
        for block in self.blocks:
            if x.requires_grad:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        if self.energy_head == 'conv1x1':
            x = self.conv_energy(x)
            x = torch.sum(x, dim=(2, 3))
        else:  # 'fc'
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x


class ConvNeXtUNet(nn.Module):
    """
    ConvNeXt의 강력한 확장성(expansion=4)을 유지하면서 512x512 해상도에서 발생하는 OOM을 피하기 위한 모델.
    - Stem(Downsample): 512x512 -> 256x256 으로 축소 (메모리 1/4 감소)
    - Body: 256x256 해상도에서 ConvNeXtBlock 수행 (expansion=4 사용 가능)
    - Upsample: 256x256 -> 512x512 복원 (pixel-perfect output 유지)
    """
    def __init__(self, input_channels=7, diopter_mode='spatial', energy_head='fc', num_blocks=9):
        super(ConvNeXtUNet, self).__init__()
        self.diopter_mode = diopter_mode
        self.energy_head = energy_head

        if diopter_mode == 'spatial' or diopter_mode == 'coc':
            in_ch = input_channels + 1
        else:
            in_ch = input_channels

        # 1. Stem & Downsample (512x512 -> 256x256)
        self.conv_in = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.Conv2d(64, 256, kernel_size=4, stride=2, padding=1) # 공간 1/2 축소, 채널 확장

        # 2. Body: ConvNeXt Blocks (256x256 at 256ch)
        self.blocks = nn.Sequential(
            *[ConvNeXtBlock(channels=256, expansion=4) for _ in range(num_blocks)]
        )

        # 3. Upsample (256x256 -> 512x512)
        # kernel=4, stride=2, padding=1 은 정확히 크기를 2배로 만듭니다.
        self.upsample = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.conv_out = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # 4. Energy output head (512x512 기준)
        if energy_head == 'conv1x1':
            self.conv_energy = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        else:  # 'fc'
            self.fc = nn.Linear(256 * 512 * 512, 1)

    def forward(self, x, diopter):
        N, C, H, W = x.shape

        if self.diopter_mode == 'spatial':
            diopter_map = diopter.view(N, 1, 1, 1).expand(N, 1, H, W)
            x = torch.cat([x, diopter_map], dim=1)
        elif self.diopter_mode == 'coc':
            pass

        # Downsample
        x_stem = F.relu(self.conv_in(x))
        x_down = F.relu(self.downsample(x_stem))

        # Body
        x_body = x_down
        for block in self.blocks:
            if x_body.requires_grad:
                x_body = checkpoint(block, x_body, use_reentrant=False)
            else:
                x_body = block(x_body)

        # Upsample (+ Res connection optional, skipped here for simplicity)
        x_up = F.relu(self.upsample(x_body))
        x_out = F.relu(self.conv_out(x_up))

        # Head
        if self.energy_head == 'conv1x1':
            eng = self.conv_energy(x_out)
            eng = torch.sum(eng, dim=(2, 3))
        else:  # 'fc'
            eng = torch.flatten(x_out, 1)
            eng = self.fc(eng)
            
        return eng


class DilatedNet(nn.Module):
    def __init__(self, input_channels=7, diopter_mode='spatial', energy_head='fc'):
        super(DilatedNet, self).__init__()
        self.diopter_mode = diopter_mode
        self.energy_head = energy_head

        if diopter_mode == 'spatial' or diopter_mode == 'coc':
            in_ch = input_channels + 1
        elif diopter_mode == 'coc_abs':
            in_ch = input_channels + 2
        else:
            in_ch = input_channels

        # Stem: 입력 채널 → 64 → 256 (SimpleResNet과 동일)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )

        # Dense Blocs
        self.block1 = self._block(64, dilation=1)
        self.block2 = self._block(64, dilation=1)

        # Dilated blocks
        self.block3 = self._block(64, dilation=2)
        self.block4 = self._block(64, dilation=4)
        self.block5 = self._block(64, dilation=8)

        # CoC modulation
        self.gamma = nn.Conv2d(1, 64, 1)
        self.beta = nn.Conv2d(1, 64, 1)

        # Energy output head
        if energy_head == 'conv1x1':
            self.conv_energy = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        else:  # 'fc'
            self.fc = nn.Linear(64 * 512 * 512, 1)

    def _block(self, ch, dilation=1):
        return nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
        )

    def forward(self, x, diopter):
        N, C, H, W = x.shape

        diopter_map = diopter.view(N, 1, 1, 1).expand(N, 1, H, W)

        if self.diopter_mode == 'spatial':
            x = torch.cat([x, diopter_map], dim=1)
        elif self.diopter_mode == 'coc':
            pass
        elif self.diopter_mode == 'coc_abs':
            diopter_abs = torch.abs(diopter_map)
            x = torch.cat([x, diopter_abs], dim=1)

        f = self.stem(x)

        # dense
        f = f + self.block1(f)
        f = f + self.block2(f)

        # dilated + CoC conditioning
        for block in [self.block3, self.block4, self.block5]:
            res = f
            f = block(f)

            gamma = self.gamma(diopter_map)
            beta = self.beta(diopter_map)
            f = gamma * f + beta

            f = f + res

        # head
        if self.energy_head == 'conv1x1':
            f = self.conv_energy(f)
            f = f.sum()
        else:  # 'fc'
            f = torch.flatten(f, 1)
            f = self.fc(f)
        return f


def save_model_architecture(model, save_path, args=None):
    """모델 구조와 파라미터 수를 .txt 파일로 저장"""
    lines = []
    lines.append("=" * 60)
    lines.append("Model Architecture: SimpleCNN")
    lines.append("=" * 60)

    if args is not None:
        lines.append(f"diopter_mode : {args.diopter_mode}")
        lines.append(f"energy_head  : {getattr(args, 'energy_head', 'fc')}")
        lines.append(f"input_channels: 7")
        lines.append("")

    lines.append(str(model))
    lines.append("")
    lines.append("-" * 60)

    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        lines.append(f"  {name:40s}  shape={str(list(param.shape)):20s}  numel={param.numel():>15,}")

    lines.append("-" * 60)
    lines.append(f"  Total parameters    : {total_params:>15,}")
    lines.append(f"  Trainable parameters: {trainable_params:>15,}")
    lines.append("=" * 60)

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
