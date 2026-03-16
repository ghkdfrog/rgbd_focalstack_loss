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
    def __init__(self, input_channels=7, diopter_mode='spatial', energy_head='fc', num_blocks=4):
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
        self.conv_expand = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        
        # Residual Blocks 쌓기 (stride=1 유지, 채널 256)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(num_blocks)]
        )

        # Energy output head
        if energy_head == 'conv1x1':
            self.conv_energy = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        else:  # 'fc'
            # 512x512 해상도 유지 시 파라미터 맞추기 위해 256 채널로 FC 레이어 구성 (기존 SimpleCNN과 동일: 약 67M)
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
        
        x = self.res_blocks(x)

        if self.energy_head == 'conv1x1':
            x = self.conv_energy(x)           # (N, 1, H, W)
            x = torch.sum(x, dim=(2, 3))      # (N, 1) 스칼라 합산
        else:  # 'fc'
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x


class DilatedResidualBlock(nn.Module):
    """
    공간 해상도를 유지하면서 넓은 Receptive Field를 보기 위한 팽창 합성곱 블록
    """
    def __init__(self, channels, dilation=2):
        super(DilatedResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, 
                               padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return F.relu(out)


class SEResidualBlock(nn.Module):
    """
    특징 채널(Coc 조건 등)에 가중치를 주는 Squeeze-and-Excitation 블록
    """
    def __init__(self, channels, reduction=16):
        super(SEResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        
        se_weight = self.se(out)
        out = out * se_weight
        
        out += residual
        return F.relu(out)


class ConvNeXtBlock(nn.Module):
    """
    표준 ConvNeXt Block (expansion=4)
    메모리 폭발을 막기 위해 expansion을 9에서 4로 줄였습니다.
    """
    def __init__(self, channels, expansion=4):
        super(ConvNeXtBlock, self).__init__()
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        
        # expansion=4로 설정하여 VRAM 사용량을 대폭 낮춤
        self.pwconv1 = nn.Linear(channels, expansion * channels)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expansion * channels, channels)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) 
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return residual + x


class SimpleConvNeXt(nn.Module):
    # 1. 괄호 안에 num_blocks=4 를 다시 추가해 줍니다!
    def __init__(self, input_channels=7, diopter_mode='spatial', energy_head='fc', num_blocks=4):
        super(SimpleConvNeXt, self).__init__()
        self.diopter_mode = diopter_mode
        self.energy_head = energy_head

        if diopter_mode == 'spatial' or diopter_mode == 'coc':
            in_ch = input_channels + 1
        else:
            in_ch = input_channels

        self.stem = nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1)
        
        self.stage1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            ConvNeXtBlock(channels=64, expansion=4)
        )
        
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            ConvNeXtBlock(channels=128, expansion=4)
        )
        
        # 2. Stage 3 수정: 전달받은 num_blocks 만큼 ConvNeXt 블록을 반복해서 쌓습니다.
        stage3_layers = [
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        ]
        for _ in range(num_blocks):
            stage3_layers.append(ConvNeXtBlock(channels=256, expansion=4))
            
        self.stage3 = nn.Sequential(*stage3_layers)

        # 3. Energy output head (그대로 유지)
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

        x = F.relu(self.stem(x))
        x = self.stage1(x)
        x = self.stage2(x)
        
        # ✅ 2. 가장 메모리를 많이 먹는 Stage 3에 Gradient Checkpointing 적용!
        # 기존: x = self.stage3(x) 
        # 변경: 블록을 하나씩 꺼내서 checkpoint로 감싸서 실행합니다.
        for module in self.stage3:
            # 텐서가 gradient를 요구할 때만(학습 중일 때만) checkpoint 작동
            if x.requires_grad:
                x = checkpoint(module, x, use_reentrant=False)
            else:
                x = module(x)

        if self.energy_head == 'conv1x1':
            x = self.conv_energy(x)
            x = torch.sum(x, dim=(2, 3))
        else:  # 'fc'
            x = torch.flatten(x, 1)
            x = self.fc(x)
            
        return x


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
