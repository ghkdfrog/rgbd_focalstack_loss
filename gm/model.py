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
