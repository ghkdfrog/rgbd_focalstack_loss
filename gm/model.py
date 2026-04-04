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



class ResidualBlock(nn.Module):
    """
    기본적인 2-Conv Residual Block. 
    공간 해상도와 채널 수를 유지합니다 (stride=1).
    """
    def __init__(self, channels, activation='relu'):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.SiLU() if activation == 'silu' else nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return self.act(out)



class SimpleCNN(nn.Module):
    def __init__(self, input_channels=7, diopter_mode='spatial', energy_head='fc'):
        super(SimpleCNN, self).__init__()
        self.diopter_mode = diopter_mode
        self.energy_head = energy_head

        # diopter_mode에 따라 입력 채널 수 결정
        if diopter_mode in ['spatial', 'coc', 'coc_signed', 'coc_abs']:
            in_ch = input_channels + 1
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
        elif self.diopter_mode in ['coc', 'coc_abs', 'coc_signed']:
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
        if diopter_mode in ['spatial', 'coc', 'coc_signed', 'coc_abs']:
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
        elif self.diopter_mode in ['coc', 'coc_abs', 'coc_signed']:
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
        if diopter_mode in ['spatial', 'coc', 'coc_signed', 'coc_abs']:
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
        elif self.diopter_mode in ['coc', 'coc_abs', 'coc_signed']:
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
    - use_film=True 이면 FiLMResidualBlock 사용 (CoC/diopter 기반 FiLM conditioning)
    - long_skip=True 이면 블록 간 1-layer interval element-wise add skip connection 적용
      (DeepFocus 스타일: block[i] input = block[i-1] output + block[i-2] output)
    - use_sharp_prior: CoC 기반 Sharpness Prior (초점 영역 선명화)
    - activation: 'relu' 또는 'silu'
    """
    def __init__(self, input_channels=7, diopter_mode='spatial', energy_head='fc',
                 num_blocks=4, channels=256, use_film=False, long_skip=False,
                 use_sharp_prior=False, activation='relu',
                 sharp_lambda_init=10.0, sharp_gamma_init=30.0):
        super(SimpleResNet, self).__init__()
        self.diopter_mode = diopter_mode
        self.energy_head = energy_head
        self.use_film = use_film
        self.long_skip = long_skip
        self.use_sharp_prior = use_sharp_prior
        self.activation = activation

        act_fn = nn.SiLU() if activation == 'silu' else nn.ReLU()
        self.act = act_fn

        if diopter_mode in ['spatial', 'coc', 'coc_signed', 'coc_abs']:
            in_ch = input_channels + 1
        else:
            in_ch = input_channels

        # 초기 특징 추출 (해상도 유지)
        self.conv_in = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1)
        self.conv_expand = nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1)

        # Residual Blocks (long_skip 시 개별 블록 접근 필요 → 항상 ModuleList)
        if use_film or long_skip:
            self.res_blocks = nn.ModuleList(
                [FiLMResidualBlock(channels, activation=activation) if use_film
                 else ResidualBlock(channels, activation=activation)
                 for _ in range(num_blocks)]
            )
        else:
            self.res_blocks = nn.Sequential(
                *[ResidualBlock(channels, activation=activation) for _ in range(num_blocks)]
            )

        # Energy output head
        if energy_head == 'conv1x1':
            self.conv_energy = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0)
        else:  # 'fc'
            self.fc = nn.Linear(channels * 512 * 512, 1)

        # Sharp Prior: 초점 영역에서 원본 RGB로 당기는 해석적 이차항
        if use_sharp_prior:
            self.sharp_lambda = nn.Parameter(torch.tensor(sharp_lambda_init))
            self.sharp_gamma = nn.Parameter(torch.tensor(sharp_gamma_init))

    def _get_cond_map(self, x, diopter, N, H, W):
        """FiLM conditioning용 condition map 추출"""
        if self.diopter_mode in ['coc', 'coc_signed', 'coc_abs']:
            return x[:, 7:8, :, :]
        else:  # spatial
            return diopter.view(N, 1, 1, 1).expand(N, 1, H, W)

    def forward(self, x, diopter):
        N, C, H, W = x.shape

        # ── 원본 RGB & 생성 중 이미지 참조 (sharp prior용) ──
        rgb_in = x[:, :3, :, :]   # 원본 RGB (3ch)
        y_pred = x[:, 4:7, :, :]  # 생성 중인 predicted image (3ch)

        # Sharp Prior용 CoC 맵 미리 저장 (spatial concat 전에)
        if self.use_sharp_prior and self.diopter_mode in ['coc', 'coc_signed', 'coc_abs']:
            coc_for_prior = x[:, 7:8, :, :]  # (N, 1, H, W)

        if self.diopter_mode == 'spatial':
            diopter_map = diopter.view(N, 1, 1, 1).expand(N, 1, H, W)
            x = torch.cat([x, diopter_map], dim=1)
        elif self.diopter_mode in ['coc', 'coc_signed', 'coc_abs']:
            pass  # CoC는 이미 x에 포함

        # FiLM: condition map 추출
        cond_map = self._get_cond_map(x, diopter, N, H, W) if self.use_film else None

        x = self.act(self.conv_in(x))
        x = self.act(self.conv_expand(x))

        if self.long_skip:
            outputs = []
            for i, block in enumerate(self.res_blocks):
                if i >= 2:
                    x = x + outputs[i - 2]
                x = block(x, cond_map) if self.use_film else block(x)
                outputs.append(x)
        elif self.use_film:
            for block in self.res_blocks:
                x = block(x, cond_map)
        else:
            x = self.res_blocks(x)

        if self.energy_head == 'conv1x1':
            x = self.conv_energy(x)
            eng = torch.sum(x, dim=(2, 3))   # (N, 1)
        else:  # 'fc'
            x = torch.flatten(x, 1)
            eng = self.fc(x)                  # (N, 1)

        # ── Sharp Prior: E_total = E_nn - λ/2 · Σ w(coc)·(y - rgb)² ──
        if self.use_sharp_prior and self.diopter_mode in ['coc', 'coc_signed', 'coc_abs']:
            coc_abs = torch.abs(coc_for_prior)  # (N, 1, H, W)
            w = torch.exp(-torch.abs(self.sharp_gamma) * coc_abs)
            sharp_penalty = 0.5 * torch.abs(self.sharp_lambda) * torch.sum(
                w * (y_pred - rgb_in) ** 2, dim=(1, 2, 3)
            )
            eng = eng - sharp_penalty.unsqueeze(-1)

        return eng


class SimpleResNetFiLM(nn.Module):
    """
    SimpleResNet + FiLM-only CoC conditioning.

    CoC 맵을 입력 채널에 합치지 않고, FiLM conditioning에만 사용.
    - CoC 모드: 입력 7ch (RGBD 4ch + predicted 3ch), CoC는 FiLM만
    - Spatial 모드: diopter를 FiLM condition으로 사용 (입력 7ch 유지)
    - 나머지 구조는 SimpleResNet과 동일 (long_skip 지원)
    - use_sharp_prior: CoC 기반 Sharpness Prior (초점 영역 선명화)
    - activation: 'relu' 또는 'silu'
    """
    def __init__(self, input_channels=7, diopter_mode='coc', energy_head='fc',
                 num_blocks=4, channels=256, long_skip=False,
                 use_sharp_prior=False, activation='relu',
                 sharp_lambda_init=10.0, sharp_gamma_init=30.0):
        super(SimpleResNetFiLM, self).__init__()
        self.diopter_mode = diopter_mode
        self.energy_head = energy_head
        self.long_skip = long_skip
        self.use_sharp_prior = use_sharp_prior
        self.activation = activation

        act_fn = nn.SiLU() if activation == 'silu' else nn.ReLU()
        self.act = act_fn

        # 항상 7ch 입력 (CoC는 FiLM에만 사용, concat X)
        in_ch = input_channels

        # 초기 특징 추출 (해상도 유지)
        self.conv_in = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1)
        self.conv_expand = nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1)

        # FiLM Residual Blocks (항상 FiLM 사용)
        self.res_blocks = nn.ModuleList(
            [FiLMResidualBlock(channels, activation=activation) for _ in range(num_blocks)]
        )

        # Energy output head
        if energy_head == 'conv1x1':
            self.conv_energy = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0)
        else:  # 'fc'
            self.fc = nn.Linear(channels * 512 * 512, 1)

        # Sharp Prior: 초점 영역에서 원본 RGB로 당기는 해석적 이차항
        if use_sharp_prior:
            self.sharp_lambda = nn.Parameter(torch.tensor(sharp_lambda_init))
            self.sharp_gamma = nn.Parameter(torch.tensor(sharp_gamma_init))

    def forward(self, x, diopter):
        N, C, H, W = x.shape

        # ── 원본 RGB & 생성 중 이미지 참조 (sharp prior용) ──
        rgb_in = x[:, :3, :, :]   # 원본 RGB (3ch)
        y_pred = x[:, 4:7, :, :]  # 생성 중인 predicted image (3ch)

        # ── Condition map 추출 & 입력 분리 ──
        if self.diopter_mode in ['coc', 'coc_signed', 'coc_abs']:
            # CoC는 8번째 채널 (index 7) — FiLM에만 사용
            cond_map = x[:, 7:8, :, :]
            # 모델 입력: CoC 제외한 7ch (RGBD 4ch + predicted 3ch)
            x = x[:, :7, :, :]
        else:  # spatial
            cond_map = diopter.view(N, 1, 1, 1).expand(N, 1, H, W)
            # spatial 모드: 입력 그대로 7ch

        x = self.act(self.conv_in(x))
        x = self.act(self.conv_expand(x))

        if self.long_skip:
            # DeepFocus 스타일: 1-layer interval element-wise add
            outputs = []
            for i, block in enumerate(self.res_blocks):
                if i >= 2:
                    x = x + outputs[i - 2]
                x = block(x, cond_map)
                outputs.append(x)
        else:
            for block in self.res_blocks:
                x = block(x, cond_map)

        if self.energy_head == 'conv1x1':
            x = self.conv_energy(x)
            eng = torch.sum(x, dim=(2, 3))   # (N, 1)
        else:  # 'fc'
            x = torch.flatten(x, 1)
            eng = self.fc(x)                  # (N, 1)

        # ── Sharp Prior: E_total = E_nn - λ/2 · Σ w(coc)·(y - rgb)² ──
        if self.use_sharp_prior and self.diopter_mode in ['coc', 'coc_signed', 'coc_abs']:
            coc_abs = torch.abs(cond_map)  # (N, 1, H, W)
            # w(coc): CoC≈0 → 1 (초점 영역), CoC↑ → 0 (블러 영역)
            w = torch.exp(-torch.abs(self.sharp_gamma) * coc_abs)
            # 이차 페널티: 초점 영역에서 y → rgb 방향으로 당기는 에너지
            sharp_penalty = 0.5 * torch.abs(self.sharp_lambda) * torch.sum(
                w * (y_pred - rgb_in) ** 2, dim=(1, 2, 3)
            )  # (N,)
            eng = eng - sharp_penalty.unsqueeze(-1)  # (N, 1)

        return eng


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
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)   # (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)   # (N, C, H, W)
        return residual + x


class FiLMConvNeXtBlock(nn.Module):
    """
    ConvNeXtBlock + SpatialFiLM conditioning.
    FiLM은 ConvNeXt 연산 후, residual add 직전에 적용.
    """
    def __init__(self, channels, expansion=4):
        super(FiLMConvNeXtBlock, self).__init__()
        hidden_dim = expansion * channels
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.pwconv1 = nn.Linear(channels, hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, channels)
        self.film = SpatialFiLM(condition_channels=1, feature_channels=channels)

    def forward(self, x, condition_map):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        x = self.film(x, condition_map)
        return residual + x


class SimpleConvNeXt(nn.Module):
    """
    SimpleResNet과 동일한 구조에서 ResidualBlock → ConvNeXtBlock 교체.
    - Stem: 8ch → 64ch → channels (conv_in + conv_expand)
    - Body: ConvNeXtBlock × num_blocks (stride=1, 해상도 유지)
    - Head: fc (512×512 전용) 또는 conv1x1 (해상도 무관)
    - use_film=True 이면 FiLMConvNeXtBlock 사용
    """
    def __init__(self, input_channels=7, diopter_mode='spatial', energy_head='fc', num_blocks=9, channels=256, use_film=False):
        super(SimpleConvNeXt, self).__init__()
        self.diopter_mode = diopter_mode
        self.energy_head = energy_head
        self.use_film = use_film

        if diopter_mode in ['spatial', 'coc', 'coc_signed', 'coc_abs']:
            in_ch = input_channels + 1
        else:
            in_ch = input_channels

        self.conv_in = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1)
        self.conv_expand = nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1)

        if use_film:
            self.blocks = nn.ModuleList(
                [FiLMConvNeXtBlock(channels=channels, expansion=4) for _ in range(num_blocks)]
            )
        else:
            self.blocks = nn.Sequential(
                *[ConvNeXtBlock(channels=channels, expansion=4) for _ in range(num_blocks)]
            )

        if energy_head == 'conv1x1':
            self.conv_energy = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0)
        else:  # 'fc'
            self.fc = nn.Linear(channels * 512 * 512, 1)

    def _get_cond_map(self, x, diopter, N, H, W):
        if self.diopter_mode in ['coc', 'coc_signed', 'coc_abs']:
            return x[:, 7:8, :, :]
        else:
            return diopter.view(N, 1, 1, 1).expand(N, 1, H, W)

    def forward(self, x, diopter):
        N, C, H, W = x.shape

        if self.diopter_mode == 'spatial':
            diopter_map = diopter.view(N, 1, 1, 1).expand(N, 1, H, W)
            x = torch.cat([x, diopter_map], dim=1)
        elif self.diopter_mode in ['coc', 'coc_signed', 'coc_abs']:
            pass

        cond_map = self._get_cond_map(x, diopter, N, H, W) if self.use_film else None

        x = F.relu(self.conv_in(x))
        x = F.relu(self.conv_expand(x))

        if self.use_film:
            for block in self.blocks:
                x = block(x, cond_map)
        else:
            x = self.blocks(x)

        if self.energy_head == 'conv1x1':
            x = self.conv_energy(x)
            x = torch.sum(x, dim=(2, 3))
        else:  # 'fc'
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x


class ConvNeXtUNet(nn.Module):
    """
    ConvNeXt + U-Net 구조: Downsample → ConvNeXt Body → Upsample.
    - channels 파라미터로 채널 수 유동 조절 가능
    - use_film=True 이면 FiLMConvNeXtBlock 사용
    """
    def __init__(self, input_channels=7, diopter_mode='spatial', energy_head='fc', num_blocks=9, channels=256, use_film=False):
        super(ConvNeXtUNet, self).__init__()
        self.diopter_mode = diopter_mode
        self.energy_head = energy_head
        self.use_film = use_film

        if diopter_mode in ['spatial', 'coc', 'coc_signed', 'coc_abs']:
            in_ch = input_channels + 1
        else:
            in_ch = input_channels

        # 1. Stem & Downsample (512x512 -> 256x256)
        self.conv_in = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.Conv2d(64, channels, kernel_size=4, stride=2, padding=1)

        # 2. Body: ConvNeXt Blocks
        if use_film:
            self.blocks = nn.ModuleList(
                [FiLMConvNeXtBlock(channels=channels, expansion=4) for _ in range(num_blocks)]
            )
        else:
            self.blocks = nn.Sequential(
                *[ConvNeXtBlock(channels=channels, expansion=4) for _ in range(num_blocks)]
            )

        # 3. Upsample (256x256 -> 512x512)
        self.upsample = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)
        self.conv_out = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

        # 4. Energy output head
        if energy_head == 'conv1x1':
            self.conv_energy = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0)
        else:  # 'fc'
            self.fc = nn.Linear(channels * 512 * 512, 1)

    def _get_cond_map(self, x, diopter, N, H, W):
        if self.diopter_mode in ['coc', 'coc_signed', 'coc_abs']:
            return x[:, 7:8, :, :]
        else:
            return diopter.view(N, 1, 1, 1).expand(N, 1, H, W)

    def forward(self, x, diopter):
        N, C, H, W = x.shape

        if self.diopter_mode == 'spatial':
            diopter_map = diopter.view(N, 1, 1, 1).expand(N, 1, H, W)
            x = torch.cat([x, diopter_map], dim=1)
        elif self.diopter_mode in ['coc', 'coc_signed', 'coc_abs']:
            pass

        # FiLM: condition map (원본 해상도)
        cond_map = self._get_cond_map(x, diopter, N, H, W) if self.use_film else None

        # Downsample
        x_stem = F.relu(self.conv_in(x))
        x_down = F.relu(self.downsample(x_stem))

        # Body
        x_body = x_down
        if self.use_film:
            # FiLM: cond_map도 동일 크기로 다운샘플
            cond_map_down = F.interpolate(cond_map, size=x_body.shape[2:], mode='bilinear', align_corners=False)
            for block in self.blocks:
                x_body = block(x_body, cond_map_down)
        else:
            x_body = self.blocks(x_body)

        # Upsample
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

        if diopter_mode in ['spatial', 'coc', 'coc_signed', 'coc_abs']:
            in_ch = input_channels + 1
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
        elif self.diopter_mode in ['coc', 'coc_signed', 'coc_abs']:
            pass

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


class SpatialFiLM(nn.Module):
    def __init__(self, condition_channels=1, feature_channels=256):
        super(SpatialFiLM, self).__init__()
        self.conv_gamma = nn.Conv2d(condition_channels, feature_channels, kernel_size=3, padding=1)
        self.conv_beta = nn.Conv2d(condition_channels, feature_channels, kernel_size=3, padding=1)

    def forward(self, x, condition_map):
        gamma = self.conv_gamma(condition_map)
        beta = self.conv_beta(condition_map)
        return x * (1 + gamma) + beta


class FiLMResidualBlock(nn.Module):
    def __init__(self, channels, activation='relu'):
        super(FiLMResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.film = SpatialFiLM(condition_channels=1, feature_channels=channels)
        self.act = nn.SiLU() if activation == 'silu' else nn.ReLU()

    def forward(self, x, condition_map):
        residual = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        out = self.film(out, condition_map)
        out += residual
        return self.act(out)


class InterleaveResNet(nn.Module):
    """
    SimpleResNet + Pixel Shuffle (Interleave/Deinterleave) 구조.

    DeepFocus의 interleave 기법 적용:
      8ch 입력 → pixel_unshuffle(r) → conv_in(channels) → ResBlocks(channels)
      → conv_pre_shuffle(3*r²) → pixel_shuffle(r) → 3ch → energy head

    interleave_rate=2 기준:
      (8, 512, 512) → unshuffle → (32, 256, 256) → stem → (256, 256, 256)
      → ResBlocks → (256, 256, 256) → conv → (12, 256, 256) → shuffle → (3, 512, 512)
      → conv1x1(3→1) + sum  또는  fc(3*512*512, 1)

    Parameters:
        input_channels: RGBD(4) + pred(3) = 7 (기본)
        diopter_mode: 'spatial' | 'coc' | 'coc_abs' | 'coc_signed'
        energy_head: 'fc' | 'conv1x1'
        num_blocks: Residual Block 수
        channels: 은닉층 채널 수 (기본 256)
        use_film: FiLM conditioning 활성화
        interleave_rate: pixel shuffle 배율 (2 = 해상도 1/2, 채널 ×4)
    """
    def __init__(self, input_channels=7, diopter_mode='spatial', energy_head='conv1x1',
                 num_blocks=4, channels=256, use_film=False, interleave_rate=2):
        super(InterleaveResNet, self).__init__()
        self.diopter_mode = diopter_mode
        self.energy_head = energy_head
        self.use_film = use_film
        self.interleave_rate = interleave_rate
        r2 = interleave_rate ** 2
        out_ch = 3  # pixel_shuffle 후 최종 채널

        if diopter_mode in ['spatial', 'coc', 'coc_signed', 'coc_abs']:
            in_ch = input_channels + 1
        else:
            in_ch = input_channels

        interleaved_ch = in_ch * r2  # e.g. 8 * 4 = 32

        # Stem: interleaved → channels (e.g. 32 → 256)
        self.conv_in = nn.Conv2d(interleaved_ch, channels, kernel_size=3, stride=1, padding=1)

        # Residual Blocks (낮은 해상도에서 동작)
        if use_film:
            self.res_blocks = nn.ModuleList(
                [FiLMResidualBlock(channels) for _ in range(num_blocks)]
            )
        else:
            self.res_blocks = nn.Sequential(
                *[ResidualBlock(channels) for _ in range(num_blocks)]
            )

        # Pre-shuffle: channels → out_ch * r² (e.g. 256 → 12)
        self.conv_pre_shuffle = nn.Conv2d(channels, out_ch * r2, kernel_size=3, stride=1, padding=1)

        # Energy head (pixel_shuffle 후 out_ch 채널 상태에서 동작)
        if energy_head == 'conv1x1':
            self.conv_energy = nn.Conv2d(out_ch, 1, kernel_size=1, stride=1, padding=0)
        else:  # 'fc'
            self.fc = nn.Linear(out_ch * 512 * 512, 1)

    def _get_cond_map(self, x, diopter, N, H, W):
        """FiLM conditioning용 condition map 추출 (원본 해상도)"""
        if self.diopter_mode in ['coc', 'coc_signed', 'coc_abs']:
            return x[:, 7:8, :, :]
        else:  # spatial
            return diopter.view(N, 1, 1, 1).expand(N, 1, H, W)

    def forward(self, x, diopter):
        N, C, H, W = x.shape
        r = self.interleave_rate

        # ── diopter 채널 결합 (원본 해상도) → 8ch ──
        if self.diopter_mode == 'spatial':
            diopter_map = diopter.view(N, 1, 1, 1).expand(N, 1, H, W)
            x = torch.cat([x, diopter_map], dim=1)
        elif self.diopter_mode in ['coc', 'coc_signed', 'coc_abs']:
            pass  # CoC는 이미 x에 포함

        # FiLM condition map (원본 해상도)
        cond_map = self._get_cond_map(x, diopter, N, H, W) if self.use_film else None

        # ── Interleave: (N, 8, H, W) → (N, 32, H/r, W/r) ──
        x = F.pixel_unshuffle(x, r)

        # ── Stem: 32 → channels ──
        x = F.relu(self.conv_in(x))

        # ── Residual Blocks (축소 해상도) ──
        if self.use_film:
            cond_map_small = F.interpolate(
                cond_map, size=(H // r, W // r),
                mode='bilinear', align_corners=False
            )
            for block in self.res_blocks:
                x = block(x, cond_map_small)
        else:
            x = self.res_blocks(x)

        # ── Pre-shuffle: channels → 3*r² (e.g. 256→12) ──
        x = self.conv_pre_shuffle(x)

        # ── Deinterleave: (N, 12, H/r, W/r) → (N, 3, H, W) ──
        x = F.pixel_shuffle(x, r)

        # ── Energy Head (3ch, 원본 해상도) ──
        if self.energy_head == 'conv1x1':
            x = self.conv_energy(x)       # (N, 3, H, W) → (N, 1, H, W)
            x = torch.sum(x, dim=(2, 3))  # (N, 1)
        else:  # 'fc'
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x


class ResUNet(nn.Module):
    """
    ResUNet: Encoder-Decoder with long skip connections.
    기존 ResidualBlock / FiLMResidualBlock 을 재활용합니다.

    use_film=True 이면 인코더·바틀넥·디코더 전체에 FiLM 적용.
    각 레벨의 condition map 은 해당 해상도로 bilinear resize.
    """
    def __init__(self, input_channels=7, diopter_mode='spatial',
                 energy_head='conv1x1', base_channels=64,
                 num_bottleneck_blocks=3, use_film=False):
        super(ResUNet, self).__init__()
        self.diopter_mode = diopter_mode
        self.energy_head = energy_head
        self.use_film = use_film

        bc = base_channels  # 64

        # ── 입력 채널 결정 ──
        if diopter_mode in ['spatial', 'coc', 'coc_signed', 'coc_abs']:
            in_ch = input_channels + 1
        else:
            in_ch = input_channels

        # ── 1. Stem: in_ch → bc (해상도 유지) ──
        self.stem = nn.Conv2d(in_ch, bc, kernel_size=3, stride=1, padding=1)

        # ── 2. Encoder ──
        # Level 1: bc → bc (H×W)
        self.enc1_block = FiLMResidualBlock(bc) if use_film else ResidualBlock(bc)
        self.down1 = nn.Conv2d(bc, bc * 2, kernel_size=3, stride=2, padding=1)  # → 128ch, H/2

        # Level 2: bc*2 → bc*2 (H/2×W/2)
        self.enc2_block = FiLMResidualBlock(bc * 2) if use_film else ResidualBlock(bc * 2)
        self.down2 = nn.Conv2d(bc * 2, bc * 4, kernel_size=3, stride=2, padding=1)  # → 256ch, H/4

        # ── 3. Bottleneck: bc*4 채널, H/4 해상도 ──
        if use_film:
            self.bottleneck = nn.ModuleList(
                [FiLMResidualBlock(bc * 4) for _ in range(num_bottleneck_blocks)]
            )
        else:
            self.bottleneck = nn.Sequential(
                *[ResidualBlock(bc * 4) for _ in range(num_bottleneck_blocks)]
            )

        # ── 4. Decoder ──
        # Up-Level 2: bc*4 → bc*2 (H/2)
        self.up2 = nn.ConvTranspose2d(bc * 4, bc * 2, kernel_size=4, stride=2, padding=1)
        self.dec2_fuse = nn.Conv2d(bc * 4, bc * 2, kernel_size=3, stride=1, padding=1)  # cat후 채널 압축
        self.dec2_block = FiLMResidualBlock(bc * 2) if use_film else ResidualBlock(bc * 2)

        # Up-Level 1: bc*2 → bc (H)
        self.up1 = nn.ConvTranspose2d(bc * 2, bc, kernel_size=4, stride=2, padding=1)
        self.dec1_fuse = nn.Conv2d(bc * 2, bc, kernel_size=3, stride=1, padding=1)  # cat후 채널 압축
        self.dec1_block = FiLMResidualBlock(bc) if use_film else ResidualBlock(bc)

        # ── 5. Energy Head ──
        if energy_head == 'conv1x1':
            self.conv_energy = nn.Conv2d(bc, 1, kernel_size=1, stride=1, padding=0)
        else:  # 'fc'
            self.fc = nn.Linear(bc * 512 * 512, 1)

    def _get_cond_map(self, x, diopter, N, H, W):
        """FiLM conditioning 용 condition map 추출"""
        if self.diopter_mode in ['coc', 'coc_signed', 'coc_abs']:
            return x[:, 7:8, :, :]  # CoC 채널
        else:  # spatial
            return diopter.view(N, 1, 1, 1).expand(N, 1, H, W)

    def forward(self, x, diopter):
        N, C, H, W = x.shape

        # ── diopter 채널 결합 ──
        if self.diopter_mode == 'spatial':
            diopter_map = diopter.view(N, 1, 1, 1).expand(N, 1, H, W)
            x = torch.cat([x, diopter_map], dim=1)
        elif self.diopter_mode in ['coc', 'coc_abs', 'coc_signed']:
            pass  # CoC는 이미 x에 포함

        # FiLM: condition map (원본 해상도)
        cond_map = self._get_cond_map(x, diopter, N, H, W) if self.use_film else None

        # ── Stem ──
        x_stem = F.relu(self.stem(x))  # (N, 64, H, W)

        # ── Encoder Level 1 ──
        if self.use_film:
            skip1 = self.enc1_block(x_stem, cond_map)  # (N, 64, H, W)
        else:
            skip1 = self.enc1_block(x_stem)
        x_d1 = F.relu(self.down1(skip1))               # (N, 128, H/2, W/2)

        # ── Encoder Level 2 ──
        if self.use_film:
            cond_half = F.interpolate(cond_map, size=x_d1.shape[2:],
                                      mode='bilinear', align_corners=False)
            skip2 = self.enc2_block(x_d1, cond_half)   # (N, 128, H/2, W/2)
        else:
            skip2 = self.enc2_block(x_d1)
        x_d2 = F.relu(self.down2(skip2))               # (N, 256, H/4, W/4)

        # ── Bottleneck ──
        x_bn = x_d2
        if self.use_film:
            cond_quarter = F.interpolate(cond_map, size=x_bn.shape[2:],
                                         mode='bilinear', align_corners=False)
            for block in self.bottleneck:
                x_bn = block(x_bn, cond_quarter)
        else:
            x_bn = self.bottleneck(x_bn)               # (N, 256, H/4, W/4)

        # ── Decoder Up-Level 2 ──
        x_up2 = F.relu(self.up2(x_bn))                 # (N, 128, H/2, W/2)
        x_up2 = torch.cat([x_up2, skip2], dim=1)       # (N, 256, H/2, W/2)
        x_up2 = F.relu(self.dec2_fuse(x_up2))          # (N, 128, H/2, W/2)
        if self.use_film:
            cond_half = F.interpolate(cond_map, size=x_up2.shape[2:],
                                      mode='bilinear', align_corners=False)
            x_up2 = self.dec2_block(x_up2, cond_half)  # (N, 128, H/2, W/2)
        else:
            x_up2 = self.dec2_block(x_up2)

        # ── Decoder Up-Level 1 ──
        x_up1 = F.relu(self.up1(x_up2))                # (N, 64, H, W)
        x_up1 = torch.cat([x_up1, skip1], dim=1)       # (N, 128, H, W)
        x_up1 = F.relu(self.dec1_fuse(x_up1))          # (N, 64, H, W)
        if self.use_film:
            x_up1 = self.dec1_block(x_up1, cond_map)   # (N, 64, H, W) — 원본 해상도
        else:
            x_up1 = self.dec1_block(x_up1)

        # ── Energy Head ──
        if self.energy_head == 'conv1x1':
            eng = self.conv_energy(x_up1)          # (N, 1, H, W)
            eng = torch.sum(eng, dim=(2, 3))       # (N, 1)
        else:  # 'fc'
            eng = torch.flatten(x_up1, 1)
            eng = self.fc(eng)
        return eng


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
