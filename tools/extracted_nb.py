# ----- MARKDOWN CELL 0 -----
# # Gradient Matching with Focal Dataset (Scene 0)
# 이 노트북은 기존의 Gradient Matching 방식을 7채널 실제 Focal Dataset(0번 Scene)을 사용하도록 수정한 버전입니다.

# ----- CODE CELL 1 -----
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 현재 폴더를 시스템 경로에 추가하여 dataset_focal 등을 바로 import할 수 있게 설정
sys.path.append(os.getcwd())

from dataset_focal import FocalDataset, calculate_psnr


use_coc = True
# 1. 데이터셋 로드 (0번 씬에 대해서만 학습)
train_ds = FocalDataset(
    data_dir='../varifocal/data',
    generated_data_dir='./data',
    split='train',
    unmatch_ratio=0, # unmatch 없이 매칭되는 이미지 40장만 사용
    use_coc=use_coc,
    return_gt=True,
    single_scene_only=True
)

print(f"Dataset length: {len(train_ds)}")

# ----- CODE CELL 2 -----
# 2. 첫 번째 씬의 특정 Focal Plane(Ground Truth) 데이터 확인
x, diopter, targets, gt = train_ds[20] # 20번 focal plane 가져오기

gt_image = gt.permute(1,2,0).cpu().numpy()
plt.imshow(gt_image)
plt.title(f'Target GT Focus (Diopter: {diopter.item():.2f})')
plt.axis('off')
plt.show()

# ----- CODE CELL 3 -----
class SimpleCNN(nn.Module):
    def __init__(self, input_channels=7, diopter_mode='spatial'):
        super(SimpleCNN, self).__init__()
        self.diopter_mode = diopter_mode
        
        # diopter_mode에 따라 입력 채널 수 결정
        if diopter_mode == 'spatial':
            in_ch = input_channels + 1 # spatial: RGBD(4) + Optim(3) + diopter(1) = 8채널
        elif diopter_mode == 'coc':
            in_ch = input_channels + 1 # coc: RGBD(4) + Optim(3) + CoC(1) = 8채널
        else:
            in_ch = input_channels
        
        # Conv Layers: 공간 해상도를 유지하기 위해 stride=1 사용
        self.conv1 = nn.Conv2d(in_ch, 16, kernel_size=3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        
        # 입력 해상도가 512x512인 점을 감안하여 Fully Connected Layer 구성
        self.fc = nn.Linear(256 * 512 * 512, 1)

    def forward(self, x, diopter):
        N, C, H, W = x.shape
        
        # Diopter 정보를 condition으로 채널 단위로 붙임
        if self.diopter_mode == 'spatial':
            diopter_map = diopter.view(N, 1, 1, 1).expand(N, 1, H, W)
            x = torch.cat([x, diopter_map], dim=1)
        elif self.diopter_mode == 'coc':
            pass # CoC 모드일 때는 학습 루프에서 x의 마지막 채널에 CoC가 붙어 들어옴

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ----- [테스트 코드] -----
# 모델 초기화 (CoC 모드)
diopter_mode = 'spatial'
if use_coc:
    diopter_mode = 'coc'
model = SimpleCNN(diopter_mode=diopter_mode).cuda()

# 입력 텐서 테스트: RGBD(4) + 현초점(3) + CoC(1) = 총 8채널
input_data = torch.randn(1, 8, 512, 512).cuda()
diopter_data = torch.tensor([1.0]).cuda() # CoC 모드에서는 사용되지 않지만 포맷을 위해 전달

output = model(input_data, diopter_data)

print(f"입력 크기 (batch, channels, H, W): {input_data.shape}")
print(f"출력 값 에너지: {output.item():.4f}")


# ----- CODE CELL 4 -----
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

optimizer = optim.Adam(model.parameters(), lr=0.0001)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

step_size = 0.2
num_steps = 50
epochs = 50 # 빠른 학습 경과 확인을 위해 에폭 설정

print("학습 시작...")

# --- Gradient Trajectory Score Matching 학습 루프 --- 
for epoch in range(epochs):
    epoch_loss = 0.0
    n_samples = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
    for batch_data in pbar:
        x, diopter, targets, gt = batch_data
        
        x = x.cuda()
        diopter = diopter.cuda()
        gt = gt.cuda()
        
        optimizer.zero_grad()
        
        # 1. 초기 노이즈 이미지
        current_image = torch.randn_like(gt).cuda()
        batch_loss = 0.0
        
        # 2. 50 스텝 Trajectory 생성 및 각 스텝마다의 Gradient 예측/학습
        for step in range(num_steps):
            current_image.requires_grad_(True)
            
            input_rgbd = x[:, :4, :, :]
            if x.shape[1] > 7:
                input_tail = x[:, 7:, :, :]
                model_input = torch.cat([input_rgbd, current_image, input_tail], dim=1)
            else:
                model_input = torch.cat([input_rgbd, current_image], dim=1)
            
            # 모델 순전파 (에너지 스칼라)
            output_data = model(model_input, diopter)
            
            # 이미지 자체에 대한 기울기 예측
            pred_grad = torch.autograd.grad(
                outputs=output_data, 
                inputs=current_image,
                grad_outputs=torch.ones_like(output_data),
                create_graph=True 
            )[0]
            
            # 실제 목표 스코어 방향 (Ground Truth로 향하도록)
            gt_grad = gt - current_image
            
            # 예측한 Gradient가 GT방향 Gradient와 같아지도록 Loss 부과
            loss = F.mse_loss(pred_grad, gt_grad)
            loss.backward() 
            batch_loss += loss.item()
            
            # 궤적을 따라 이미지를 업데이트
            with torch.no_grad():
                current_image = (current_image + step_size * pred_grad).detach()

        # 50번의 궤적의 gradient가 누적되었기 때문에 optimizer step
        optimizer.step()
        
        avg_step_loss = batch_loss / num_steps
        epoch_loss += avg_step_loss
        n_samples += 1
        
        pbar.set_postfix({'loss': f"{avg_step_loss:.4f}"})
        
    print(f"Epoch {epoch + 1:3d} | Average Loss per step: {epoch_loss / n_samples:.6f}")

print("학습 완료!")

# ----- CODE CELL 5 -----
# --- 추론(Inference) 테스트 및 과정 시각화 ---
x, diopter, targets, gt = train_ds[20] # 동일한 20번 평면에 대한 추론
x = x.unsqueeze(0).cuda()
diopter = diopter.unsqueeze(0).cuda()
gt = gt.unsqueeze(0).cuda()

# 생성할 이미지 텐서 초기화
generated_image = torch.randn_like(gt).cuda()

step_size = 0.2
num_steps = 100

history = [generated_image.detach().squeeze().cpu().numpy().transpose(1, 2, 0)]

for i in range(1, num_steps + 1):
    # 매 스텝마다 새롭게 그래디언트 추적이 가능하도록 설정
    generated_image.requires_grad_(True)
    
    input_rgbd = x[:, :4, :, :]
    
    # ⚠️ 핵심 수정 부분: 데이터셋에서 넘어온 채널이 7보다 크면(CoC가 있으면) 뒤에 CoC를 연결해 8채널 보장
    if x.shape[1] > 7:
        input_tail = x[:, 7:, :, :]
        model_input = torch.cat([input_rgbd, generated_image, input_tail], dim=1)
    else:
        model_input = torch.cat([input_rgbd, generated_image], dim=1)
    
    # 에너지 스코어 예측
    score = model(model_input, diopter)
    
    # 예측된 스코어에 대한 generated_image의 그래디언트 계산
    # (역전파 그래프 꼬임 방지를 위해 grad_outputs 추가 및 create_graph=False)
    grad = torch.autograd.grad(
        outputs=score, 
        inputs=generated_image,
        grad_outputs=torch.ones_like(score),
        create_graph=False
    )[0]
    
    with torch.no_grad():
        # 그래디언트 방향을 따라 이미지 생성 (Langevin Dynamics)
        generated_image = (generated_image + step_size * grad).detach()
    
    if i % 10 == 0:
        im = generated_image.detach().squeeze().cpu().numpy().transpose(1, 2, 0)
        history.append(im)

final_image = torch.clamp(generated_image, min=0.0, max=1.0)
psnr = calculate_psnr(final_image.squeeze().cpu(), gt.squeeze().cpu())

import numpy as np

fig, axes = plt.subplots(1, 6, figsize=(20, 4))
step_labels = [0, num_steps//5, num_steps//5*2, num_steps//5*3, num_steps//5*4, num_steps]

for idx, ax in enumerate(axes):
    ax.imshow(np.clip(history[idx], 0, 1))
    ax.set_title(f"Step {step_labels[idx]}")
    ax.axis('off')

plt.tight_layout()
plt.show()

print(f"Final PSNR: {psnr:.2f} dB")


# ----- CODE CELL 6 -----
# 결과 비교
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(np.clip(gt.squeeze().cpu().numpy().transpose(1,2,0), 0, 1))
axes[0].set_title("Ground Truth Focus")
axes[0].axis('off')

axes[1].imshow(np.clip(final_image.detach().squeeze().cpu().numpy().transpose(1,2,0), 0, 1))
axes[1].set_title(f"Generated Focus (PSNR: {psnr:.2f} dB)")
axes[1].axis('off')

plt.tight_layout()
plt.show()

# ----- CODE CELL 7 -----


