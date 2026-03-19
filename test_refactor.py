"""Test script to verify model refactoring"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from gm.model import SimpleResNet, SimpleConvNeXt, ConvNeXtUNet

device = 'cpu'
H, W = 32, 32  # small resolution for speed

test_cases = [
    ("SimpleResNet(128)",          SimpleResNet, dict(channels=128, use_film=False, energy_head='conv1x1')),
    ("SimpleResNet(128)+FiLM",     SimpleResNet, dict(channels=128, use_film=True,  energy_head='conv1x1')),
    ("SimpleConvNeXt(128)",        SimpleConvNeXt, dict(channels=128, use_film=False, energy_head='conv1x1', num_blocks=2)),
    ("SimpleConvNeXt(128)+FiLM",   SimpleConvNeXt, dict(channels=128, use_film=True,  energy_head='conv1x1', num_blocks=2)),
    ("ConvNeXtUNet(128)",          ConvNeXtUNet, dict(channels=128, use_film=False, energy_head='conv1x1', num_blocks=2)),
    ("ConvNeXtUNet(128)+FiLM",     ConvNeXtUNet, dict(channels=128, use_film=True,  energy_head='conv1x1', num_blocks=2)),
]

for name, cls, kwargs in test_cases:
    try:
        model = cls(diopter_mode='coc', **kwargs).to(device)
        params = sum(p.numel() for p in model.parameters())
        
        # dummy forward
        x = torch.randn(1, 8, H, W).to(device)  # 8ch: RGBD(4) + Optim(3) + CoC(1)
        d = torch.tensor([0.5]).to(device)
        out = model(x, d)
        
        print(f"  OK  {name:30s} params={params:>10,}  out={out.shape}")
    except Exception as e:
        print(f"  FAIL {name:30s} {e}")

print("\nAll tests completed!")
