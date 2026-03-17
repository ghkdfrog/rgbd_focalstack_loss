import sys
sys.path.insert(0, '.')
from gm.model import SimpleCNN, SimpleCNNStride, SimpleResNet, SimpleConvNeXt

models = [
    ("SimpleCNN (0310 runs, fc)",        SimpleCNN(7, 'coc', 'fc')),
    ("SimpleCNNStride (0313 run, fc)",   SimpleCNNStride(7, 'coc', 'fc')),
    ("SimpleResNet (0314 run, fc, 4blk)",SimpleResNet(7, 'coc', 'fc', 4)),
    ("SimpleConvNeXt (NEW, fc, 4blk)",   SimpleConvNeXt(7, 'coc', 'fc', 4)),
]

print(f"{'Model':<42} {'Total Params':>15}  {'Backbone Only':>15}")
print("-" * 75)
for name, m in models:
    total = sum(p.numel() for p in m.parameters())
    # fc head = 256*512*512 + 1 = 67,108,865
    fc_params = 256 * 512 * 512 + 1
    backbone = total - fc_params
    print(f"{name:<42} {total:>15,}  {backbone:>15,}")
