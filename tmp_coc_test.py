import numpy as np

D = 0.004
film_len_dist = 0.017
fov = 0.7854
W = 512
cocScale = 30.0

film_width = 2.0 * film_len_dist * np.tan(fov / 2.0)

depths = np.linspace(0.1, 10.0, 10)  # 0.1m to 10m
dp_dm = 1.0 / depths

print(f"{'Depth':>8} {'0.1D CoC':>12} {'4.0D CoC':>12}")
print("-" * 35)

for i in range(len(depths)):
    d = depths[i]
    dm = dp_dm[i]
    
    # Far focal plane (0.1D, focus at 10m)
    f01 = 0.1
    fl01 = 1.0 / film_len_dist + f01
    coc01 = D * ((fl01 - dm) / (fl01 - f01) - 1.0)
    coc_norm01 = np.clip(np.abs(coc01) / film_width * W / cocScale, 0.0, 1.0)
    
    # Near focal plane (4.0D, focus at 0.25m)
    f40 = 4.0
    fl40 = 1.0 / film_len_dist + f40
    coc40 = D * ((fl40 - dm) / (fl40 - f40) - 1.0)
    coc_norm40 = np.clip(np.abs(coc40) / film_width * W / cocScale, 0.0, 1.0)
    
    print(f"{d:>8.2f} {coc_norm01:>12.4f} {coc_norm40:>12.4f}")

# Also print the weight maps (using gamma=30.0)
gamma = 30.0
print(f"\nWeight with gamma={gamma}")
print(f"{'Depth':>8} {'0.1D W':>12} {'4.0D W':>12}")
print("-" * 35)

for i in range(len(depths)):
    d = depths[i]
    dm = dp_dm[i]
    
    f01 = 0.1; fl01 = 1.0 / film_len_dist + f01
    coc01 = np.clip(np.abs(D * ((fl01 - dm) / (fl01 - f01) - 1.0)) / film_width * W / cocScale, 0.0, 1.0)
    w01 = np.exp(-gamma * coc01)
    
    f40 = 4.0; fl40 = 1.0 / film_len_dist + f40
    coc40 = np.clip(np.abs(D * ((fl40 - dm) / (fl40 - f40) - 1.0)) / film_width * W / cocScale, 0.0, 1.0)
    w40 = np.exp(-gamma * coc40)
    
    print(f"{d:>8.2f} {w01:>12.4f} {w40:>12.4f}")
