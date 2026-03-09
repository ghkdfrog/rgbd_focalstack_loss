"""
Data Augmentation Pipeline for Loss Estimation Model

Supports two intensity levels:
  - Strong: significant degradation (PSNR 15~35 dB)
  - Weak:   moderate degradation   (PSNR 30~40 dB)

NOTE: Uses only numpy, scipy, and Pillow (no OpenCV dependency)
"""
import numpy as np
from enum import Enum
from scipy.ndimage import gaussian_filter
from PIL import Image
import io


class AugType(Enum):
    GAUSSIAN_NOISE = "gaussian_noise"
    POISSON_NOISE = "poisson_noise"
    GAUSSIAN_BLUR = "gaussian_blur"
    JPEG_COMPRESSION = "jpeg_compression"
    CHROMATIC_ABERRATION = "chromatic_aberration"
    COMBINED = "combined"


# ============================================================
# Strong Augmentations (PSNR 15~35 dB)
# ============================================================

def add_gaussian_noise(image, sigma=None, rng=None):
    if rng is None: rng = np.random.default_rng()
    if sigma is None: sigma = rng.uniform(0.01, 0.08)
    noise = rng.normal(0, sigma, image.shape).astype(np.float32)
    return np.clip(image + noise, 0, 1).astype(np.float32)

def add_poisson_noise(image, lam=None, rng=None):
    if rng is None: rng = np.random.default_rng()
    if lam is None: lam = rng.uniform(20, 80)
    scaled = image * lam
    noisy = rng.poisson(np.maximum(scaled, 0)).astype(np.float32) / lam
    return np.clip(noisy, 0, 1).astype(np.float32)

def apply_gaussian_blur(image, sigma=None, rng=None):
    if rng is None: rng = np.random.default_rng()
    if sigma is None: sigma = rng.uniform(0.5, 3.0)
    blurred = gaussian_filter(image, sigma=(sigma, sigma, 0))
    return np.clip(blurred, 0, 1).astype(np.float32)

def jpeg_compression(image, quality=None, rng=None):
    if rng is None: rng = np.random.default_rng()
    if quality is None: quality = int(rng.integers(30, 80))
    img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode='RGB')
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    decoded = Image.open(buffer).convert('RGB')
    return np.array(decoded).astype(np.float32) / 255.0

def chromatic_aberration(image, shift=None, rng=None):
    if rng is None: rng = np.random.default_rng()
    if shift is None: shift = int(rng.integers(1, 4))
    result = image.copy()
    dx_r = int(rng.integers(-shift, shift + 1))
    dy_r = int(rng.integers(-shift, shift + 1))
    result[:, :, 0] = np.roll(np.roll(image[:, :, 0], dx_r, axis=1), dy_r, axis=0)
    dx_b = int(rng.integers(-shift, shift + 1))
    dy_b = int(rng.integers(-shift, shift + 1))
    result[:, :, 2] = np.roll(np.roll(image[:, :, 2], dx_b, axis=1), dy_b, axis=0)
    return np.clip(result, 0, 1).astype(np.float32)

def apply_combined(image, rng=None):
    if rng is None: rng = np.random.default_rng()
    augmentations = [
        lambda img: add_gaussian_noise(img, rng=rng),
        lambda img: apply_gaussian_blur(img, rng=rng),
        lambda img: jpeg_compression(img, rng=rng),
        lambda img: chromatic_aberration(img, rng=rng),
    ]
    n_augs = rng.integers(2, 4)
    chosen = rng.choice(len(augmentations), size=n_augs, replace=False)
    result = image.copy()
    for idx in chosen:
        result = augmentations[idx](result)
    return result


# ============================================================
# Weak Augmentations (PSNR 30~40 dB)
# Increased intensity to create visible gap from Clean data
# ============================================================

def add_gaussian_noise_weak(image, sigma=None, rng=None):
    """sigma in [0.008, 0.025] — visible but mild noise"""
    if rng is None: rng = np.random.default_rng()
    if sigma is None: sigma = rng.uniform(0.008, 0.025)
    noise = rng.normal(0, sigma, image.shape).astype(np.float32)
    return np.clip(image + noise, 0, 1).astype(np.float32)

def apply_gaussian_blur_weak(image, sigma=None, rng=None):
    """sigma in [0.5, 1.2] — noticeable softness"""
    if rng is None: rng = np.random.default_rng()
    if sigma is None: sigma = rng.uniform(0.5, 1.2)
    blurred = gaussian_filter(image, sigma=(sigma, sigma, 0))
    return np.clip(blurred, 0, 1).astype(np.float32)

def jpeg_compression_weak(image, quality=None, rng=None):
    """quality in [50, 85] — mild block artifacts"""
    if rng is None: rng = np.random.default_rng()
    if quality is None: quality = int(rng.integers(50, 85))
    img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode='RGB')
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    decoded = Image.open(buffer).convert('RGB')
    return np.array(decoded).astype(np.float32) / 255.0

def brightness_shift(image, delta=None, rng=None):
    """delta in [-0.08, 0.08]"""
    if rng is None: rng = np.random.default_rng()
    if delta is None: delta = rng.uniform(-0.08, 0.08)
    return np.clip(image + delta, 0, 1).astype(np.float32)

def contrast_adjust(image, factor=None, rng=None):
    """factor in [0.80, 1.20]"""
    if rng is None: rng = np.random.default_rng()
    if factor is None: factor = rng.uniform(0.80, 1.20)
    mean = image.mean()
    return np.clip((image - mean) * factor + mean, 0, 1).astype(np.float32)

def apply_weak_combined(image, rng=None):
    """Apply 1-2 random weak augmentations"""
    if rng is None: rng = np.random.default_rng()
    augmentations = [
        lambda img: add_gaussian_noise_weak(img, rng=rng),
        lambda img: apply_gaussian_blur_weak(img, rng=rng),
        lambda img: jpeg_compression_weak(img, rng=rng),
        lambda img: brightness_shift(img, rng=rng),
        lambda img: contrast_adjust(img, rng=rng),
    ]
    n_augs = rng.integers(1, 3)
    chosen = rng.choice(len(augmentations), size=n_augs, replace=False)
    result = image.copy()
    for idx in chosen:
        result = augmentations[idx](result)
    return result


# ============================================================
# Dispatchers
# ============================================================

def augment_image(image, aug_type=None, rng=None):
    """Apply a single STRONG augmentation"""
    if rng is None: rng = np.random.default_rng()
    if aug_type is None: aug_type = rng.choice(list(AugType))
    aug_functions = {
        AugType.GAUSSIAN_NOISE: add_gaussian_noise,
        AugType.POISSON_NOISE: add_poisson_noise,
        AugType.GAUSSIAN_BLUR: apply_gaussian_blur,
        AugType.JPEG_COMPRESSION: jpeg_compression,
        AugType.CHROMATIC_ABERRATION: chromatic_aberration,
        AugType.COMBINED: apply_combined,
    }
    degraded = aug_functions[aug_type](image, rng=rng)
    return degraded, aug_type.value

def augment_image_weak(image, rng=None):
    """Apply a single WEAK augmentation"""
    if rng is None: rng = np.random.default_rng()
    weak_fns = [
        ('gaussian_noise_weak', add_gaussian_noise_weak),
        ('blur_weak', apply_gaussian_blur_weak),
        ('jpeg_weak', jpeg_compression_weak),
        ('brightness_shift', brightness_shift),
        ('contrast_adjust', contrast_adjust),
        ('weak_combined', apply_weak_combined),
    ]
    idx = rng.integers(0, len(weak_fns))
    name, fn = weak_fns[idx]
    degraded = fn(image, rng=rng)
    return degraded, name


if __name__ == "__main__":
    print("Testing augmentations...")
    test_img = np.random.rand(64, 64, 3).astype(np.float32)

    print("\n--- Strong Augmentations ---")
    for aug_type in AugType:
        result, name = augment_image(test_img, aug_type=aug_type)
        assert result.shape == test_img.shape
        assert result.min() >= 0 and result.max() <= 1
        print(f"  {name}: OK")

    print("\n--- Weak Augmentations ---")
    for _ in range(6):
        result, name = augment_image_weak(test_img)
        assert result.shape == test_img.shape
        assert result.min() >= 0 and result.max() <= 1
        print(f"  {name}: OK")

    print("\nAll augmentation tests passed!")
