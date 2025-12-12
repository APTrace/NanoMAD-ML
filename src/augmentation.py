#!/usr/bin/env python3
"""
data_augmentation.py

Data augmentation transforms for MAD diffraction data.

These transforms are designed to be applied DURING TRAINING (online augmentation),
not during data generation. This is the standard approach because:
1. Less disk space (don't store augmented copies)
2. More variety (different random augmentations each epoch)
3. More flexible (can adjust augmentation strategy without regenerating data)

Physics-preserving transforms:
- 90° rotations (diffraction has no preferred orientation)
- Horizontal/vertical flips (same reason)
- Intensity scaling (simulates different exposure times)
- Noise injection (simulates worse data quality)

IMPORTANT: When transforming labels, be aware that:
- |F_T| and |F_A| scale as √(intensity)
- sin(Δφ) and cos(Δφ) are unchanged by intensity scaling

Author: Claude (Anthropic) + Thomas
Date: December 2024
Based on: NanoMAD expert recommendations (NanoMAD_CNN_Design.md)
"""

import numpy as np
from typing import Tuple, Dict, Optional


# =============================================================================
# GEOMETRIC TRANSFORMS (Preserve Physics Exactly)
# =============================================================================

def random_rot90(
    intensity: np.ndarray,
    labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply random 90° rotation to both intensity and labels.

    Rotates by 0°, 90°, 180°, or 270° (chosen randomly).

    Parameters
    ----------
    intensity : np.ndarray
        Intensity patch, shape (H, W, n_energies) or (H, W)
    labels : np.ndarray
        Label patch, shape (H, W, 4) where channels are |F_T|, |F_A|, sin(Δφ), cos(Δφ)

    Returns
    -------
    intensity_rot, labels_rot : tuple of np.ndarray
        Rotated arrays with same shapes as inputs
    """
    k = np.random.randint(0, 4)  # Number of 90° rotations

    intensity_rot = np.rot90(intensity, k, axes=(0, 1))
    labels_rot = np.rot90(labels, k, axes=(0, 1))

    return intensity_rot.copy(), labels_rot.copy()


def random_flip(
    intensity: np.ndarray,
    labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply random horizontal and/or vertical flips.

    Each axis is flipped with 50% probability, independently.

    Parameters
    ----------
    intensity : np.ndarray
        Intensity patch, shape (H, W, n_energies) or (H, W)
    labels : np.ndarray
        Label patch, shape (H, W, 4)

    Returns
    -------
    intensity_flip, labels_flip : tuple of np.ndarray
        Flipped arrays
    """
    intensity_flip = intensity.copy()
    labels_flip = labels.copy()

    # Flip along axis 0 (vertical flip) with 50% probability
    if np.random.random() > 0.5:
        intensity_flip = np.flip(intensity_flip, axis=0)
        labels_flip = np.flip(labels_flip, axis=0)

    # Flip along axis 1 (horizontal flip) with 50% probability
    if np.random.random() > 0.5:
        intensity_flip = np.flip(intensity_flip, axis=1)
        labels_flip = np.flip(labels_flip, axis=1)

    return intensity_flip.copy(), labels_flip.copy()


# =============================================================================
# INTENSITY TRANSFORMS
# =============================================================================

def random_intensity_scale(
    intensity: np.ndarray,
    labels: np.ndarray,
    scale_range: Tuple[float, float] = (0.5, 2.0)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale intensity by a random factor (simulates different exposure times).

    Physics: If I → scale × I, then |F|² → scale × |F|², so |F| → √scale × |F|.
    The phase (sin(Δφ), cos(Δφ)) is unchanged by intensity scaling.

    Parameters
    ----------
    intensity : np.ndarray
        Intensity patch, shape (H, W, n_energies)
    labels : np.ndarray
        Label patch, shape (H, W, 4)
        Channels: |F_T|, |F_A|, sin(Δφ), cos(Δφ)
    scale_range : tuple
        (min_scale, max_scale) for uniform random sampling
        Default (0.5, 2.0) simulates 0.5× to 2× exposure variation

    Returns
    -------
    intensity_scaled, labels_scaled : tuple of np.ndarray
        Scaled arrays
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])
    sqrt_scale = np.sqrt(scale)

    intensity_scaled = intensity * scale

    labels_scaled = labels.copy()
    labels_scaled[..., 0] *= sqrt_scale  # |F_T| scales as √I
    labels_scaled[..., 1] *= sqrt_scale  # |F_A| scales as √I
    # labels_scaled[..., 2] unchanged     # sin(Δφ) — phase independent of intensity
    # labels_scaled[..., 3] unchanged     # cos(Δφ) — phase independent of intensity

    return intensity_scaled, labels_scaled


def add_poisson_noise(
    intensity: np.ndarray,
    noise_level: float = None,
    noise_range: Tuple[float, float] = (0.01, 0.1)
) -> np.ndarray:
    """
    Add Poisson-like noise to intensity.

    For Poisson statistics: σ(I) ≈ √I
    We simulate this as: I_noisy = I + Normal(0, noise_level × √I)

    Labels are NOT changed — noise doesn't affect the underlying truth.

    Parameters
    ----------
    intensity : np.ndarray
        Intensity patch, shape (H, W, n_energies)
    noise_level : float, optional
        Noise amplitude. If None, randomly sampled from noise_range.
    noise_range : tuple
        (min_noise, max_noise) for random sampling when noise_level is None

    Returns
    -------
    intensity_noisy : np.ndarray
        Intensity with added noise (clipped to be non-negative)
    """
    if noise_level is None:
        noise_level = np.random.uniform(noise_range[0], noise_range[1])

    # Poisson-like noise: σ ∝ √I
    sigma = noise_level * np.sqrt(intensity + 1)
    noise = np.random.normal(0, sigma)

    intensity_noisy = intensity + noise
    intensity_noisy = np.maximum(intensity_noisy, 0)  # Intensity can't be negative

    return intensity_noisy


# =============================================================================
# COMBINED AUGMENTATION PIPELINE
# =============================================================================

def augment_sample(
    intensity: np.ndarray,
    labels: np.ndarray,
    apply_rotation: bool = True,
    apply_flip: bool = True,
    apply_intensity_scale: bool = True,
    apply_noise: bool = True,
    intensity_scale_range: Tuple[float, float] = (0.5, 2.0),
    noise_range: Tuple[float, float] = (0.01, 0.1),
    noise_probability: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a complete augmentation pipeline to a single sample.

    This is the main function to use during training.

    Parameters
    ----------
    intensity : np.ndarray
        Intensity patch, shape (H, W, n_energies)
    labels : np.ndarray
        Label patch, shape (H, W, 4)
    apply_rotation : bool
        Whether to apply random 90° rotation
    apply_flip : bool
        Whether to apply random flips
    apply_intensity_scale : bool
        Whether to apply random intensity scaling
    apply_noise : bool
        Whether to add noise
    intensity_scale_range : tuple
        Range for intensity scaling factor
    noise_range : tuple
        Range for noise level
    noise_probability : float
        Probability of adding noise (0 to 1)

    Returns
    -------
    intensity_aug, labels_aug : tuple of np.ndarray
        Augmented arrays

    Example
    -------
    >>> # In PyTorch DataLoader's __getitem__:
    >>> intensity, labels = self.data[idx]
    >>> intensity, labels = augment_sample(intensity, labels)
    >>> return {'intensity': intensity, 'labels': labels}
    """
    intensity_aug = intensity.copy()
    labels_aug = labels.copy()

    # 1. Geometric transforms (don't change values, just arrangement)
    if apply_rotation:
        intensity_aug, labels_aug = random_rot90(intensity_aug, labels_aug)

    if apply_flip:
        intensity_aug, labels_aug = random_flip(intensity_aug, labels_aug)

    # 2. Intensity scaling (changes both intensity and magnitude labels)
    if apply_intensity_scale:
        intensity_aug, labels_aug = random_intensity_scale(
            intensity_aug, labels_aug,
            scale_range=intensity_scale_range
        )

    # 3. Noise injection (only changes intensity, not labels)
    if apply_noise and np.random.random() < noise_probability:
        intensity_aug = add_poisson_noise(
            intensity_aug,
            noise_range=noise_range
        )

    return intensity_aug, labels_aug


# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    print("Data Augmentation Module")
    print("=" * 60)

    # Create dummy data
    H, W, n_energies = 16, 16, 8
    intensity = np.random.rand(H, W, n_energies) * 1000 + 100
    labels = np.random.rand(H, W, 4)
    labels[..., :2] = np.abs(labels[..., :2]) * 100  # Magnitudes positive
    labels[..., 2:] = labels[..., 2:] * 2 - 1  # sin/cos in [-1, 1]

    print(f"\nOriginal shapes:")
    print(f"  Intensity: {intensity.shape}")
    print(f"  Labels: {labels.shape}")

    # Test each transform
    print("\nTesting transforms:")

    # 1. Rotation
    I_rot, L_rot = random_rot90(intensity, labels)
    print(f"  Rotation: intensity sum preserved = {np.isclose(intensity.sum(), I_rot.sum())}")

    # 2. Flip
    I_flip, L_flip = random_flip(intensity, labels)
    print(f"  Flip: intensity sum preserved = {np.isclose(intensity.sum(), I_flip.sum())}")

    # 3. Intensity scale
    I_scaled, L_scaled = random_intensity_scale(intensity, labels, scale_range=(2.0, 2.0))
    print(f"  Scale (2x): intensity doubled = {np.isclose(I_scaled.sum(), 2 * intensity.sum())}")
    print(f"  Scale (2x): |F_T| scaled by √2 = {np.isclose(L_scaled[..., 0].mean(), labels[..., 0].mean() * np.sqrt(2))}")
    print(f"  Scale (2x): sin(Δφ) unchanged = {np.isclose(L_scaled[..., 2].mean(), labels[..., 2].mean())}")

    # 4. Noise
    I_noisy = add_poisson_noise(intensity, noise_level=0.1)
    print(f"  Noise: intensity changed = {not np.allclose(intensity, I_noisy)}")
    print(f"  Noise: all values >= 0 = {(I_noisy >= 0).all()}")

    # 5. Full pipeline
    I_aug, L_aug = augment_sample(intensity, labels)
    print(f"  Full augmentation: shapes preserved = {I_aug.shape == intensity.shape and L_aug.shape == labels.shape}")

    print("\nAll tests passed!")
