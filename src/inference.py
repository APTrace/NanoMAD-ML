#!/usr/bin/env python3
"""
inference.py

Run inference on multi-energy diffraction data using trained MADNet model.

Supports both 2D (single particle) and 3D (full volume) modes.

Modes:
------
- 3D (default): Processes 3D intensity volumes slice-by-slice with Hann window blending.
- 2D: Processes single-particle patch data from .npz training files.

Usage:
------
    # 3D inference (default mode)
    python src/inference.py -c models/checkpoint_v5.pt \\
        --intensity data_3d.npy \\
        --energies energies.npy \\
        --log-transform-intensity \\
        -o output/

    # 2D inference (single particle mode)
    python src/inference.py --mode 2d -c models/checkpoint_v5.pt \\
        --test-file particle_0000.npz \\
        --log-transform-intensity \\
        -o output/

Author: Claude (Anthropic) + Thomas
Date: December 2024
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional
import time

# Optional scipy for post-processing
try:
    from scipy.ndimage import gaussian_filter1d, maximum_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# =============================================================================
# PATCH UTILITIES (shared)
# =============================================================================

def create_hann_window_2d(size: int) -> np.ndarray:
    """Create a 2D Hann window for smooth patch blending."""
    hann_1d = np.hanning(size)
    return np.outer(hann_1d, hann_1d).astype(np.float32)


def extract_patches_2d(image: np.ndarray, patch_size: int = 16) -> np.ndarray:
    """Extract non-overlapping patches from 2D multi-channel image."""
    H, W, C = image.shape
    n_patches_h = H // patch_size
    n_patches_w = W // patch_size

    patches = []
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            y_start = i * patch_size
            x_start = j * patch_size
            patch = image[y_start:y_start+patch_size, x_start:x_start+patch_size, :]
            patches.append(patch)

    return np.array(patches)


def extract_patches_overlapping(
    image: np.ndarray,
    patch_size: int = 16,
    stride: int = 8
) -> Tuple[np.ndarray, list]:
    """Extract overlapping patches with given stride."""
    H, W, C = image.shape

    patches = []
    positions = []

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch = image[i:i+patch_size, j:j+patch_size, :]
            patches.append(patch)
            positions.append((i, j))

    return np.array(patches), positions


def stitch_patches_2d(patches: np.ndarray, grid_shape: Tuple[int, int] = None) -> np.ndarray:
    """Stitch non-overlapping patches back into full image."""
    if patches.ndim == 3:
        n_patches, patch_h, patch_w = patches.shape
        has_channels = False
    else:
        n_patches, patch_h, patch_w, C = patches.shape
        has_channels = True

    if grid_shape is None:
        grid_size = int(np.sqrt(n_patches))
        assert grid_size * grid_size == n_patches
        grid_shape = (grid_size, grid_size)

    grid_h, grid_w = grid_shape
    full_h = grid_h * patch_h
    full_w = grid_w * patch_w

    if has_channels:
        full_image = np.zeros((full_h, full_w, C), dtype=patches.dtype)
    else:
        full_image = np.zeros((full_h, full_w), dtype=patches.dtype)

    for idx in range(n_patches):
        i = idx // grid_w
        j = idx % grid_w
        y_start = i * patch_h
        x_start = j * patch_w
        full_image[y_start:y_start+patch_h, x_start:x_start+patch_w] = patches[idx]

    return full_image


def stitch_patches_blended(
    patches: np.ndarray,
    positions: list,
    output_shape: Tuple[int, int],
    patch_size: int = 16
) -> np.ndarray:
    """Stitch overlapping patches using Hann window weighted blending."""
    H, W = output_shape

    output = np.zeros((H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)

    hann = create_hann_window_2d(patch_size)

    for patch, (y, x) in zip(patches, positions):
        output[y:y+patch_size, x:x+patch_size] += patch * hann
        weight_sum[y:y+patch_size, x:x+patch_size] += hann

    blended = output / np.maximum(weight_sum, 1e-8)
    return blended


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained MADNet model from checkpoint."""
    try:
        from src.mad_model import MADNet
    except ImportError:
        from mad_model import MADNet

    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint.get('config', {})
    model = MADNet(
        n_energies=config.get('n_energies', 8),
        base_channels=config.get('base_channels', 32),
        sf_feature_dim=config.get('sf_feature_dim', 32),
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', 'unknown')
    best_val = checkpoint.get('best_val_loss', None)

    print(f"  Loaded from epoch: {epoch}")
    if best_val is not None:
        print(f"  Best val loss: {best_val:.6f}")
    elif 'val_losses' in checkpoint:
        best_val = min(v['loss'] for v in checkpoint['val_losses'])
        print(f"  Best val loss: {best_val:.6f}")
    print(f"  Device: {device}")

    return model, checkpoint


# =============================================================================
# POST-PROCESSING
# =============================================================================

def enforce_unit_circle(results: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Enforce sin²(Δφ) + cos²(Δφ) = 1 by normalizing."""
    corrected = results.copy()

    if 'sin_delta_phi' in results and 'cos_delta_phi' in results:
        sin_phi = results['sin_delta_phi']
        cos_phi = results['cos_delta_phi']

        magnitude = np.sqrt(sin_phi**2 + cos_phi**2)
        magnitude = np.maximum(magnitude, 1e-10)

        corrected['sin_delta_phi'] = sin_phi / magnitude
        corrected['cos_delta_phi'] = cos_phi / magnitude
        corrected['delta_phi'] = np.arctan2(
            corrected['sin_delta_phi'],
            corrected['cos_delta_phi']
        )

    return corrected


def smooth_results_along_z(results: Dict[str, np.ndarray], sigma: float = 1.0) -> Dict[str, np.ndarray]:
    """Apply Gaussian smoothing along Z-axis to reduce slice artifacts."""
    if not SCIPY_AVAILABLE:
        print("  WARNING: scipy not available, skipping Z-smoothing")
        return results

    smoothed = {}

    for key in ['F_T', 'F_A', 'F_N']:
        if key in results:
            smoothed[key] = gaussian_filter1d(results[key], sigma=sigma, axis=0)

    if 'sin_delta_phi' in results and 'cos_delta_phi' in results:
        sin_smooth = gaussian_filter1d(results['sin_delta_phi'], sigma=sigma, axis=0)
        cos_smooth = gaussian_filter1d(results['cos_delta_phi'], sigma=sigma, axis=0)

        magnitude = np.sqrt(sin_smooth**2 + cos_smooth**2)
        magnitude = np.maximum(magnitude, 1e-10)
        sin_smooth = sin_smooth / magnitude
        cos_smooth = cos_smooth / magnitude

        smoothed['sin_delta_phi'] = sin_smooth
        smoothed['cos_delta_phi'] = cos_smooth
        smoothed['delta_phi'] = np.arctan2(sin_smooth, cos_smooth)
    elif 'delta_phi' in results:
        smoothed['delta_phi'] = gaussian_filter1d(results['delta_phi'], sigma=sigma, axis=0)
        smoothed['sin_delta_phi'] = np.sin(smoothed['delta_phi'])
        smoothed['cos_delta_phi'] = np.cos(smoothed['delta_phi'])

    return smoothed


def enhance_local_contrast(
    results: Dict[str, np.ndarray],
    gamma: float = 2.0,
    window_size: int = 7
) -> Dict[str, np.ndarray]:
    """Enhance local contrast to reduce haze artifact."""
    if not SCIPY_AVAILABLE:
        print("  WARNING: scipy not available, skipping contrast enhancement")
        return results

    enhanced = results.copy()

    for key in ['F_T', 'F_A']:
        if key in results:
            arr = results[key]
            local_max = maximum_filter(arr, size=window_size)
            relative = arr / (local_max + 1e-10)
            relative_enhanced = np.power(relative, gamma)
            enhanced[key] = relative_enhanced * local_max

    return enhanced


def compute_F_N(F_T: np.ndarray, F_A: np.ndarray, delta_phi: np.ndarray) -> np.ndarray:
    """Compute |F_N| using law of cosines: |F_N|² = |F_T|² + |F_A|² - 2|F_T||F_A|cos(Δφ)."""
    F_N_squared = F_T**2 + F_A**2 - 2 * F_T * F_A * np.cos(delta_phi)
    F_N_squared = np.maximum(F_N_squared, 0)
    return np.sqrt(F_N_squared)


# =============================================================================
# 3D DATA LOADING AND INFERENCE
# =============================================================================

def load_3d_data(intensity_path: str, energies_path: str = None,
                 f_prime_path: str = None, f_double_prime_path: str = None,
                 element: str = 'Ni', data_dir: str = '.') -> Dict[str, np.ndarray]:
    """Load 3D intensity data and scattering factors."""
    print(f"\nLoading 3D intensity data from: {intensity_path}")
    intensity = np.load(intensity_path)

    print(f"  Raw shape: {intensity.shape}")

    shape = intensity.shape
    if len(shape) != 4:
        raise ValueError(f"Expected 4D array, got shape {shape}")

    # Auto-detect axis order
    if shape[-1] == 8:
        if shape[0] == shape[1]:
            print(f"  Detected format: (H, W, D, E) - transposing to (D, H, W, E)")
            intensity = np.transpose(intensity, (2, 0, 1, 3))
        else:
            print(f"  Detected format: (D, H, W, E)")
    elif shape[0] == 8:
        print(f"  Detected format: (E, D, H, W) - transposing to (D, H, W, E)")
        intensity = np.transpose(intensity, (1, 2, 3, 0))
    else:
        print(f"  WARNING: Could not auto-detect axis order. Assuming (D, H, W, E)")

    n_slices, H, W, n_energies = intensity.shape
    print(f"  Final shape: ({n_slices}, {H}, {W}, {n_energies})")
    print(f"  Value range: [{intensity.min():.2e}, {intensity.max():.2e}]")

    # Get f'/f''
    if f_prime_path and f_double_prime_path:
        print(f"\nLoading f'/f'' from files...")
        f_prime = np.load(f_prime_path)
        f_double_prime = np.load(f_double_prime_path)
    elif energies_path:
        print(f"\nLoading energies and computing f'/f'' for {element}...")
        energies = np.load(energies_path)
        print(f"  Energies: {energies.tolist()} eV")

        from core_shell import ScatteringFactors
        sf = ScatteringFactors(data_dir=data_dir)
        f_prime = np.array([sf.get_f_prime(element, E) for E in energies])
        f_double_prime = np.array([sf.get_f_double_prime(element, E) for E in energies])
    else:
        raise ValueError("Must provide either --energies or (--f-prime and --f-double-prime)")

    print(f"  f'  range: [{f_prime.min():.2f}, {f_prime.max():.2f}]")
    print(f"  f'' range: [{f_double_prime.min():.2f}, {f_double_prime.max():.2f}]")

    return {
        'intensity': intensity.astype(np.float32),
        'f_prime': f_prime.astype(np.float32),
        'f_double_prime': f_double_prime.astype(np.float32),
        'n_slices': n_slices,
    }


def run_inference_3d(
    model: torch.nn.Module,
    data: Dict[str, np.ndarray],
    device: str = 'cuda',
    patch_size: int = 16,
    use_blending: bool = True,
    stride: int = 8,
    log_transform_intensity: bool = False,
) -> Dict[str, np.ndarray]:
    """Run inference on 3D data slice-by-slice."""
    intensity = data['intensity'].copy()

    if log_transform_intensity:
        print("  Applying log1p transform to input intensities...")
        print(f"    Before: range [{intensity.min():.2e}, {intensity.max():.2e}]")
        intensity = np.log1p(intensity)
        print(f"    After:  range [{intensity.min():.2f}, {intensity.max():.2f}]")

    f_prime = data['f_prime']
    f_double_prime = data['f_double_prime']
    n_slices, H, W, n_energies = intensity.shape

    print(f"\n{'='*60}")
    print("RUNNING 3D INFERENCE")
    print(f"{'='*60}")
    print(f"  Volume: {n_slices} slices × {H}×{W} pixels")
    print(f"  Mode: {'OVERLAPPING with Hann blending' if use_blending else 'NON-OVERLAPPING'}")
    if use_blending:
        print(f"  Stride: {stride}")

    fp_tensor = torch.from_numpy(f_prime).float().to(device)
    fpp_tensor = torch.from_numpy(f_double_prime).float().to(device)

    F_T_3d = np.zeros((n_slices, H, W), dtype=np.float32)
    F_A_3d = np.zeros((n_slices, H, W), dtype=np.float32)
    sin_phi_3d = np.zeros((n_slices, H, W), dtype=np.float32)
    cos_phi_3d = np.zeros((n_slices, H, W), dtype=np.float32)

    start_time = time.time()

    with torch.no_grad():
        for z in range(n_slices):
            slice_2d = intensity[z]

            if use_blending:
                patches, positions = extract_patches_overlapping(slice_2d, patch_size, stride)
            else:
                patches = extract_patches_2d(slice_2d, patch_size)
                positions = None

            patches_tensor = torch.from_numpy(patches).float()
            patches_tensor = patches_tensor.permute(0, 3, 1, 2).to(device)

            n_patches = patches_tensor.shape[0]
            fp_batch = fp_tensor.unsqueeze(0).expand(n_patches, -1)
            fpp_batch = fpp_tensor.unsqueeze(0).expand(n_patches, -1)

            output = model(patches_tensor, fp_batch, fpp_batch)
            output = output.permute(0, 2, 3, 1).cpu().numpy()

            F_T_patches = np.expm1(output[..., 0])
            F_A_patches = np.expm1(output[..., 1])
            sin_patches = output[..., 2]
            cos_patches = output[..., 3]

            if use_blending:
                F_T_3d[z] = stitch_patches_blended(F_T_patches, positions, (H, W), patch_size)
                F_A_3d[z] = stitch_patches_blended(F_A_patches, positions, (H, W), patch_size)
                sin_phi_3d[z] = stitch_patches_blended(sin_patches, positions, (H, W), patch_size)
                cos_phi_3d[z] = stitch_patches_blended(cos_patches, positions, (H, W), patch_size)
            else:
                F_T_3d[z] = stitch_patches_2d(F_T_patches)
                F_A_3d[z] = stitch_patches_2d(F_A_patches)
                sin_phi_3d[z] = stitch_patches_2d(sin_patches)
                cos_phi_3d[z] = stitch_patches_2d(cos_patches)

            if (z + 1) % 10 == 0 or z == n_slices - 1:
                elapsed = time.time() - start_time
                rate = (z + 1) / elapsed
                eta = (n_slices - z - 1) / rate if rate > 0 else 0
                print(f"  Slice {z+1:3d}/{n_slices} | {elapsed:.1f}s elapsed | ETA: {eta:.1f}s")

    delta_phi_3d = np.arctan2(sin_phi_3d, cos_phi_3d)
    F_N_3d = compute_F_N(F_T_3d, F_A_3d, delta_phi_3d)

    print(f"\n  Total time: {time.time() - start_time:.1f}s")

    return {
        'F_T': F_T_3d,
        'F_A': F_A_3d,
        'F_N': F_N_3d,
        'delta_phi': delta_phi_3d,
        'sin_delta_phi': sin_phi_3d,
        'cos_delta_phi': cos_phi_3d,
    }


# =============================================================================
# 2D DATA LOADING AND INFERENCE
# =============================================================================

def load_2d_data_npz(filepath: str) -> Dict[str, np.ndarray]:
    """Load 2D test data from .npz file (training data format)."""
    print(f"Loading test data from: {filepath}")

    data = np.load(filepath, allow_pickle=True)

    result = {
        'X': data['X'],
        'f_prime': data['f_prime'],
        'f_double_prime': data['f_double_prime'],
    }

    if 'Y' in data:
        result['Y'] = data['Y']
        print(f"  Ground truth labels: available")
    else:
        print(f"  Ground truth labels: not available")

    if 'energies' in data:
        result['energies'] = data['energies']

    print(f"  Intensity patches: {result['X'].shape}")

    return result


def load_2d_data_separate(
    intensity_path: str,
    f_prime_path: str = None,
    f_double_prime_path: str = None,
    energies_path: str = None,
    element: str = 'Ni',
    data_dir: str = '.'
) -> Dict[str, np.ndarray]:
    """Load 2D test data from separate numpy files."""
    print(f"Loading intensity from: {intensity_path}")

    X = np.load(intensity_path)
    if X.ndim == 3:
        X = X[np.newaxis, ...]

    print(f"  Intensity shape: {X.shape}")

    if f_prime_path and f_double_prime_path:
        f_prime = np.load(f_prime_path)
        f_double_prime = np.load(f_double_prime_path)
    elif energies_path:
        energies = np.load(energies_path)
        print(f"  Computing f'/f'' for {element}...")

        from core_shell import ScatteringFactors
        sf = ScatteringFactors(data_dir=data_dir)
        f_prime = np.array([sf.get_f_prime(element, E) for E in energies])
        f_double_prime = np.array([sf.get_f_double_prime(element, E) for E in energies])
    else:
        raise ValueError("Must provide either (--f-prime and --f-double-prime) or --energies")

    return {
        'X': X,
        'f_prime': f_prime,
        'f_double_prime': f_double_prime,
    }


def run_inference_2d(
    model: torch.nn.Module,
    data: Dict[str, np.ndarray],
    device: str = 'cuda',
    log_transform_intensity: bool = False,
) -> Dict[str, np.ndarray]:
    """Run inference on 2D patch data."""
    try:
        from src.mad_model import predict_and_convert
    except ImportError:
        from mad_model import predict_and_convert

    print("\nRunning 2D inference...")

    X = data['X'].copy()

    if log_transform_intensity:
        print(f"  Applying log1p transform...")
        print(f"    Before: range [{X.min():.2e}, {X.max():.2e}]")
        X = np.log1p(X)
        print(f"    After:  range [{X.min():.2f}, {X.max():.2f}]")

    intensity_tensor = torch.from_numpy(X).float()
    intensity_tensor = intensity_tensor.permute(0, 3, 1, 2)

    fp_tensor = torch.from_numpy(data['f_prime']).float()
    fs_tensor = torch.from_numpy(data['f_double_prime']).float()

    results = predict_and_convert(
        model,
        intensity_tensor,
        fp_tensor,
        fs_tensor,
        log_scale_magnitudes=True
    )

    print(f"  Processed {X.shape[0]} patches")
    print(f"  |F_T| range: [{results['F_T'].min():.1f}, {results['F_T'].max():.1f}]")
    print(f"  |F_A| range: [{results['F_A'].min():.1f}, {results['F_A'].max():.1f}]")

    return results


# =============================================================================
# EXPORT AND VISUALIZATION
# =============================================================================

def export_3d_arrays(results: Dict[str, np.ndarray], output_dir: str):
    """Export 3D arrays as .npy files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nExporting 3D arrays to: {output_path}")

    for name in ['F_T', 'F_A', 'F_N', 'delta_phi']:
        arr = results[name]
        filepath = output_path / f"{name}_3d.npy"
        np.save(filepath, arr)
        print(f"  {filepath.name}: shape={arr.shape}, range=[{arr.min():.2f}, {arr.max():.2f}]")

    combined_path = output_path / 'all_mad_parameters_3d.npz'
    np.savez(combined_path, **results)
    print(f"  all_mad_parameters_3d.npz")


def export_2d_arrays(results: Dict[str, np.ndarray], output_dir: str):
    """Export 2D stitched arrays as .npz files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nExporting 2D arrays to: {output_path}")

    for name in ['F_T', 'F_A', 'F_N', 'delta_phi']:
        if name in results:
            arr = results[name]
            filepath = output_path / f"{name}.npz"
            np.savez(filepath, data=arr, shape=arr.shape)
            print(f"  {filepath.name}: shape={arr.shape}")

    combined_path = output_path / 'all_mad_parameters.npz'
    np.savez(combined_path, **results)


def visualize_3d_results(results: Dict[str, np.ndarray], save_path: str = None):
    """Visualize central slices of 3D results."""
    import matplotlib.pyplot as plt

    F_T = results['F_T']
    F_A = results['F_A']
    F_N = results['F_N']
    delta_phi = results['delta_phi']

    n_slices = F_T.shape[0]
    z_center = n_slices // 2
    y_center = F_T.shape[1] // 2
    x_center = F_T.shape[2] // 2

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # Row 0: XY plane (Z center)
    axes[0, 0].imshow(F_T[z_center], cmap='viridis')
    axes[0, 0].set_title(f'|F_T| - Z={z_center} (XY)')
    axes[0, 1].imshow(F_A[z_center], cmap='plasma')
    axes[0, 1].set_title(f'|F_A| - Z={z_center}')
    axes[0, 2].imshow(F_N[z_center], cmap='cividis')
    axes[0, 2].set_title(f'|F_N| - Z={z_center}')
    im = axes[0, 3].imshow(delta_phi[z_center], cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[0, 3].set_title(f'Δφ - Z={z_center}')
    plt.colorbar(im, ax=axes[0, 3])

    # Row 1: XZ plane (Y center)
    axes[1, 0].imshow(F_T[:, y_center, :], cmap='viridis', aspect='auto')
    axes[1, 0].set_title(f'|F_T| - Y={y_center} (XZ)')
    axes[1, 1].imshow(F_A[:, y_center, :], cmap='plasma', aspect='auto')
    axes[1, 1].set_title(f'|F_A| - Y={y_center}')
    axes[1, 2].imshow(F_N[:, y_center, :], cmap='cividis', aspect='auto')
    axes[1, 2].set_title(f'|F_N| - Y={y_center}')
    axes[1, 3].imshow(delta_phi[:, y_center, :], cmap='twilight', vmin=-np.pi, vmax=np.pi, aspect='auto')
    axes[1, 3].set_title(f'Δφ - Y={y_center}')

    # Row 2: YZ plane (X center)
    axes[2, 0].imshow(F_T[:, :, x_center], cmap='viridis', aspect='auto')
    axes[2, 0].set_title(f'|F_T| - X={x_center} (YZ)')
    axes[2, 1].imshow(F_A[:, :, x_center], cmap='plasma', aspect='auto')
    axes[2, 1].set_title(f'|F_A| - X={x_center}')
    axes[2, 2].imshow(F_N[:, :, x_center], cmap='cividis', aspect='auto')
    axes[2, 2].set_title(f'|F_N| - X={x_center}')
    axes[2, 3].imshow(delta_phi[:, :, x_center], cmap='twilight', vmin=-np.pi, vmax=np.pi, aspect='auto')
    axes[2, 3].set_title(f'Δφ - X={x_center}')

    plt.suptitle('3D MAD Parameter Extraction - Orthogonal Slices', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved: {save_path}")

    plt.show()


def visualize_2d_results(results: Dict[str, np.ndarray], save_path: str = None):
    """Visualize 2D stitched results."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    arrays = [
        ('F_T', results['F_T'], 'viridis', '|F_T|'),
        ('F_A', results['F_A'], 'plasma', '|F_A|'),
        ('F_N', results['F_N'], 'cividis', '|F_N|'),
        ('delta_phi', results['delta_phi'], 'twilight', 'Δφ'),
    ]

    for idx, (key, arr, cmap, title) in enumerate(arrays):
        vmin = -np.pi if key == 'delta_phi' else 0
        vmax = np.pi if key == 'delta_phi' else np.percentile(arr[arr > 0], 99) if np.any(arr > 0) else 1

        im = axes[0, idx].imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0, idx].set_title(f'{title} (max={arr.max():.1f})')
        plt.colorbar(im, ax=axes[0, idx])

        if key != 'delta_phi':
            im = axes[1, idx].imshow(np.log10(arr + 1), cmap=cmap)
            axes[1, idx].set_title(f'log10({title} + 1)')
            plt.colorbar(im, ax=axes[1, idx])
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = results['F_A'] / (results['F_T'] + 1e-10)
                ratio = np.clip(ratio, 0, 1)
            im = axes[1, idx].imshow(ratio, cmap='coolwarm', vmin=0, vmax=1)
            axes[1, idx].set_title('|F_A|/|F_T| (Ni fraction)')
            plt.colorbar(im, ax=axes[1, idx])

    plt.suptitle('MAD Parameter Extraction Results', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved: {save_path}")

    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run MAD inference (2D or 3D mode)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 3D inference (default)
    python src/inference.py -c models/checkpoint_v5.pt \\
        --intensity data_3d.npy \\
        --energies energies.npy \\
        --log-transform-intensity \\
        -o output_3d/

    # 2D inference on single particle
    python src/inference.py --mode 2d -c models/checkpoint_v5.pt \\
        --test-file particle_0000.npz \\
        --log-transform-intensity \\
        -o output_2d/
        """
    )

    # Mode
    parser.add_argument('--mode', type=str, default='3d', choices=['2d', '3d'],
                        help='Inference mode: 2d (single particle) or 3d (volume, default)')

    # Model
    parser.add_argument('-c', '--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt)')

    # Data sources
    parser.add_argument('-t', '--test-file', type=str, default=None,
                        help='Path to test data .npz file (2D mode)')
    parser.add_argument('--intensity', type=str, default=None,
                        help='Path to intensity array (.npy)')
    parser.add_argument('--energies', type=str, default=None,
                        help='Path to energies array (.npy)')
    parser.add_argument('--f-prime', type=str, default=None,
                        help="Path to f'(E) array")
    parser.add_argument('--f-double-prime', type=str, default=None,
                        help="Path to f''(E) array")
    parser.add_argument('--element', type=str, default='Ni',
                        help='Element for f\'/f\'\' (default: Ni)')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing .f1f2 files (default: data/)')

    # Output
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip visualization')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')

    # Input preprocessing
    parser.add_argument('--log-transform-intensity', action='store_true',
                        help='Apply log1p to input intensities. '
                             'CRITICAL: Must match training config!')

    # 3D-specific options
    parser.add_argument('--no-blend', action='store_true',
                        help='Disable overlapping patch blending (3D mode)')
    parser.add_argument('--stride', type=int, default=4,
                        help='Stride for overlapping patches (default: 4 = 75%% overlap)')

    # Post-processing
    parser.add_argument('--smooth-z', type=float, default=1.0, metavar='SIGMA',
                        help='Z-axis Gaussian smoothing sigma (default: 1.0)')
    parser.add_argument('--no-smooth-z', action='store_true',
                        help='Disable Z-axis smoothing')
    parser.add_argument('--no-unit-circle', action='store_true',
                        help='Disable unit circle enforcement')
    parser.add_argument('--enhance-contrast', action='store_true',
                        help='Enable local contrast enhancement (experimental)')
    parser.add_argument('--contrast-gamma', type=float, default=2.0,
                        help='Gamma for contrast enhancement')
    parser.add_argument('--contrast-window', type=int, default=7,
                        help='Window size for local max computation')

    args = parser.parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    model, checkpoint = load_model(args.checkpoint, args.device)

    # =========================================================================
    # 3D MODE
    # =========================================================================
    if args.mode == '3d':
        if not args.intensity:
            parser.error("3D mode requires --intensity")

        data = load_3d_data(
            intensity_path=args.intensity,
            energies_path=args.energies,
            f_prime_path=args.f_prime,
            f_double_prime_path=args.f_double_prime,
            element=args.element,
            data_dir=args.data_dir,
        )

        use_blending = not args.no_blend
        results = run_inference_3d(
            model, data, args.device,
            use_blending=use_blending,
            stride=args.stride,
            log_transform_intensity=args.log_transform_intensity,
        )

        # Post-processing
        print(f"\n{'='*60}")
        print("POST-PROCESSING")
        print(f"{'='*60}")

        if not args.no_unit_circle:
            print("  Enforcing unit circle constraint...")
            results = enforce_unit_circle(results)

        if not args.no_smooth_z and args.smooth_z > 0:
            print(f"  Applying Z-axis smoothing (sigma={args.smooth_z})...")
            results = smooth_results_along_z(results, sigma=args.smooth_z)

        if args.enhance_contrast:
            print(f"  Applying contrast enhancement (gamma={args.contrast_gamma})...")
            results = enhance_local_contrast(
                results,
                gamma=args.contrast_gamma,
                window_size=args.contrast_window
            )

        # Recompute F_N after post-processing
        results['F_N'] = compute_F_N(results['F_T'], results['F_A'], results['delta_phi'])

        print(f"\n{'='*60}")
        print("FINAL OUTPUT STATISTICS")
        print(f"{'='*60}")
        print(f"  |F_T| range: [{results['F_T'].min():.2f}, {results['F_T'].max():.2f}]")
        print(f"  |F_A| range: [{results['F_A'].min():.2f}, {results['F_A'].max():.2f}]")
        print(f"  |F_N| range: [{results['F_N'].min():.2f}, {results['F_N'].max():.2f}]")
        print(f"  Δφ range: [{results['delta_phi'].min():.2f}, {results['delta_phi'].max():.2f}] rad")

        export_3d_arrays(results, args.output_dir)

        if not args.no_plot:
            viz_path = output_path / 'visualization_3d.png'
            visualize_3d_results(results, save_path=str(viz_path))

    # =========================================================================
    # 2D MODE
    # =========================================================================
    else:
        if args.test_file:
            data = load_2d_data_npz(args.test_file)
        elif args.intensity:
            data = load_2d_data_separate(
                intensity_path=args.intensity,
                f_prime_path=args.f_prime,
                f_double_prime_path=args.f_double_prime,
                energies_path=args.energies,
                element=args.element,
                data_dir=args.data_dir,
            )
        else:
            parser.error("2D mode requires --test-file or --intensity")

        predictions = run_inference_2d(
            model, data, args.device,
            log_transform_intensity=args.log_transform_intensity,
        )

        # Stitch patches
        stitched = {}
        for key in ['F_T', 'F_A', 'delta_phi', 'sin_delta_phi', 'cos_delta_phi']:
            if key in predictions:
                stitched[key] = stitch_patches_2d(predictions[key])

        # Compute F_N
        stitched['F_N'] = compute_F_N(stitched['F_T'], stitched['F_A'], stitched['delta_phi'])

        print(f"\n{'='*60}")
        print("OUTPUT STATISTICS")
        print(f"{'='*60}")
        print(f"  |F_T| range: [{stitched['F_T'].min():.2f}, {stitched['F_T'].max():.2f}]")
        print(f"  |F_A| range: [{stitched['F_A'].min():.2f}, {stitched['F_A'].max():.2f}]")
        print(f"  |F_N| range: [{stitched['F_N'].min():.2f}, {stitched['F_N'].max():.2f}]")

        export_2d_arrays(stitched, args.output_dir)

        if not args.no_plot:
            viz_path = output_path / 'visualization_2d.png'
            visualize_2d_results(stitched, save_path=str(viz_path))

    print("\n Done!")


if __name__ == '__main__':
    main()
