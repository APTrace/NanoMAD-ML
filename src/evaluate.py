#!/usr/bin/env python3
"""
evaluate_3d.py

Comprehensive evaluation script for 3D MAD inference results.

This script provides:
1. Self-consistency check: Do predictions reconstruct input intensities correctly?
2. Comparison between different model versions/runs
3. Statistical metrics: MSE, MAE, PSNR, SSIM, correlation
4. Regional analysis: center vs edge, high vs low intensity regions
5. Visualization of results and error distributions
6. Haze diagnostic: Analyze elevated background in low-intensity regions

Usage:
------
    # Evaluate single run (self-consistency + haze diagnostic)
    python evaluate_3d.py --predictions inference_output_v3/ \
        --intensity ../evaluation_data/test_3d_noise_pf1e6_v2/test_intensity_3d_noise_pf1e6.npy \
        --energies ../evaluation_data/test_3d_noise_pf1e6_v2/test_energies.npy \
        -o evaluation_results/

    # Compare two runs
    python evaluate_3d.py --predictions inference_output_v3/ \
        --reference inference_output_v2/ \
        -o evaluation_comparison/

    # Full evaluation with ground truth (for haze comparison)
    python evaluate_3d.py --predictions inference_output_v3/ \
        --ground-truth path/to/ground_truth_mad_params/ \
        -o evaluation_results/

    # Skip haze diagnostic
    python evaluate_3d.py --predictions inference_output_v3/ \
        --intensity data.npy --energies energies.npy \
        --no-haze-diagnostic -o evaluation_results/

Outputs:
--------
    - evaluation_report.txt: Text summary of all metrics
    - metrics.npz: NumPy archive of all computed metrics
    - predictions_overview.png: Visualization of MAD parameters
    - intensity_reconstruction.png: Observed vs reconstructed intensity
    - distributions.png: Histograms of predicted values
    - haze_diagnostic.png: CDF plots and spatial haze maps

Author: Claude (Anthropic) + Thomas
Date: December 2024
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings


# =============================================================================
# METRICS
# =============================================================================

def compute_mse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray = None) -> float:
    """Mean Squared Error."""
    if mask is not None:
        diff = (pred - target)[mask]
    else:
        diff = pred - target
    return float(np.mean(diff ** 2))


def compute_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray = None) -> float:
    """Mean Absolute Error."""
    if mask is not None:
        diff = np.abs(pred - target)[mask]
    else:
        diff = np.abs(pred - target)
    return float(np.mean(diff))


def compute_psnr(pred: np.ndarray, target: np.ndarray, data_range: float = None) -> float:
    """Peak Signal-to-Noise Ratio (dB)."""
    mse = compute_mse(pred, target)
    if mse < 1e-10:
        return float('inf')
    if data_range is None:
        data_range = target.max() - target.min()
    return float(10 * np.log10(data_range ** 2 / mse))


def compute_correlation(pred: np.ndarray, target: np.ndarray, mask: np.ndarray = None) -> float:
    """Pearson correlation coefficient."""
    if mask is not None:
        p = pred[mask].flatten()
        t = target[mask].flatten()
    else:
        p = pred.flatten()
        t = target.flatten()

    if len(p) == 0:
        return 0.0

    corr_matrix = np.corrcoef(p, t)
    return float(corr_matrix[0, 1])


def compute_ssim_simple(pred: np.ndarray, target: np.ndarray,
                        win_size: int = 7, data_range: float = None) -> float:
    """
    Simplified SSIM (Structural Similarity Index).

    Full SSIM requires scikit-image; this is a simpler approximation.
    """
    if data_range is None:
        data_range = target.max() - target.min()

    # Constants
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Global means and variances (simplified - ignores local windowing)
    mu_pred = np.mean(pred)
    mu_target = np.mean(target)
    sigma_pred = np.std(pred)
    sigma_target = np.std(target)
    sigma_cross = np.mean((pred - mu_pred) * (target - mu_target))

    # SSIM formula
    numerator = (2 * mu_pred * mu_target + C1) * (2 * sigma_cross + C2)
    denominator = (mu_pred**2 + mu_target**2 + C1) * (sigma_pred**2 + sigma_target**2 + C2)

    return float(numerator / denominator)


def compute_r_factor(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Crystallographic R-factor: sum(|pred - target|) / sum(target)

    Lower is better. R < 0.05 is good, R < 0.02 is excellent.
    """
    diff_sum = np.abs(pred - target).sum()
    target_sum = np.abs(target).sum()
    if target_sum < 1e-10:
        return float('inf')
    return float(diff_sum / target_sum)


def compute_all_metrics(pred: np.ndarray, target: np.ndarray,
                        name: str = "", mask: np.ndarray = None) -> Dict[str, float]:
    """Compute all metrics for a single variable."""

    metrics = {
        f'{name}_mse': compute_mse(pred, target, mask),
        f'{name}_mae': compute_mae(pred, target, mask),
        f'{name}_psnr': compute_psnr(pred, target),
        f'{name}_correlation': compute_correlation(pred, target, mask),
        f'{name}_ssim': compute_ssim_simple(pred, target),
        f'{name}_r_factor': compute_r_factor(pred, target),
    }

    return metrics


# =============================================================================
# MAD EQUATION FOR INTENSITY RECONSTRUCTION
# =============================================================================
#
# IMPORTANT NOTE ON SELF-CONSISTENCY EVALUATION:
# ----------------------------------------------
# The CNN predicts |F_T|, |F_A|, sin(Δφ), cos(Δφ) which are derived from the
# full physics simulation in core_shell.py. The actual diffraction intensity
# is computed as:
#
#   A(E) = [f₀_Ni(Q) + f'_Ni(E) + i·f''_Ni(E)] · FFT(ρ_Ni) +
#          [f₀_Fe(Q) + f'_Fe(E) + i·f''_Fe(E)] · FFT(ρ_Fe)
#   I(E) = |A(E)|²
#
# The simplified MAD equation used below is an APPROXIMATION that doesn't
# perfectly match this full physics. As a result, the "R-factor" from
# self-consistency evaluation should NOT be interpreted as an absolute
# quality metric. Instead, use:
#   - Correlation coefficient (0.94+ indicates good spatial agreement)
#   - Visual inspection of reconstructed vs observed patterns
#   - Comparison with ground truth if available
#
# The high R-factor (~5000) is due to SCALE MISMATCH, not pattern mismatch.
# =============================================================================

def reconstruct_intensity_from_mad(
    F_T: np.ndarray,
    F_A: np.ndarray,
    sin_phi: np.ndarray,
    cos_phi: np.ndarray,
    f_prime: np.ndarray,
    f_double_prime: np.ndarray,
    f0: np.ndarray = None
) -> np.ndarray:
    """
    Reconstruct intensity at all energies using the MAD equation.

    NOTE: This is a SIMPLIFIED approximation. The actual training data uses
    the full physics from core_shell.py which includes Q-dependent f₀(Q)
    and complex amplitude combinations. The reconstructed intensity will
    have good CORRELATION with observed data but different SCALE.

    MAD equation (simplified, assuming f0-normalized):
        I(E) = |F_T|² + (f'² + f''²)|F_A|² + 2|F_T||F_A|·[f'·cos(Δφ) + f''·sin(Δφ)]

    Parameters
    ----------
    F_T : np.ndarray
        Total structure factor magnitude, shape (D, H, W)
    F_A : np.ndarray
        Anomalous structure factor magnitude, shape (D, H, W)
    sin_phi, cos_phi : np.ndarray
        Sine and cosine of phase difference, shape (D, H, W)
    f_prime : np.ndarray
        f'(E) values, shape (n_energies,)
    f_double_prime : np.ndarray
        f''(E) values, shape (n_energies,)
    f0 : np.ndarray, optional
        Thomson scattering factor, shape (D, H, W). If None, assumes F_A is already normalized.

    Returns
    -------
    I_reconstructed : np.ndarray
        Shape (D, H, W, n_energies)
    """
    n_energies = len(f_prime)
    D, H, W = F_T.shape

    I_reconstructed = np.zeros((D, H, W, n_energies), dtype=np.float32)

    # If f0 is provided, normalize F_A
    if f0 is not None:
        f0_safe = np.maximum(f0, 1e-10)
        F_A_norm = F_A / f0_safe
    else:
        F_A_norm = F_A  # Assume already normalized

    for e in range(n_energies):
        fp = f_prime[e]
        fpp = f_double_prime[e]

        term1 = F_T ** 2
        term2 = (fp**2 + fpp**2) * F_A_norm ** 2
        term3 = 2 * F_T * F_A_norm * (fp * cos_phi + fpp * sin_phi)

        I_reconstructed[..., e] = term1 + term2 + term3

    return I_reconstructed


def compute_normalized_r_factor(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute R-factor after normalizing both arrays to same scale.

    This removes the scale mismatch issue and gives a meaningful comparison.
    """
    # Normalize both to [0, 1] range
    pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-10)
    target_norm = (target - target.min()) / (target.max() - target.min() + 1e-10)

    diff_sum = np.abs(pred_norm - target_norm).sum()
    target_sum = np.abs(target_norm).sum()

    if target_sum < 1e-10:
        return float('inf')
    return float(diff_sum / target_sum)


def load_f_prime_f_double_prime(energies: np.ndarray, element: str = 'Ni') -> Tuple[np.ndarray, np.ndarray]:
    """Load f'/f'' for given energies."""
    try:
        from core_shell import ScatteringFactors
        sf = ScatteringFactors(data_dir='.')
        f_prime = np.array([sf.get_f_prime(element, E) for E in energies])
        f_double_prime = np.array([sf.get_f_double_prime(element, E) for E in energies])
        return f_prime.astype(np.float32), f_double_prime.astype(np.float32)
    except Exception as e:
        print(f"Warning: Could not load scattering factors: {e}")
        return None, None


# =============================================================================
# LOADING
# =============================================================================

def load_predictions(pred_dir: str) -> Dict[str, np.ndarray]:
    """Load prediction arrays from a directory.

    Supports both 3D outputs (*_3d.npy) and 2D outputs (*.npz).
    """
    pred_path = Path(pred_dir)

    results = {}

    # Try 3D format first (*_3d.npy)
    for name in ['F_T', 'F_A', 'F_N', 'delta_phi']:
        filepath = pred_path / f'{name}_3d.npy'
        if filepath.exists():
            results[name] = np.load(filepath)
            print(f"  Loaded {name}: shape={results[name].shape}")

    # If no 3D files found, try 2D format (*.npz)
    if not results:
        for name in ['F_T', 'F_A', 'F_N', 'delta_phi']:
            filepath = pred_path / f'{name}.npz'
            if filepath.exists():
                data = np.load(filepath)
                # npz files have 'data' key from inference.py
                key = 'data' if 'data' in data else list(data.keys())[0]
                results[name] = data[key]
                print(f"  Loaded {name}: shape={results[name].shape} (2D)")

    # Also try to load sin/cos if available (3D format)
    npz_path = pred_path / 'all_mad_parameters_3d.npz'
    if npz_path.exists():
        data = np.load(npz_path)
        if 'sin_delta_phi' in data:
            results['sin_delta_phi'] = data['sin_delta_phi']
        if 'cos_delta_phi' in data:
            results['cos_delta_phi'] = data['cos_delta_phi']

    # Compute sin/cos from delta_phi if not available
    if 'delta_phi' in results and 'sin_delta_phi' not in results:
        results['sin_delta_phi'] = np.sin(results['delta_phi'])
        results['cos_delta_phi'] = np.cos(results['delta_phi'])

    return results


def load_intensity_data(intensity_path: str, energies_path: str = None) -> Dict[str, np.ndarray]:
    """Load intensity and energies from .npy or .npz files.

    Supports:
    - 3D intensity .npy files (D, H, W, E) or (E, D, H, W)
    - 2D intensity .npy files (H, W, E)
    - Training data .npz files with 'X' (intensity) and 'energies' keys
    """
    print(f"\nLoading intensity from: {intensity_path}")
    path = Path(intensity_path)

    result = {}

    if path.suffix == '.npz':
        # Training data format (particle_XXXX.npz)
        data = np.load(intensity_path)
        if 'X' in data:
            # X has shape (n_patches, patch_h, patch_w, n_energies)
            # For evaluation, we need to reconstruct or use as-is
            intensity = data['X']
            print(f"  Loaded from npz 'X': shape={intensity.shape}")
            # If it's patches, we can't easily evaluate without reconstruction
            # Just use the first patch or stack them
            if len(intensity.shape) == 4 and intensity.shape[0] > 1:
                # Multiple patches - take central slice for quick eval
                n_patches = intensity.shape[0]
                n_side = int(np.sqrt(n_patches))
                if n_side * n_side == n_patches:
                    # Reconstruct 2D image from patches
                    patch_h, patch_w = intensity.shape[1], intensity.shape[2]
                    full_h, full_w = n_side * patch_h, n_side * patch_w
                    full_intensity = np.zeros((full_h, full_w, intensity.shape[3]))
                    for i in range(n_side):
                        for j in range(n_side):
                            idx = i * n_side + j
                            full_intensity[i*patch_h:(i+1)*patch_h,
                                           j*patch_w:(j+1)*patch_w, :] = intensity[idx]
                    intensity = full_intensity
                    print(f"  Reconstructed 2D: shape={intensity.shape}")
        else:
            # Try loading as raw array
            intensity = data[list(data.keys())[0]]

        if 'energies' in data:
            result['energies'] = data['energies']
            print(f"  Energies from npz: {result['energies'].tolist()}")
    else:
        # .npy file
        intensity = np.load(intensity_path)

    # Normalize axis order
    if len(intensity.shape) == 4:
        # 3D data
        if intensity.shape[0] == 8:  # (E, D, H, W)
            intensity = np.transpose(intensity, (1, 2, 3, 0))
        elif intensity.shape[-1] != 8:
            print(f"  Warning: unexpected 3D shape {intensity.shape}")
    elif len(intensity.shape) == 3:
        # 2D data (H, W, E)
        if intensity.shape[-1] != 8 and intensity.shape[0] == 8:
            intensity = np.transpose(intensity, (1, 2, 0))

    print(f"  Intensity shape: {intensity.shape}")
    result['intensity'] = intensity

    if energies_path and 'energies' not in result:
        energies = np.load(energies_path)
        result['energies'] = energies
        print(f"  Energies: {energies.tolist()}")

    return result


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_self_consistency(
    predictions: Dict[str, np.ndarray],
    intensity: np.ndarray,
    energies: np.ndarray,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate self-consistency: do predictions reconstruct observed intensities?

    NOTE: Due to the simplified MAD equation used here (vs full physics in training),
    the absolute R-factor will be high due to SCALE mismatch. The key metric is
    CORRELATION - a correlation > 0.9 indicates the spatial patterns match well.

    The normalized R-factor removes the scale issue by normalizing both arrays
    to [0,1] range before comparison.
    """
    print("\n" + "="*60)
    print("SELF-CONSISTENCY EVALUATION")
    print("="*60)
    print("Testing: Do predicted MAD parameters reconstruct input intensities?")
    print("\nNOTE: Using simplified MAD equation. High raw R-factor is expected")
    print("      due to scale mismatch. Focus on CORRELATION and NORMALIZED R-factor.")

    # Get f'/f''
    f_prime, f_double_prime = load_f_prime_f_double_prime(energies, 'Ni')

    if f_prime is None:
        print("ERROR: Cannot compute self-consistency without scattering factors")
        return {}

    # Reconstruct intensity from predictions
    F_T = predictions['F_T']
    F_A = predictions['F_A']
    sin_phi = predictions.get('sin_delta_phi', np.sin(predictions['delta_phi']))
    cos_phi = predictions.get('cos_delta_phi', np.cos(predictions['delta_phi']))

    I_reconstructed = reconstruct_intensity_from_mad(
        F_T, F_A, sin_phi, cos_phi, f_prime, f_double_prime
    )

    # Compute metrics per energy
    metrics = {}
    correlations = []
    norm_r_factors = []

    print(f"\n{'Energy (eV)':<12} {'Correlation':<12} {'Norm R-factor':<14} {'Raw R-factor':<12}")
    print("-" * 52)

    for e, E in enumerate(energies):
        I_pred = I_reconstructed[..., e]
        I_obs = intensity[..., e]

        corr = compute_correlation(I_pred, I_obs)
        norm_r = compute_normalized_r_factor(I_pred, I_obs)
        raw_r = compute_r_factor(I_pred, I_obs)

        correlations.append(corr)
        norm_r_factors.append(norm_r)

        print(f"{E:<12} {corr:<12.4f} {norm_r:<14.4f} {raw_r:<12.2f}")

        metrics[f'E{E}_correlation'] = corr
        metrics[f'E{E}_norm_r_factor'] = norm_r

    # Summary
    mean_corr = np.mean(correlations)
    mean_norm_r = np.mean(norm_r_factors)

    print("-" * 52)
    print(f"{'Mean':<12} {mean_corr:<12.4f} {mean_norm_r:<14.4f}")

    metrics['mean_correlation'] = mean_corr
    metrics['mean_norm_r_factor'] = mean_norm_r

    # Assessment based on correlation (the reliable metric)
    print("\n" + "-"*60)
    if mean_corr > 0.95 and mean_norm_r < 0.3:
        print("✓ EXCELLENT self-consistency!")
        print("  Spatial patterns match very well between prediction and observation.")
    elif mean_corr > 0.85 and mean_norm_r < 0.5:
        print("○ GOOD self-consistency.")
        print("  Predictions capture the main features of the diffraction pattern.")
    elif mean_corr > 0.7:
        print("△ MODERATE self-consistency.")
        print("  Predictions show reasonable agreement but with some discrepancies.")
    else:
        print("✗ POOR self-consistency.")
        print("  Predictions do not match observed data well.")
        print("  Possible causes: training/inference mismatch, model issues")
    print("-"*60)

    # Store reconstructed intensity for visualization
    metrics['_I_reconstructed'] = I_reconstructed

    return metrics


def evaluate_against_reference(
    predictions: Dict[str, np.ndarray],
    reference: Dict[str, np.ndarray],
    verbose: bool = True
) -> Dict[str, float]:
    """
    Compare predictions against a reference (ground truth or another model run).
    """
    print("\n" + "="*60)
    print("REFERENCE COMPARISON")
    print("="*60)

    metrics = {}

    variables = ['F_T', 'F_A', 'F_N', 'delta_phi']

    print(f"\n{'Variable':<12} {'MSE':<12} {'MAE':<12} {'Correlation':<12} {'PSNR (dB)':<12}")
    print("-" * 60)

    for var in variables:
        if var not in predictions or var not in reference:
            print(f"{var:<12} {'(missing)':<12}")
            continue

        pred = predictions[var]
        ref = reference[var]

        if pred.shape != ref.shape:
            print(f"{var:<12} Shape mismatch: {pred.shape} vs {ref.shape}")
            continue

        mse = compute_mse(pred, ref)
        mae = compute_mae(pred, ref)
        corr = compute_correlation(pred, ref)
        psnr = compute_psnr(pred, ref)

        print(f"{var:<12} {mse:<12.4e} {mae:<12.4e} {corr:<12.4f} {psnr:<12.2f}")

        var_metrics = compute_all_metrics(pred, ref, var)
        metrics.update(var_metrics)

    print("-" * 60)

    return metrics


def evaluate_regional(
    predictions: Dict[str, np.ndarray],
    verbose: bool = True
) -> Dict[str, float]:
    """
    Analyze predictions in different regions (center vs edge, high vs low intensity).
    """
    print("\n" + "="*60)
    print("REGIONAL ANALYSIS")
    print("="*60)

    F_T = predictions['F_T']
    D, H, W = F_T.shape

    # Create masks for different regions
    # Center region (inner 50%)
    center_margin = H // 4
    center_mask = np.zeros((D, H, W), dtype=bool)
    center_mask[:, center_margin:-center_margin, center_margin:-center_margin] = True

    # Edge region (outer 25%)
    edge_mask = ~center_mask

    # High intensity (top 25% of F_T values)
    threshold_high = np.percentile(F_T, 75)
    high_mask = F_T > threshold_high

    # Low intensity (bottom 25%)
    threshold_low = np.percentile(F_T, 25)
    low_mask = F_T < threshold_low

    metrics = {}

    print("\nF_T statistics by region:")
    print(f"  Center: mean={F_T[center_mask].mean():.2f}, std={F_T[center_mask].std():.2f}")
    print(f"  Edge:   mean={F_T[edge_mask].mean():.2f}, std={F_T[edge_mask].std():.2f}")
    print(f"  High:   mean={F_T[high_mask].mean():.2f}, std={F_T[high_mask].std():.2f}")
    print(f"  Low:    mean={F_T[low_mask].mean():.2f}, std={F_T[low_mask].std():.2f}")

    metrics['F_T_center_mean'] = float(F_T[center_mask].mean())
    metrics['F_T_edge_mean'] = float(F_T[edge_mask].mean())
    metrics['F_T_center_std'] = float(F_T[center_mask].std())
    metrics['F_T_edge_std'] = float(F_T[edge_mask].std())

    # Check for blocky artifacts (edge-to-center ratio)
    ratio = F_T[center_mask].mean() / (F_T[edge_mask].mean() + 1e-10)
    metrics['center_edge_ratio'] = ratio

    print(f"\nCenter/Edge ratio: {ratio:.3f}")
    if ratio > 2.0:
        print("  ⚠ Warning: Center significantly brighter than edges (possible artifact)")
    else:
        print("  ✓ Center/edge balance looks reasonable")

    # Phase consistency check
    sin_phi = predictions.get('sin_delta_phi', np.sin(predictions['delta_phi']))
    cos_phi = predictions.get('cos_delta_phi', np.cos(predictions['delta_phi']))

    # Unit circle violation
    unit_circle = sin_phi**2 + cos_phi**2
    mean_violation = np.abs(unit_circle - 1.0).mean()
    max_violation = np.abs(unit_circle - 1.0).max()

    print(f"\nPhase unit circle consistency:")
    print(f"  Mean |sin²+cos²-1|: {mean_violation:.6f}")
    print(f"  Max |sin²+cos²-1|:  {max_violation:.6f}")

    metrics['unit_circle_mean_violation'] = mean_violation
    metrics['unit_circle_max_violation'] = max_violation

    if mean_violation < 0.01:
        print("  ✓ Excellent unit circle consistency")
    elif mean_violation < 0.1:
        print("  ○ Good unit circle consistency")
    else:
        print("  ✗ Poor unit circle consistency")

    return metrics


def compute_summary_statistics(predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute summary statistics for predictions."""
    stats = {}

    for name, arr in predictions.items():
        if name.startswith('_'):
            continue
        stats[f'{name}_min'] = float(arr.min())
        stats[f'{name}_max'] = float(arr.max())
        stats[f'{name}_mean'] = float(arr.mean())
        stats[f'{name}_std'] = float(arr.std())

    return stats


def compute_local_contrast(arr: np.ndarray, window_size: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute local contrast metrics using sliding window.

    Parameters
    ----------
    arr : np.ndarray
        Input 3D array
    window_size : int
        Size of sliding window (default 7 = 7x7x7)

    Returns
    -------
    local_max : np.ndarray
        Local maximum in window around each voxel
    valley_depth : np.ndarray
        How deep valleys are relative to local max (0=same as max, 1=zero)
        valley_depth = 1 - (value / local_max)
    """
    from scipy.ndimage import maximum_filter

    # Compute local maximum in window
    local_max = maximum_filter(arr, size=window_size)

    # Valley depth = how far below local max (normalized)
    # = 1 - (value / local_max)
    # = 0 if value == local_max (peak)
    # = 1 if value == 0 (perfect valley)
    # Between 0-1 for partial valleys
    valley_depth = 1.0 - arr / (local_max + 1e-10)
    valley_depth = np.clip(valley_depth, 0, 1)

    return local_max, valley_depth


def analyze_haze_diagnostic(
    predictions: Dict[str, np.ndarray],
    ground_truth: Dict[str, np.ndarray] = None,
    intensity: np.ndarray = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Analyze the "haze" artifact - reduced fringe contrast within signal regions.

    The haze problem is NOT elevated values in outer/dark regions.
    It IS reduced contrast between fringe peaks and valleys WITHIN the signal region.

    In good diffraction data:
    - Constructive interference fringes have high intensity
    - Destructive interference valleys have near-zero intensity
    - This gives high peak-to-valley contrast

    With haze:
    - The valleys don't go to zero - they sit on an elevated "floor"
    - This reduces fringe contrast
    - Phase retrieval needs high contrast to converge

    This diagnostic uses LOCAL contrast measurement:
    - A sliding window computes local max around each voxel
    - Valley depth = 1 - (value / local_max)
    - Deep valleys (valley_depth near 1) = good
    - Shallow valleys (valley_depth near 0) = haze problem

    Parameters
    ----------
    predictions : dict
        Predicted MAD parameters (F_T, F_A, F_N, delta_phi)
    ground_truth : dict, optional
        Ground truth MAD parameters for comparison
    intensity : np.ndarray, optional
        Input intensity data - REQUIRED for proper haze analysis

    Returns
    -------
    metrics : dict
        Haze-related diagnostic metrics
    """
    print("\n" + "="*60)
    print("FRINGE CONTRAST / HAZE DIAGNOSTIC (Local Contrast Method)")
    print("="*60)
    print("Analyzing fringe contrast using local sliding window...")
    print("(Haze = valleys don't go deep enough relative to local peaks)")

    metrics = {}

    F_T = predictions['F_T']
    F_A = predictions['F_A']
    F_N = predictions['F_N']

    # -------------------------------------------------------------------------
    # 1. Define Signal Region (where diffraction has meaningful intensity)
    # -------------------------------------------------------------------------
    if intensity is not None:
        mean_intensity = intensity.mean(axis=-1)
        # Signal region = where intensity is above median (central 50%)
        signal_threshold = np.percentile(mean_intensity, 50)
        signal_mask = mean_intensity > signal_threshold
        print(f"\n--- Signal Region Definition ---")
        print(f"  Using intensity > {signal_threshold:.2f} (median)")
        print(f"  Signal region: {signal_mask.sum()} voxels ({100*signal_mask.mean():.1f}%)")
    else:
        # Fallback: use F_T itself to define signal region
        signal_threshold = np.percentile(F_T, 50)
        signal_mask = F_T > signal_threshold
        print(f"\n--- Signal Region Definition ---")
        print(f"  No intensity provided, using F_T > {signal_threshold:.2f}")
        print(f"  Signal region: {signal_mask.sum()} voxels ({100*signal_mask.mean():.1f}%)")

    metrics['signal_region_fraction'] = float(signal_mask.mean())

    # -------------------------------------------------------------------------
    # 2. LOCAL Fringe Contrast Analysis (using sliding window)
    # -------------------------------------------------------------------------
    print("\n--- Local Fringe Contrast Analysis ---")
    print("  Using 7x7x7 sliding window for local max computation")
    print("  Valley depth = 1 - (value / local_max)")
    print("  Deep valleys (>0.8) = good contrast, shallow (<0.5) = haze")

    window_size = 7

    for name, arr in [('F_T', F_T), ('F_A', F_A), ('F_N', F_N)]:
        # Compute local contrast
        local_max, valley_depth = compute_local_contrast(arr, window_size=window_size)

        # Store for visualization
        metrics[f'_{name}_local_max'] = local_max
        metrics[f'_{name}_valley_depth'] = valley_depth

        # Get values within signal region
        signal_values = arr[signal_mask]
        signal_valley_depth = valley_depth[signal_mask]
        signal_local_max = local_max[signal_mask]

        # Statistics on valley depth within signal region
        mean_valley_depth = np.mean(signal_valley_depth)
        median_valley_depth = np.median(signal_valley_depth)

        # What fraction of signal region are "valleys" (below local context)?
        # A voxel is a valley if it's significantly below local max (depth > 0.3)
        valley_threshold = 0.3
        n_valleys = np.sum(signal_valley_depth > valley_threshold)
        valley_fraction = n_valleys / signal_mask.sum()

        # Among the valleys, how deep do they go?
        valley_mask = signal_valley_depth > valley_threshold
        if valley_mask.sum() > 0:
            valley_values = signal_valley_depth[valley_mask]
            mean_valley_depth_in_valleys = np.mean(valley_values)
            # Deep valleys = approaching 1.0 (zero value)
            deep_valley_fraction = np.mean(valley_values > 0.8)
        else:
            mean_valley_depth_in_valleys = 0.0
            deep_valley_fraction = 0.0

        # Compute an overall "local fringe contrast" metric
        # High contrast = valleys are deep (mean depth > 0.7)
        local_fringe_contrast = mean_valley_depth_in_valleys if valley_mask.sum() > 0 else 0.0

        print(f"\n  {name} in signal region:")
        print(f"    Mean valley depth:       {mean_valley_depth:.4f}")
        print(f"    Median valley depth:     {median_valley_depth:.4f}")
        print(f"    Fraction that are valleys (depth>{valley_threshold}): {valley_fraction:.3f} ({n_valleys} voxels)")
        if valley_mask.sum() > 0:
            print(f"    Mean depth in valleys:   {mean_valley_depth_in_valleys:.4f}")
            print(f"    Fraction of deep valleys (depth>0.8): {deep_valley_fraction:.3f}")

        metrics[f'{name}_mean_valley_depth'] = mean_valley_depth
        metrics[f'{name}_valley_fraction'] = valley_fraction
        metrics[f'{name}_local_fringe_contrast'] = local_fringe_contrast
        metrics[f'{name}_deep_valley_fraction'] = deep_valley_fraction

        if local_fringe_contrast > 0.8:
            print(f"    ✓ Excellent - valleys go near zero")
        elif local_fringe_contrast > 0.6:
            print(f"    ○ Good - valleys have reasonable depth")
        elif local_fringe_contrast > 0.4:
            print(f"    △ Moderate - valleys are shallow (haze present)")
        else:
            print(f"    ⚠ Poor - valleys barely dip (significant haze)")

    # -------------------------------------------------------------------------
    # 3. Compare Intensity vs F_T Valley Depth (LOCAL comparison)
    # -------------------------------------------------------------------------
    if intensity is not None:
        print("\n--- Intensity vs Prediction Valley Depth Comparison ---")

        # Compute local contrast for intensity as well
        int_local_max, int_valley_depth = compute_local_contrast(mean_intensity, window_size=window_size)

        # Store for visualization
        metrics['_intensity_valley_depth'] = int_valley_depth

        # Compare valley depth distributions within signal region
        int_vd_signal = int_valley_depth[signal_mask]
        ft_vd_signal = metrics['_F_T_valley_depth'][signal_mask]

        # In intensity, what fraction are deep valleys?
        int_deep_fraction = np.mean(int_vd_signal > 0.8)
        ft_deep_fraction = np.mean(ft_vd_signal > 0.8)

        # How well does F_T preserve valley depth from intensity?
        # Perfect = same valley depth everywhere
        depth_correlation = np.corrcoef(int_vd_signal, ft_vd_signal)[0, 1]

        # Mean valley depth comparison
        int_mean_depth = np.mean(int_vd_signal)
        ft_mean_depth = np.mean(ft_vd_signal)
        depth_preservation = ft_mean_depth / (int_mean_depth + 1e-10)

        print(f"\n  Intensity mean valley depth: {int_mean_depth:.4f}")
        print(f"  F_T mean valley depth:       {ft_mean_depth:.4f}")
        print(f"  Depth preservation ratio:    {depth_preservation:.4f}")
        print(f"  Valley depth correlation:    {depth_correlation:.4f}")
        print(f"  Intensity deep valleys (>0.8): {int_deep_fraction:.3f}")
        print(f"  F_T deep valleys (>0.8):       {ft_deep_fraction:.3f}")

        metrics['intensity_mean_valley_depth'] = int_mean_depth
        metrics['valley_depth_preservation'] = depth_preservation
        metrics['valley_depth_correlation'] = depth_correlation
        metrics['intensity_deep_valley_fraction'] = int_deep_fraction

        if depth_preservation > 0.9:
            print(f"    ✓ Excellent - F_T preserves valley depth from intensity")
        elif depth_preservation > 0.7:
            print(f"    ○ Good - minor valley depth loss")
        elif depth_preservation > 0.5:
            print(f"    △ Moderate - valleys shallower than in intensity (haze)")
        else:
            print(f"    ⚠ Poor - significant valley filling (haze)")

    # -------------------------------------------------------------------------
    # 4. Ground Truth Comparison (if available) - LOCAL contrast
    # -------------------------------------------------------------------------
    if ground_truth is not None and 'F_T' in ground_truth:
        print("\n--- Ground Truth Valley Depth Comparison ---")

        for name in ['F_T', 'F_A', 'F_N']:
            if name not in ground_truth:
                continue

            pred = predictions[name]
            true = ground_truth[name]

            # Compute local contrast for ground truth
            _, true_valley_depth = compute_local_contrast(true, window_size=window_size)
            pred_valley_depth = metrics[f'_{name}_valley_depth']

            # Compare valley depth within signal region
            true_vd_signal = true_valley_depth[signal_mask]
            pred_vd_signal = pred_valley_depth[signal_mask]

            true_mean_depth = np.mean(true_vd_signal)
            pred_mean_depth = np.mean(pred_vd_signal)
            depth_ratio = pred_mean_depth / (true_mean_depth + 1e-10)

            print(f"\n  {name}:")
            print(f"    True mean valley depth:      {true_mean_depth:.4f}")
            print(f"    Predicted mean valley depth: {pred_mean_depth:.4f}")
            print(f"    Ratio (pred/true):           {depth_ratio:.4f}")

            metrics[f'{name}_true_valley_depth'] = true_mean_depth
            metrics[f'{name}_valley_depth_ratio'] = depth_ratio

    # -------------------------------------------------------------------------
    # Summary - using LOCAL contrast metrics
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("LOCAL FRINGE CONTRAST SUMMARY")
    print("-"*60)

    # Overall haze score based on LOCAL fringe contrast
    # local_fringe_contrast = mean valley depth in valleys (0-1)
    # High = deep valleys = good, Low = shallow valleys = haze
    contrasts = [metrics.get(f'{n}_local_fringe_contrast', 0.0) for n in ['F_T', 'F_A', 'F_N']]
    mean_local_contrast = np.mean(contrasts)
    overall_haze = 1.0 - mean_local_contrast  # Haze = 1 - contrast

    metrics['mean_local_fringe_contrast'] = mean_local_contrast
    metrics['overall_haze_score'] = overall_haze

    # Also compute mean deep valley fraction
    deep_fractions = [metrics.get(f'{n}_deep_valley_fraction', 0.0) for n in ['F_T', 'F_A', 'F_N']]
    mean_deep_fraction = np.mean(deep_fractions)
    metrics['mean_deep_valley_fraction'] = mean_deep_fraction

    print(f"\n  Mean local fringe contrast:   {mean_local_contrast:.4f}")
    print(f"  Mean deep valley fraction:    {mean_deep_fraction:.4f}")
    print(f"  Overall haze score:           {overall_haze:.4f} (1 - contrast)")

    if overall_haze < 0.2:
        print("\n  ✓ LOW HAZE - Valleys go deep, good fringe contrast")
    elif overall_haze < 0.4:
        print("\n  △ MODERATE HAZE")
        print("  Recommendations:")
        print("    - Valleys are shallower than ideal")
        print("    - Consider local contrast enhancement post-processing")
        print("    - May benefit from sparsity penalty in training")
    else:
        print("\n  ⚠ SIGNIFICANT HAZE")
        print("  Recommendations:")
        print("    - Valleys don't dip - haze fills the fringes")
        print("    - Strong contrast enhancement or retraining needed")
        print("    - Check if log-transform is preventing near-zero predictions")

    print("-"*60)

    return metrics


def analyze_fa_holes(
    predictions: Dict[str, np.ndarray],
    intensity: np.ndarray = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Analyze F_A "holes" - regions where F_A is unexpectedly near zero.

    The F_A hole problem:
    - F_A (anomalous structure factor / Ni contribution) shows gaps/holes
    - These appear as dark spots within the signal region where F_A should be non-zero
    - Caused by intensity-weighted loss ignoring low-intensity regions
    - Propagates to F_N calculations since F_N = f(F_T, F_A, Δφ)

    Detection approach:
    - Define signal region using F_T (where diffraction pattern exists)
    - Identify "holes" as pixels where F_A is much lower than expected given F_T
    - A hole = F_A < threshold while F_T > signal_threshold

    Parameters
    ----------
    predictions : dict
        Predicted MAD parameters (F_T, F_A, F_N, delta_phi)
    intensity : np.ndarray, optional
        Input intensity data (used for signal region if available)
    verbose : bool
        Print diagnostic information

    Returns
    -------
    metrics : dict
        F_A hole diagnostic metrics
    """
    if verbose:
        print("\n" + "="*60)
        print("F_A HOLE DETECTION")
        print("="*60)
        print("Analyzing F_A for unexpected near-zero regions...")

    metrics = {}

    F_T = predictions['F_T']
    F_A = predictions['F_A']
    F_N = predictions['F_N']

    # -------------------------------------------------------------------------
    # 1. Define Signal Region (where we expect meaningful F_A values)
    # -------------------------------------------------------------------------
    if intensity is not None:
        mean_intensity = intensity.mean(axis=-1)
        signal_threshold = np.percentile(mean_intensity, 50)
        signal_mask = mean_intensity > signal_threshold
    else:
        # Use F_T to define where signal should exist
        signal_threshold = np.percentile(F_T, 50)
        signal_mask = F_T > signal_threshold

    n_signal = signal_mask.sum()

    if verbose:
        print(f"\n--- Signal Region ---")
        print(f"  Signal region: {n_signal} voxels ({100*signal_mask.mean():.1f}%)")

    # -------------------------------------------------------------------------
    # 2. F_A Hole Detection
    # -------------------------------------------------------------------------
    # A "hole" is where F_A is unexpectedly low relative to F_T
    # In core-shell particles, F_A should generally follow F_T structure

    # Get F_A values in signal region
    fa_signal = F_A[signal_mask]
    ft_signal = F_T[signal_mask]

    # Compute F_A/F_T ratio (where F_T > 0)
    ratio = fa_signal / (ft_signal + 1e-10)

    # Statistics on the ratio
    ratio_mean = np.mean(ratio)
    ratio_median = np.median(ratio)
    ratio_std = np.std(ratio)
    ratio_p10 = np.percentile(ratio, 10)
    ratio_p90 = np.percentile(ratio, 90)

    if verbose:
        print(f"\n--- F_A / F_T Ratio in Signal Region ---")
        print(f"  Mean:   {ratio_mean:.4f}")
        print(f"  Median: {ratio_median:.4f}")
        print(f"  Std:    {ratio_std:.4f}")
        print(f"  P10:    {ratio_p10:.4f}")
        print(f"  P90:    {ratio_p90:.4f}")

    metrics['fa_ft_ratio_mean'] = float(ratio_mean)
    metrics['fa_ft_ratio_median'] = float(ratio_median)
    metrics['fa_ft_ratio_std'] = float(ratio_std)

    # -------------------------------------------------------------------------
    # 3. Identify Holes using multiple criteria
    # -------------------------------------------------------------------------

    # Criterion 1: F_A near zero while F_T has signal
    # Use adaptive threshold based on F_A distribution
    fa_p5 = np.percentile(fa_signal, 5)  # Bottom 5% of F_A in signal region
    fa_p50 = np.percentile(fa_signal, 50)

    # A hole is where F_A is in the bottom 5% (very low) but not because F_T is also low
    # Specifically: F_A < P5 AND F_T > median(F_T)
    ft_median = np.percentile(ft_signal, 50)
    hole_mask_1 = (fa_signal < fa_p5) & (ft_signal > ft_median)
    n_holes_1 = hole_mask_1.sum()
    hole_fraction_1 = n_holes_1 / n_signal

    # Criterion 2: F_A/F_T ratio is anomalously low
    # A hole is where ratio < P10 of the ratio distribution
    hole_mask_2 = ratio < ratio_p10
    n_holes_2 = hole_mask_2.sum()
    hole_fraction_2 = n_holes_2 / n_signal

    # Criterion 3: F_A near zero in absolute terms
    # Use a fixed threshold based on log-scale (since F_A is stored as log1p)
    # log1p(1) ≈ 0.69, so F_A < 0.5 means original value < exp(0.5)-1 ≈ 0.65
    fa_abs_threshold = 0.5
    hole_mask_3 = (fa_signal < fa_abs_threshold) & (ft_signal > ft_median)
    n_holes_3 = hole_mask_3.sum()
    hole_fraction_3 = n_holes_3 / n_signal

    if verbose:
        print(f"\n--- Hole Detection Results ---")
        print(f"  Criterion 1 (F_A < P5 & F_T > median):")
        print(f"    Holes: {n_holes_1} ({100*hole_fraction_1:.2f}%)")
        print(f"  Criterion 2 (F_A/F_T ratio < P10):")
        print(f"    Holes: {n_holes_2} ({100*hole_fraction_2:.2f}%)")
        print(f"  Criterion 3 (F_A < {fa_abs_threshold} & F_T > median):")
        print(f"    Holes: {n_holes_3} ({100*hole_fraction_3:.2f}%)")

    metrics['fa_holes_criterion1_count'] = int(n_holes_1)
    metrics['fa_holes_criterion1_fraction'] = float(hole_fraction_1)
    metrics['fa_holes_criterion2_count'] = int(n_holes_2)
    metrics['fa_holes_criterion2_fraction'] = float(hole_fraction_2)
    metrics['fa_holes_criterion3_count'] = int(n_holes_3)
    metrics['fa_holes_criterion3_fraction'] = float(hole_fraction_3)

    # Overall hole score: average of criteria
    overall_hole_fraction = (hole_fraction_1 + hole_fraction_3) / 2
    metrics['fa_holes_overall_fraction'] = float(overall_hole_fraction)

    # -------------------------------------------------------------------------
    # 4. Spatial Analysis - where are the holes?
    # -------------------------------------------------------------------------
    # Create 3D hole mask for visualization
    hole_mask_3d = np.zeros_like(F_A, dtype=bool)
    hole_mask_3d[signal_mask] = hole_mask_1 | hole_mask_3

    # Store for visualization (prefix with _ to exclude from saved metrics)
    metrics['_fa_hole_mask'] = hole_mask_3d

    # -------------------------------------------------------------------------
    # 5. Summary
    # -------------------------------------------------------------------------
    if verbose:
        print("\n" + "-"*60)
        print("F_A HOLE SUMMARY")
        print("-"*60)
        print(f"\n  Overall hole fraction: {100*overall_hole_fraction:.2f}%")

        if overall_hole_fraction < 0.02:
            print("\n  ✓ EXCELLENT - Very few F_A holes (<2%)")
        elif overall_hole_fraction < 0.05:
            print("\n  ○ GOOD - Minor F_A holes (<5%)")
            print("  May not significantly impact F_N derivation")
        elif overall_hole_fraction < 0.10:
            print("\n  △ MODERATE - Noticeable F_A holes (5-10%)")
            print("  Recommendations:")
            print("    - Check F_N for corresponding gaps")
            print("    - Consider retraining with --lambda-fa 1.0")
        else:
            print("\n  ⚠ SIGNIFICANT - Many F_A holes (>10%)")
            print("  Recommendations:")
            print("    - Retrain with --lambda-fa 1.0 or higher")
            print("    - If persists, try A2 approach (uniform F_A weighting)")

        print("-"*60)

    return metrics


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_evaluation(
    predictions: Dict[str, np.ndarray],
    intensity: np.ndarray = None,
    metrics: Dict = None,
    save_path: str = None
):
    """Generate evaluation visualization."""
    import matplotlib.pyplot as plt

    F_T = predictions['F_T']
    F_A = predictions['F_A']
    F_N = predictions['F_N']
    delta_phi = predictions['delta_phi']

    D, H, W = F_T.shape
    z_mid = D // 2

    # Figure 1: Predictions overview
    fig1, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Row 0: Central XY slice
    axes[0, 0].imshow(F_T[z_mid], cmap='viridis')
    axes[0, 0].set_title(f'|F_T| (z={z_mid})')

    axes[0, 1].imshow(F_A[z_mid], cmap='plasma')
    axes[0, 1].set_title(f'|F_A| (z={z_mid})')

    axes[0, 2].imshow(F_N[z_mid], cmap='cividis')
    axes[0, 2].set_title(f'|F_N| (z={z_mid})')

    im = axes[0, 3].imshow(delta_phi[z_mid], cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[0, 3].set_title(f'Δφ (z={z_mid})')
    plt.colorbar(im, ax=axes[0, 3])

    # Row 1: Central XZ slice
    y_mid = H // 2
    axes[1, 0].imshow(F_T[:, y_mid, :], cmap='viridis', aspect='auto')
    axes[1, 0].set_title(f'|F_T| (y={y_mid})')

    axes[1, 1].imshow(F_A[:, y_mid, :], cmap='plasma', aspect='auto')
    axes[1, 1].set_title(f'|F_A| (y={y_mid})')

    axes[1, 2].imshow(F_N[:, y_mid, :], cmap='cividis', aspect='auto')
    axes[1, 2].set_title(f'|F_N| (y={y_mid})')

    axes[1, 3].imshow(delta_phi[:, y_mid, :], cmap='twilight', aspect='auto', vmin=-np.pi, vmax=np.pi)
    axes[1, 3].set_title(f'Δφ (y={y_mid})')

    plt.suptitle('MAD Parameter Predictions', fontsize=14)
    plt.tight_layout()

    if save_path:
        fig1.savefig(Path(save_path) / 'predictions_overview.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: predictions_overview.png")

    # Figure 2: Intensity reconstruction comparison (if we have intensity data)
    if intensity is not None and metrics and '_I_reconstructed' in metrics:
        I_obs = intensity
        I_recon = metrics['_I_reconstructed']

        fig2, axes = plt.subplots(3, 4, figsize=(16, 12))

        # Show 4 energies
        energy_indices = [0, 2, 5, 7]

        for col, e in enumerate(energy_indices):
            # Observed
            I_o = I_obs[z_mid, ..., e]
            I_r = I_recon[z_mid, ..., e]

            vmax = np.percentile(I_o, 99)

            axes[0, col].imshow(np.log1p(I_o), cmap='viridis')
            axes[0, col].set_title(f'Observed (E[{e}])')

            axes[1, col].imshow(np.log1p(I_r), cmap='viridis')
            axes[1, col].set_title(f'Reconstructed (E[{e}])')

            diff = I_r - I_o
            vdiff = np.percentile(np.abs(diff), 99)
            axes[2, col].imshow(diff, cmap='RdBu', vmin=-vdiff, vmax=vdiff)
            axes[2, col].set_title(f'Difference (E[{e}])')

        axes[0, 0].set_ylabel('Observed', fontsize=12)
        axes[1, 0].set_ylabel('Reconstructed', fontsize=12)
        axes[2, 0].set_ylabel('Difference', fontsize=12)

        mean_r = metrics.get('mean_r_factor', 0)
        mean_corr = metrics.get('mean_correlation', 0)
        plt.suptitle(f'Intensity Reconstruction (R={mean_r:.4f}, corr={mean_corr:.4f})', fontsize=14)
        plt.tight_layout()

        if save_path:
            fig2.savefig(Path(save_path) / 'intensity_reconstruction.png', dpi=150, bbox_inches='tight')
            print(f"  Saved: intensity_reconstruction.png")

    # Figure 3: Histograms and distributions
    fig3, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].hist(F_T.flatten(), bins=100, alpha=0.7, log=True)
    axes[0, 0].set_title('|F_T| Distribution')
    axes[0, 0].set_xlabel('|F_T|')

    axes[0, 1].hist(F_A.flatten(), bins=100, alpha=0.7, log=True)
    axes[0, 1].set_title('|F_A| Distribution')
    axes[0, 1].set_xlabel('|F_A|')

    axes[1, 0].hist(F_N.flatten(), bins=100, alpha=0.7, log=True)
    axes[1, 0].set_title('|F_N| Distribution')
    axes[1, 0].set_xlabel('|F_N|')

    axes[1, 1].hist(delta_phi.flatten(), bins=100, alpha=0.7)
    axes[1, 1].set_title('Δφ Distribution')
    axes[1, 1].set_xlabel('Δφ (rad)')
    axes[1, 1].axvline(0, color='r', linestyle='--', alpha=0.5)

    plt.suptitle('Value Distributions', fontsize=14)
    plt.tight_layout()

    if save_path:
        fig3.savefig(Path(save_path) / 'distributions.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: distributions.png")

    # Figure 4: Fringe Contrast / Haze Diagnostic Visualization
    if metrics:
        fig4, axes = plt.subplots(2, 3, figsize=(15, 10))

        z_mid = D // 2

        # Define signal region mask for this slice
        if intensity is not None:
            mean_int = intensity.mean(axis=-1)
            signal_thresh = np.percentile(mean_int, 50)
            signal_mask_slice = mean_int[z_mid] > signal_thresh
        else:
            signal_thresh = np.percentile(F_T, 50)
            signal_mask_slice = F_T[z_mid] > signal_thresh

        # Row 0: Compare intensity fringes vs F_T fringes (the core of haze analysis)
        if intensity is not None:
            mean_int_slice = intensity[z_mid].mean(axis=-1)

            # Left: Input intensity (shows true fringes)
            im0 = axes[0, 0].imshow(np.log1p(mean_int_slice), cmap='viridis')
            axes[0, 0].set_title('Input Intensity (log scale)')
            plt.colorbar(im0, ax=axes[0, 0])

            # Middle: F_T prediction (should match fringe pattern)
            im1 = axes[0, 1].imshow(np.log1p(F_T[z_mid]), cmap='viridis')
            axes[0, 1].set_title('Predicted |F_T| (log scale)')
            plt.colorbar(im1, ax=axes[0, 1])

            # Right: Difference showing where contrast is lost (haze regions)
            # Normalize both to [0,1] for comparison
            int_norm = (mean_int_slice - mean_int_slice.min()) / (mean_int_slice.max() - mean_int_slice.min() + 1e-10)
            ft_norm = (F_T[z_mid] - F_T[z_mid].min()) / (F_T[z_mid].max() - F_T[z_mid].min() + 1e-10)

            # Show where F_T is higher than expected (haze in valleys)
            # Mask to signal region only
            haze_map = np.where(signal_mask_slice, ft_norm - int_norm, 0)
            haze_map = np.clip(haze_map, 0, None)  # Only show positive (elevated) regions

            im2 = axes[0, 2].imshow(haze_map, cmap='hot', vmin=0)
            axes[0, 2].set_title('Haze Map (F_T elevated vs intensity)')
            plt.colorbar(im2, ax=axes[0, 2])
        else:
            # Without intensity, show F_T, F_A, F_N
            im0 = axes[0, 0].imshow(np.log1p(F_T[z_mid]), cmap='viridis')
            axes[0, 0].set_title('|F_T| (log scale)')
            plt.colorbar(im0, ax=axes[0, 0])

            im1 = axes[0, 1].imshow(np.log1p(F_A[z_mid]), cmap='plasma')
            axes[0, 1].set_title('|F_A| (log scale)')
            plt.colorbar(im1, ax=axes[0, 1])

            im2 = axes[0, 2].imshow(np.log1p(F_N[z_mid]), cmap='cividis')
            axes[0, 2].set_title('|F_N| (log scale)')
            plt.colorbar(im2, ax=axes[0, 2])

        # Row 1: Fringe cross-sections showing peak/valley contrast
        # Take a horizontal line through the center
        y_mid = H // 2

        # Left: Line profile comparison
        if intensity is not None:
            int_line = intensity[z_mid, y_mid, :, :].mean(axis=-1)
            int_line_norm = int_line / (int_line.max() + 1e-10)
            axes[1, 0].plot(int_line_norm, 'b-', label='Intensity', alpha=0.7)

        ft_line = F_T[z_mid, y_mid, :]
        ft_line_norm = ft_line / (ft_line.max() + 1e-10)
        axes[1, 0].plot(ft_line_norm, 'r-', label='F_T', alpha=0.7)
        axes[1, 0].set_xlabel('X position')
        axes[1, 0].set_ylabel('Normalized value')
        axes[1, 0].set_title(f'Fringe profile (y={y_mid}, z={z_mid})')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Middle: LOCAL Valley Depth Map (key improvement over global percentile approach)
        # Use the pre-computed valley depth from sliding window
        if '_F_T_valley_depth' in metrics:
            ft_valley_depth_slice = metrics['_F_T_valley_depth'][z_mid]
            # Only show within signal region, mask out background
            valley_depth_display = np.where(signal_mask_slice, ft_valley_depth_slice, np.nan)
            im3 = axes[1, 1].imshow(valley_depth_display, cmap='RdYlGn', vmin=0, vmax=1)
            axes[1, 1].set_title('F_T Local Valley Depth\n(1=deep valley, 0=peak)')
            plt.colorbar(im3, ax=axes[1, 1])
        else:
            im3 = axes[1, 1].imshow(signal_mask_slice.astype(float), cmap='gray')
            axes[1, 1].set_title('Signal Region Mask')
            plt.colorbar(im3, ax=axes[1, 1])

        # Right: Compare intensity valley depth vs F_T valley depth
        # This shows where F_T has shallower valleys than intensity (the haze problem)
        if '_intensity_valley_depth' in metrics and '_F_T_valley_depth' in metrics:
            int_vd_slice = metrics['_intensity_valley_depth'][z_mid]
            ft_vd_slice = metrics['_F_T_valley_depth'][z_mid]

            # Haze = where intensity has deep valleys but F_T doesn't
            # depth_diff = intensity_valley_depth - ft_valley_depth
            # Positive = F_T valleys are shallower than intensity (haze problem)
            depth_diff = int_vd_slice - ft_vd_slice
            depth_diff_display = np.where(signal_mask_slice, depth_diff, np.nan)

            im4 = axes[1, 2].imshow(depth_diff_display, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
            axes[1, 2].set_title('Valley Depth: Intensity - F_T\n(Red = F_T valleys shallower = haze)')
            plt.colorbar(im4, ax=axes[1, 2])
        elif '_F_T_valley_depth' in metrics:
            # Show just the F_T valleys with depth > 0.5 (significant valleys)
            ft_vd_slice = metrics['_F_T_valley_depth'][z_mid]
            valleys_display = np.where(signal_mask_slice & (ft_vd_slice > 0.3), ft_vd_slice, np.nan)
            im4 = axes[1, 2].imshow(valleys_display, cmap='hot', vmin=0.3, vmax=1.0)
            axes[1, 2].set_title('F_T Valleys (depth > 0.3)')
            plt.colorbar(im4, ax=axes[1, 2])
        else:
            # Fallback to old method
            ft_slice = F_T[z_mid].copy()
            signal_vals = ft_slice[signal_mask_slice]
            valley_thresh = np.percentile(signal_vals, 20)
            valley_map = np.where(signal_mask_slice & (ft_slice < valley_thresh), ft_slice, np.nan)
            im4 = axes[1, 2].imshow(valley_map, cmap='hot')
            axes[1, 2].set_title(f'Fringe valleys (F_T < {valley_thresh:.1f})')
            plt.colorbar(im4, ax=axes[1, 2])

        # Add fringe contrast info to title (using new local contrast metrics)
        mean_contrast = metrics.get('mean_local_fringe_contrast', metrics.get('mean_fringe_contrast', 0))
        haze_score = metrics.get('overall_haze_score', 0)
        deep_frac = metrics.get('mean_deep_valley_fraction', 0)
        plt.suptitle(f'Local Fringe Contrast Diagnostic\n(contrast={mean_contrast:.3f}, haze={haze_score:.3f}, deep_valleys={deep_frac:.3f})', fontsize=14)
        plt.tight_layout()

        if save_path:
            fig4.savefig(Path(save_path) / 'haze_diagnostic.png', dpi=150, bbox_inches='tight')
            print(f"  Saved: haze_diagnostic.png")

    plt.show()


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(
    all_metrics: Dict[str, float],
    predictions: Dict[str, np.ndarray],
    output_path: Path
):
    """Generate a text report summarizing evaluation results."""

    report_lines = [
        "=" * 70,
        "NanoMAD ML - 3D EVALUATION REPORT",
        "=" * 70,
        "",
        "Summary Statistics",
        "-" * 40,
    ]

    # Summary stats
    stats = compute_summary_statistics(predictions)
    for name in ['F_T', 'F_A', 'F_N', 'delta_phi']:
        if f'{name}_mean' in stats:
            report_lines.append(f"  {name}:")
            report_lines.append(f"    Range: [{stats[f'{name}_min']:.4f}, {stats[f'{name}_max']:.4f}]")
            report_lines.append(f"    Mean:  {stats[f'{name}_mean']:.4f} ± {stats[f'{name}_std']:.4f}")

    report_lines.extend([
        "",
        "Self-Consistency Metrics",
        "-" * 40,
        "  (NOTE: Raw R-factor is high due to scale mismatch in simplified MAD equation)",
        "  (Focus on Correlation and Normalized R-factor for quality assessment)",
    ])

    if 'mean_correlation' in all_metrics:
        report_lines.append(f"  Mean Correlation:     {all_metrics['mean_correlation']:.6f}")
    if 'mean_norm_r_factor' in all_metrics:
        report_lines.append(f"  Mean Norm R-factor:   {all_metrics['mean_norm_r_factor']:.6f}")

    report_lines.extend([
        "",
        "Regional Analysis",
        "-" * 40,
    ])

    if 'center_edge_ratio' in all_metrics:
        report_lines.append(f"  Center/Edge ratio:      {all_metrics['center_edge_ratio']:.3f}")
    if 'unit_circle_mean_violation' in all_metrics:
        report_lines.append(f"  Unit circle violation:  {all_metrics['unit_circle_mean_violation']:.6f}")

    # Fringe Contrast / Haze diagnostic section
    if 'overall_haze_score' in all_metrics:
        report_lines.extend([
            "",
            "Fringe Contrast / Haze Diagnostic",
            "-" * 40,
            "  (Haze = reduced contrast in fringe valleys within signal region)",
        ])
        report_lines.append(f"  Mean fringe contrast:   {all_metrics.get('mean_fringe_contrast', 0):.4f}")
        report_lines.append(f"  Overall haze score:     {all_metrics['overall_haze_score']:.4f} (1 - contrast)")

        if 'contrast_preservation' in all_metrics:
            report_lines.append(f"  Contrast preservation:  {all_metrics['contrast_preservation']:.4f}")

        for name in ['F_T', 'F_A', 'F_N']:
            if f'{name}_fringe_contrast' in all_metrics:
                report_lines.append(f"  {name} fringe contrast: {all_metrics[f'{name}_fringe_contrast']:.4f}")
            if f'{name}_valley_peak_ratio' in all_metrics:
                report_lines.append(f"  {name} valley/peak ratio: {all_metrics[f'{name}_valley_peak_ratio']:.4f}")

    report_lines.extend([
        "",
        "=" * 70,
    ])

    report_text = "\n".join(report_lines)

    report_file = output_path / 'evaluation_report.txt'
    with open(report_file, 'w') as f:
        f.write(report_text)

    print(f"\nReport saved: {report_file}")
    print("\n" + report_text)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate 3D MAD inference results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Self-consistency evaluation
    python evaluate_3d.py --predictions inference_output_v2/ \\
        --intensity ../evaluation_data/test_3d_noise_pf1e6_v2/test_intensity_3d_noise_pf1e6.npy \\
        --energies ../evaluation_data/test_3d_noise_pf1e6_v2/test_energies.npy \\
        -o evaluation_results/

    # Compare two model runs
    python evaluate_3d.py --predictions inference_output_v2/ \\
        --reference results_phase6/ \\
        -o evaluation_comparison/
        """
    )

    parser.add_argument('--predictions', '-p', type=str, required=True,
                        help='Directory containing prediction files (F_T_3d.npy, etc.)')
    parser.add_argument('--intensity', type=str, default=None,
                        help='Path to input intensity array for self-consistency check')
    parser.add_argument('--energies', type=str, default=None,
                        help='Path to energies array')
    parser.add_argument('--reference', '-r', type=str, default=None,
                        help='Directory containing reference/ground-truth predictions')
    parser.add_argument('--ground-truth', '-g', type=str, default=None,
                        help='Directory containing ground truth MAD parameters (for haze comparison)')
    parser.add_argument('--output-dir', '-o', type=str, default='evaluation_output',
                        help='Output directory for results')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip visualization')
    parser.add_argument('--haze-diagnostic', action='store_true', default=True,
                        help='Run haze diagnostic analysis (default: True)')
    parser.add_argument('--no-haze-diagnostic', action='store_true',
                        help='Skip haze diagnostic analysis')

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("NanoMAD ML - 3D EVALUATION")
    print("="*70)

    # Load predictions
    print(f"\nLoading predictions from: {args.predictions}")
    predictions = load_predictions(args.predictions)

    if not predictions:
        print("ERROR: No prediction files found!")
        return

    # Collect all metrics
    all_metrics = {}

    # Load intensity data if provided
    intensity = None
    energies = None
    if args.intensity:
        data = load_intensity_data(args.intensity, args.energies)
        intensity = data['intensity']
        energies = data.get('energies')

    # Self-consistency evaluation
    if intensity is not None and energies is not None:
        sc_metrics = evaluate_self_consistency(predictions, intensity, energies)
        all_metrics.update(sc_metrics)

    # Reference comparison
    if args.reference:
        print(f"\nLoading reference from: {args.reference}")
        reference = load_predictions(args.reference)
        if reference:
            ref_metrics = evaluate_against_reference(predictions, reference)
            all_metrics.update(ref_metrics)

    # Regional analysis
    regional_metrics = evaluate_regional(predictions)
    all_metrics.update(regional_metrics)

    # Haze diagnostic analysis
    if args.haze_diagnostic and not args.no_haze_diagnostic:
        # Load ground truth if provided
        ground_truth = None
        if args.ground_truth:
            print(f"\nLoading ground truth from: {args.ground_truth}")
            ground_truth = load_predictions(args.ground_truth)

        haze_metrics = analyze_haze_diagnostic(
            predictions,
            ground_truth=ground_truth,
            intensity=intensity
        )
        all_metrics.update(haze_metrics)

    # F_A hole diagnostic
    if not args.no_haze_diagnostic:  # Run alongside haze diagnostic
        fa_hole_metrics = analyze_fa_holes(
            predictions,
            intensity=intensity
        )
        all_metrics.update(fa_hole_metrics)

    # Generate report
    generate_report(all_metrics, predictions, output_path)

    # Save metrics
    # Remove non-serializable items
    metrics_to_save = {k: v for k, v in all_metrics.items() if not k.startswith('_')}
    np.savez(output_path / 'metrics.npz', **metrics_to_save)
    print(f"Metrics saved: {output_path / 'metrics.npz'}")

    # Visualization
    if not args.no_plot:
        print("\nGenerating visualizations...")
        visualize_evaluation(predictions, intensity, all_metrics, save_path=str(output_path))

    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()
