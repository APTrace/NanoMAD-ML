#!/usr/bin/env python3
"""
validate_ground_truth.py

Validate that our computed ground truth labels (F_T, F_A, Δφ) are correct
by reconstructing intensities using the MAD equation and comparing to
our simulated diffraction patterns.

The MAD equation:
    I(Q,E) = |F_T|² + (f'²+f''²)|F_A/f₀|² + 2|F_T||F_A/f₀|·[f'·cos(Δφ)+f''·sin(Δφ)]

If our ground truth is correct, the reconstructed intensity should match
our simulated intensity (within numerical precision).

This is the validation the NanoMAD expert recommended!

Author: Claude (Anthropic) + Thomas
Date: December 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

# Import from core_shell - use try/except for flexible import paths
try:
    # When running from repository root: python src/validate.py
    from src.core_shell import (
        create_particle_with_shape,
        apply_displacement_to_particle,
        create_layered_displacement_field,
        create_random_strain_field,
        compute_diffraction_oversampled_cropped,
        ScatteringFactors,
        compute_f0_thomson,
        compute_ground_truth_labels,
        center_crop_2d,
        SPECIES_NI, SPECIES_FE,
        DEFAULT_PIXEL_SIZE,
    )
except ImportError:
    # When running from src directory: python validate.py
    from core_shell import (
        create_particle_with_shape,
        apply_displacement_to_particle,
        create_layered_displacement_field,
        create_random_strain_field,
        compute_diffraction_oversampled_cropped,
        ScatteringFactors,
        compute_f0_thomson,
        compute_ground_truth_labels,
        center_crop_2d,
        SPECIES_NI, SPECIES_FE,
        DEFAULT_PIXEL_SIZE,
    )


# =============================================================================
# CONFIGURATION
# =============================================================================

# Grid sizes
GRID_SIZE = 256          # Particle grid
GRID_SIZE_OUTPUT = 128   # Cropped reciprocal space

# Physical parameters
PIXEL_SIZE = 5.0  # Ångströms

# Energies around Ni K-edge (8320 eV)
ENERGIES = [8313, 8318, 8323, 8328, 8333, 8338, 8343, 8348]  # Centered on Ni K-edge (8333 eV)

# Displacement parameters
INTERFACE_AMPLITUDE = 1.5
SURFACE_AMPLITUDE = 0.8
RANDOM_AMPLITUDE = 0.3
RANDOM_CORRELATION = 0.08

# Bragg peak
Q_BRAGG = 3.09  # Å⁻¹ (Ni 111)


# =============================================================================
# MAD EQUATION RECONSTRUCTION
# =============================================================================

def reconstruct_intensity_from_ground_truth(
    F_T_mag: np.ndarray,
    F_A_mag: np.ndarray,
    delta_phi: np.ndarray,
    f_prime: float,
    f_double_prime: float,
    f0: np.ndarray,
    verbose: bool = False
) -> np.ndarray:
    """
    Reconstruct intensity using the MAD equation.
    
    MAD equation:
        I = |F_T|² + (f'² + f''²)|F_A/f₀|² + 2|F_T||F_A/f₀|·[f'·cos(Δφ) + f''·sin(Δφ)]
    
    Parameters
    ----------
    F_T_mag : np.ndarray
        Total structure factor magnitude (Thomson-weighted)
    F_A_mag : np.ndarray
        Anomalous structure factor magnitude (Thomson-weighted)
    delta_phi : np.ndarray
        Phase difference Δφ = φ_T - φ_A
    f_prime : float
        f'(E) for Ni at this energy
    f_double_prime : float
        f''(E) for Ni at this energy
    f0 : np.ndarray
        Thomson scattering factor f₀(Q) for Ni at each Q-point
        
    Returns
    -------
    I_reconstructed : np.ndarray
        Reconstructed intensity
    """
    
    # Precompute
    cos_dphi = np.cos(delta_phi)
    sin_dphi = np.sin(delta_phi)
    
    # Normalized F_A (NanoMAD does: fa = mFA/mF0)
    # Avoid division by zero
    f0_safe = np.where(f0 > 1e-10, f0, 1e-10)
    F_A_normalized = F_A_mag / f0_safe
    
    # Three terms of the MAD equation
    term1 = F_T_mag**2
    term2 = (f_prime**2 + f_double_prime**2) * F_A_normalized**2
    term3 = 2 * F_T_mag * F_A_normalized * (f_prime * cos_dphi + f_double_prime * sin_dphi)
    
    I_reconstructed = term1 + term2 + term3
    
    if verbose:
        print(f"    MAD reconstruction at E with f'={f_prime:.2f}, f''={f_double_prime:.2f}")
        print(f"      Term 1 (|F_T|²) range: [{term1.min():.2e}, {term1.max():.2e}]")
        print(f"      Term 2 (anomalous) range: [{term2.min():.2e}, {term2.max():.2e}]")
        print(f"      Term 3 (interference) range: [{term3.min():.2e}, {term3.max():.2e}]")
        print(f"      I_reconstructed range: [{I_reconstructed.min():.2e}, {I_reconstructed.max():.2e}]")
    
    return I_reconstructed


def validate_ground_truth(
    particle: np.ndarray,
    energies: List[float],
    pixel_size: float,
    scattering_factors: ScatteringFactors,
    output_size: int,
    verbose: bool = True
) -> Dict:
    """
    Full validation: compute ground truth, simulate diffraction, reconstruct, compare.
    
    Parameters
    ----------
    particle : np.ndarray
        Composition-resolved particle (2, Ny, Nx), can be complex
    energies : list
        List of energies in eV
    pixel_size : float
        Pixel size in Ångströms
    scattering_factors : ScatteringFactors
        Object with f'(E) and f''(E) data
    output_size : int
        Size to crop reciprocal space to
    verbose : bool
        Print detailed output
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'labels': Ground truth labels
        - 'intensities_simulated': Dict of simulated intensities per energy
        - 'intensities_reconstructed': Dict of reconstructed intensities per energy
        - 'r_factors': R-factor per energy (measure of agreement)
        - 'correlations': Correlation coefficient per energy
        - 'passed': True if validation passed (R-factors < threshold)
    """
    
    grid_size = particle.shape[1]
    
    if verbose:
        print("\n" + "="*70)
        print("GROUND TRUTH VALIDATION")
        print("="*70)
        print(f"\nGrid size: {grid_size} → crop to {output_size}")
        print(f"Energies: {energies} eV")
    
    # -------------------------------------------------------------------------
    # Step 1: Compute ground truth labels
    # -------------------------------------------------------------------------
    
    if verbose:
        print("\n--- Step 1: Computing ground truth labels ---")
    
    labels = compute_ground_truth_labels(
        particle=particle,
        pixel_size=pixel_size,
        output_size=output_size,
        verbose=verbose
    )
    
    F_T_mag = labels['F_T_mag']
    F_A_mag = labels['F_A_mag']
    delta_phi = labels['delta_phi']
    f0_Ni = labels['f0_Ni']  # f₀(Q) for Ni at each Q-point
    
    # -------------------------------------------------------------------------
    # Step 2: Simulate diffraction at each energy (our "observed" data)
    # -------------------------------------------------------------------------
    
    if verbose:
        print("\n--- Step 2: Simulating diffraction at each energy ---")
    
    diffractions = compute_diffraction_oversampled_cropped(
        particle=particle,
        energies=energies,
        pixel_size=pixel_size,
        scattering_factors=scattering_factors,
        output_size=output_size,
        verbose=verbose
    )
    
    # Convert to intensities
    intensities_simulated = {}
    for E in energies:
        intensities_simulated[E] = np.abs(diffractions[E])**2
    
    # -------------------------------------------------------------------------
    # Step 3: Reconstruct intensity from ground truth using MAD equation
    # -------------------------------------------------------------------------
    
    if verbose:
        print("\n--- Step 3: Reconstructing intensity from ground truth ---")
    
    intensities_reconstructed = {}
    r_factors = {}
    correlations = {}
    
    for E in energies:
        # Get f'(E) and f''(E) for Ni at this energy
        f_prime = scattering_factors.get_f_prime('Ni', E)
        f_double_prime = scattering_factors.get_f_double_prime('Ni', E)
        
        # Reconstruct using MAD equation
        I_recon = reconstruct_intensity_from_ground_truth(
            F_T_mag=F_T_mag,
            F_A_mag=F_A_mag,
            delta_phi=delta_phi,
            f_prime=f_prime,
            f_double_prime=f_double_prime,
            f0=f0_Ni,
            verbose=False
        )
        
        intensities_reconstructed[E] = I_recon
        
        # Compute R-factor (crystallographic agreement metric)
        I_sim = intensities_simulated[E]
        diff = np.abs(I_recon - I_sim)
        r_factor = diff.sum() / (I_sim.sum() + 1e-10)
        r_factors[E] = r_factor
        
        # Compute correlation
        corr = np.corrcoef(I_recon.flatten(), I_sim.flatten())[0, 1]
        correlations[E] = corr
        
        if verbose:
            quality = "excellent" if r_factor < 0.02 else "good" if r_factor < 0.05 else "poor"
            print(f"  E={E} eV: f'={f_prime:+.2f}, f''={f_double_prime:.2f}")
            print(f"    R-factor: {r_factor:.6f} ({quality}), Correlation: {corr:.6f}")
    
    # -------------------------------------------------------------------------
    # Step 4: Summary
    # -------------------------------------------------------------------------
    
    mean_r_factor = np.mean(list(r_factors.values()))
    mean_correlation = np.mean(list(correlations.values()))
    
    # Validation passes if R-factors are small
    # R < 0.05 is good in crystallography, R < 0.02 is excellent
    # Small discrepancies are expected due to numerical precision
    passed = mean_r_factor < 0.05 and mean_correlation > 0.99
    
    if verbose:
        print("\n--- Validation Summary ---")
        print(f"  Mean R-factor: {mean_r_factor:.6f} (want < 0.05, excellent if < 0.02)")
        print(f"  Mean correlation: {mean_correlation:.6f} (want > 0.99)")
        print(f"  VALIDATION {'PASSED ✓' if passed else 'FAILED ✗'}")
    
    return {
        'labels': labels,
        'intensities_simulated': intensities_simulated,
        'intensities_reconstructed': intensities_reconstructed,
        'r_factors': r_factors,
        'correlations': correlations,
        'mean_r_factor': mean_r_factor,
        'mean_correlation': mean_correlation,
        'passed': passed,
    }


def plot_validation_results(
    results: Dict,
    energies: List[float],
    title: str = "Ground Truth Validation"
):
    """
    Plot comparison of simulated vs reconstructed intensities.
    """
    
    n_energies = len(energies)
    
    # Select a few energies to show
    if n_energies <= 4:
        show_energies = energies
    else:
        # Show first, middle, and last
        indices = [0, n_energies//3, 2*n_energies//3, n_energies-1]
        show_energies = [energies[i] for i in indices]
    
    n_show = len(show_energies)
    
    fig, axes = plt.subplots(3, n_show, figsize=(4*n_show, 12))
    
    for col, E in enumerate(show_energies):
        I_sim = results['intensities_simulated'][E]
        I_recon = results['intensities_reconstructed'][E]
        
        # Log scale for display
        I_sim_log = np.log10(I_sim + 1)
        I_recon_log = np.log10(I_recon + 1)
        
        vmin = min(I_sim_log.min(), I_recon_log.min())
        vmax = max(I_sim_log.max(), I_recon_log.max())
        
        # Row 0: Simulated
        im0 = axes[0, col].imshow(I_sim_log, cmap='viridis', origin='lower',
                                   vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f'Simulated\nE={E} eV', fontsize=10)
        if col == n_show - 1:
            plt.colorbar(im0, ax=axes[0, col], label='log₁₀(I+1)')
        
        # Row 1: Reconstructed
        im1 = axes[1, col].imshow(I_recon_log, cmap='viridis', origin='lower',
                                   vmin=vmin, vmax=vmax)
        axes[1, col].set_title(f'Reconstructed\n(from F_T, F_A, Δφ)', fontsize=10)
        if col == n_show - 1:
            plt.colorbar(im1, ax=axes[1, col], label='log₁₀(I+1)')
        
        # Row 2: Difference (linear scale, centered at 0)
        diff = I_recon - I_sim
        diff_max = max(abs(diff.min()), abs(diff.max()))
        if diff_max < 1e-10:
            diff_max = 1.0
        
        im2 = axes[2, col].imshow(diff, cmap='RdBu', origin='lower',
                                   vmin=-diff_max, vmax=diff_max)
        r_factor = results['r_factors'][E]
        corr = results['correlations'][E]
        quality = "excellent" if r_factor < 0.02 else "good" if r_factor < 0.05 else "poor"
        axes[2, col].set_title(f'Difference\nR={r_factor:.4f} ({quality})', fontsize=10)
        if col == n_show - 1:
            plt.colorbar(im2, ax=axes[2, col], label='I_recon - I_sim')
    
    # Row labels
    axes[0, 0].set_ylabel('Simulated', fontsize=12)
    axes[1, 0].set_ylabel('Reconstructed', fontsize=12)
    axes[2, 0].set_ylabel('Difference', fontsize=12)
    
    status = "PASSED ✓" if results['passed'] else "FAILED ✗"
    fig.suptitle(f'{title}\nMean R-factor: {results["mean_r_factor"]:.6f}, '
                 f'Mean correlation: {results["mean_correlation"]:.6f} — {status}',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_ground_truth_components(labels: Dict, title: str = "Ground Truth Components"):
    """
    Plot the ground truth label components.
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # |F_T|
    im00 = axes[0, 0].imshow(np.log10(labels['F_T_mag'] + 1), cmap='viridis', origin='lower')
    axes[0, 0].set_title('|F_T| (Total)', fontsize=11)
    plt.colorbar(im00, ax=axes[0, 0], label='log₁₀(|F_T|+1)')
    
    # |F_A|
    im01 = axes[0, 1].imshow(np.log10(labels['F_A_mag'] + 1), cmap='viridis', origin='lower')
    axes[0, 1].set_title('|F_A| (Anomalous = Ni)', fontsize=11)
    plt.colorbar(im01, ax=axes[0, 1], label='log₁₀(|F_A|+1)')
    
    # |F_N|
    im02 = axes[0, 2].imshow(np.log10(labels['F_N_mag'] + 1), cmap='viridis', origin='lower')
    axes[0, 2].set_title('|F_N| (Non-anomalous = Fe)', fontsize=11)
    plt.colorbar(im02, ax=axes[0, 2], label='log₁₀(|F_N|+1)')
    
    # Δφ
    im10 = axes[1, 0].imshow(labels['delta_phi'], cmap='twilight', origin='lower',
                              vmin=-np.pi, vmax=np.pi)
    axes[1, 0].set_title('Δφ = φ_T - φ_A', fontsize=11)
    plt.colorbar(im10, ax=axes[1, 0], label='Δφ (rad)')
    
    # sin(Δφ)
    im11 = axes[1, 1].imshow(labels['sin_delta_phi'], cmap='RdBu', origin='lower',
                              vmin=-1, vmax=1)
    axes[1, 1].set_title('sin(Δφ)', fontsize=11)
    plt.colorbar(im11, ax=axes[1, 1], label='sin(Δφ)')
    
    # cos(Δφ)
    im12 = axes[1, 2].imshow(labels['cos_delta_phi'], cmap='RdBu', origin='lower',
                              vmin=-1, vmax=1)
    axes[1, 2].set_title('cos(Δφ)', fontsize=11)
    plt.colorbar(im12, ax=axes[1, 2], label='cos(Δφ)')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN VALIDATION TESTS
# =============================================================================

def run_validation_tests(data_dir: str = '.', show_plots: bool = True):
    """
    Run validation on several test particles.
    """
    
    print("\n" + "="*70)
    print("GROUND TRUTH VALIDATION SUITE")
    print("="*70)
    print("\nThis script validates that our computed ground truth labels")
    print("(F_T, F_A, Δφ) can reconstruct the simulated intensities via")
    print("the MAD equation. If R-factors are ~0 and correlations are ~1,")
    print("our ground truth is correct!")
    
    # Load scattering factors
    print(f"\nLoading scattering factors from: {data_dir}")
    sf = ScatteringFactors(data_dir=data_dir)
    
    # Test configurations
    test_configs = [
        {
            'name': 'Simple Hexagon (no strain)',
            'shape': 'hexagon',
            'anisotropy': 1.0,
            'apply_strain': False,
        },
        {
            'name': 'Hexagon with strain',
            'shape': 'hexagon',
            'anisotropy': 1.0,
            'apply_strain': True,
        },
        {
            'name': 'Elongated hexagon with strain',
            'shape': 'hexagon',
            'anisotropy': 1.3,
            'apply_strain': True,
        },
        {
            'name': 'Random polygon with strain',
            'shape': 'polygon',
            'n_vertices': 8,
            'apply_strain': True,
        },
    ]
    
    results_all = []
    
    for config in test_configs:
        print("\n" + "-"*70)
        print(f"TEST: {config['name']}")
        print("-"*70)
        
        # Create particle
        if config['shape'] == 'hexagon':
            shape_params = {'anisotropy': config.get('anisotropy', 1.0)}
            particle, info = create_particle_with_shape(
                grid_size=GRID_SIZE,
                shape_type='hexagon',
                outer_radius=40,
                core_fraction=0.5,
                shape_params=shape_params,
            )
        else:
            shape_params = {
                'min_corners': config.get('n_vertices', 7),
                'max_corners': config.get('n_vertices', 7) + 1,
            }
            particle, info = create_particle_with_shape(
                grid_size=GRID_SIZE,
                shape_type='polygon',
                outer_radius=40,
                core_fraction=0.5,
                shape_params=shape_params,
            )
        
        # Apply strain if requested
        if config.get('apply_strain', False):
            outer_mask = info['outer_mask']
            core_mask = info['core_mask']
            outer_vertices = info.get('outer_vertices', [])
            
            # Create displacement field
            displacement_analytic, _ = create_layered_displacement_field(
                core_mask=core_mask,
                outer_mask=outer_mask,
                pixel_size=PIXEL_SIZE,
                interface_amplitude=INTERFACE_AMPLITUDE,
                surface_amplitude=SURFACE_AMPLITUDE,
                corner_positions=outer_vertices,
                verbose=False
            )
            
            displacement_random = create_random_strain_field(
                grid_size=GRID_SIZE,
                pixel_size=PIXEL_SIZE,
                displacement_amplitude=RANDOM_AMPLITUDE,
                correlation_length=RANDOM_CORRELATION,
                verbose=False
            )
            
            displacement = displacement_analytic + displacement_random
            
            particle = apply_displacement_to_particle(
                particle=particle,
                displacement=displacement,
                q_bragg_magnitude=Q_BRAGG
            )
        
        # Run validation
        results = validate_ground_truth(
            particle=particle,
            energies=ENERGIES,
            pixel_size=PIXEL_SIZE,
            scattering_factors=sf,
            output_size=GRID_SIZE_OUTPUT,
            verbose=True
        )
        
        results['config'] = config
        results_all.append(results)
        
        # Plot if requested
        if show_plots:
            plot_ground_truth_components(
                results['labels'],
                title=f"{config['name']}: Ground Truth Components"
            )
            
            plot_validation_results(
                results,
                energies=ENERGIES,
                title=f"{config['name']}: Validation"
            )
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    all_passed = True
    for results in results_all:
        name = results['config']['name']
        status = "PASSED ✓" if results['passed'] else "FAILED ✗"
        r_factor = results['mean_r_factor']
        corr = results['mean_correlation']
        print(f"  {name}: {status}")
        print(f"    R-factor: {r_factor:.6f}, Correlation: {corr:.6f}")
        if not results['passed']:
            all_passed = False
    
    print("\n" + "-"*70)
    if all_passed:
        print("ALL TESTS PASSED ✓")
        print("Ground truth labels are consistent with simulated intensities!")
    else:
        print("SOME TESTS FAILED ✗")
        print("Investigate the failing cases.")
    print("-"*70)
    
    return results_all


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import sys
    
    # Allow specifying data directory as command line argument
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = '.'
    
    # Run validation
    results = run_validation_tests(data_dir=data_dir, show_plots=True)
