#!/usr/bin/env python3
"""
visualization.py

Visualization functions for NanoMAD core-shell particle simulation.

These functions are extracted from core_shell.py to keep the main physics
module focused on computation. They are optional for production use but
helpful for debugging and understanding the data.

Author: Claude (Anthropic) + Thomas
Date: December 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple

# Import from core_shell for constants and utility functions
try:
    # When running from repository root
    from src.core_shell import (
        SPECIES_NI, SPECIES_FE,
        get_diffraction_intensity,
        reconstruct,
        process_block_at_energy,
        ScatteringFactors,
    )
except ImportError:
    # When running from src directory
    from core_shell import (
        SPECIES_NI, SPECIES_FE,
        get_diffraction_intensity,
        reconstruct,
        process_block_at_energy,
        ScatteringFactors,
    )


# =============================================================================
# PARTICLE VISUALIZATION
# =============================================================================

def plot_particle(particle: np.ndarray, info: Dict[str, Any] = None,
                  title: str = "Core-Shell Particle") -> None:
    """
    Plot a composition-resolved particle showing Ni map, Fe map, and total density.
    """

    if particle.ndim == 3:
        _plot_particle_composition_resolved(particle, info, title)
    else:
        _plot_particle_single(particle, info, title)


def _plot_particle_single(particle: np.ndarray, info: Dict[str, Any],
                          title: str) -> None:
    """Plot a single 2D density map."""

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(particle, cmap='viridis', origin='lower')
    plt.colorbar(im, ax=ax, label='Electron Density (relative)')

    if info is not None:
        title += f"\nGrid: {info['grid_size']}×{info['grid_size']}, "
        title += f"Outer R: {info['outer_radius']:.0f}, Core R: {info['core_radius']:.0f}"

    ax.set_title(title)
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')

    plt.tight_layout()
    plt.show()


def _plot_particle_composition_resolved(particle: np.ndarray, info: Dict[str, Any],
                                         title: str) -> None:
    """Plot a composition-resolved particle with Ni, Fe, and total views."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ni_map = particle[SPECIES_NI]
    fe_map = particle[SPECIES_FE]
    total_map = ni_map + fe_map

    vmin_species = 0
    vmax_species = max(ni_map.max(), fe_map.max())

    im0 = axes[0].imshow(ni_map, cmap='Blues', origin='lower',
                          vmin=vmin_species, vmax=vmax_species)
    axes[0].set_title('Nickel (Ni) Density')
    axes[0].set_xlabel('x (pixels)')
    axes[0].set_ylabel('y (pixels)')
    plt.colorbar(im0, ax=axes[0], label='Ni fraction')

    im1 = axes[1].imshow(fe_map, cmap='Oranges', origin='lower',
                          vmin=vmin_species, vmax=vmax_species)
    axes[1].set_title('Iron (Fe) Density')
    axes[1].set_xlabel('x (pixels)')
    axes[1].set_ylabel('y (pixels)')
    plt.colorbar(im1, ax=axes[1], label='Fe fraction')

    im2 = axes[2].imshow(total_map, cmap='viridis', origin='lower')
    axes[2].set_title('Total Density (Ni + Fe)')
    axes[2].set_xlabel('x (pixels)')
    axes[2].set_ylabel('y (pixels)')
    plt.colorbar(im2, ax=axes[2], label='Total fraction')

    if info is not None:
        core = info['core_composition']
        shell = info['shell_composition']
        title += f"\nCore: Ni={core['Ni']:.0%}, Fe={core['Fe']:.0%}"
        title += f" | Shell: Ni={shell['Ni']:.0%}, Fe={shell['Fe']:.0%}"
        title += f"\nGeometry: Outer R={info['outer_radius']:.0f}, Core R={info['core_radius']:.0f}"

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


# =============================================================================
# DIFFRACTION VISUALIZATION
# =============================================================================

def plot_diffraction(diffraction: np.ndarray, log_scale: bool = True,
                     title: str = "Diffraction Pattern") -> None:
    """Plot a diffraction pattern (intensity)."""

    intensity = get_diffraction_intensity(diffraction, log_scale=log_scale)

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(intensity, cmap='viridis', origin='lower')

    label = 'log₁₀(Intensity)' if log_scale else 'Intensity'
    plt.colorbar(im, ax=ax, label=label)

    ax.set_title(title)
    ax.set_xlabel('qx (pixels⁻¹)')
    ax.set_ylabel('qy (pixels⁻¹)')

    plt.tight_layout()
    plt.show()


def plot_diffraction_multi_energy(
    diffractions: Dict[float, np.ndarray],
    log_scale: bool = True,
    title: str = "Diffraction Patterns at Multiple Energies"
) -> None:
    """
    Plot diffraction patterns at multiple energies side by side.

    Parameters
    ----------
    diffractions : dict
        Dictionary mapping energy (float) to diffraction pattern (2D complex array)

    log_scale : bool
        If True, plot log10(intensity)

    title : str
        Overall figure title
    """

    energies = sorted(diffractions.keys())
    n_energies = len(energies)

    fig, axes = plt.subplots(1, n_energies, figsize=(5*n_energies, 5))

    if n_energies == 1:
        axes = [axes]

    # Compute all intensities to get common colorscale
    intensities = [get_diffraction_intensity(diffractions[E], log_scale) for E in energies]
    vmin = min(I.min() for I in intensities)
    vmax = max(I.max() for I in intensities)

    for ax, E, intensity in zip(axes, energies, intensities):
        im = ax.imshow(intensity, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(f'E = {E:.1f} eV')
        ax.set_xlabel('qx (pixels⁻¹)')
        ax.set_ylabel('qy (pixels⁻¹)')

    # Add colorbar to the last axis
    label = 'log₁₀(Intensity)' if log_scale else 'Intensity'
    plt.colorbar(im, ax=axes[-1], label=label)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


# =============================================================================
# RECONSTRUCTION VISUALIZATION
# =============================================================================

def plot_reconstruction(amplitude: np.ndarray, phase: np.ndarray,
                        title_prefix: str = "Reconstruction") -> None:
    """Plot amplitude and phase from reconstruction side by side."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im0 = axes[0].imshow(amplitude, cmap='viridis', origin='lower')
    axes[0].set_title(f'{title_prefix}: Amplitude')
    axes[0].set_xlabel('x (pixels)')
    axes[0].set_ylabel('y (pixels)')
    plt.colorbar(im0, ax=axes[0], label='Amplitude')

    im1 = axes[1].imshow(phase, cmap='twilight', origin='lower',
                         vmin=-np.pi, vmax=np.pi)
    axes[1].set_title(f'{title_prefix}: Phase')
    axes[1].set_xlabel('x (pixels)')
    axes[1].set_ylabel('y (pixels)')
    plt.colorbar(im1, ax=axes[1], label='Phase (radians)')

    plt.tight_layout()
    plt.show()


def plot_reconstruction_multi_energy(
    diffractions: Dict[float, np.ndarray],
    title: str = "Reconstructions at Multiple Energies"
) -> None:
    """
    Plot amplitude and phase reconstructions for multiple energies.

    Parameters
    ----------
    diffractions : dict
        Dictionary mapping energy to diffraction pattern

    title : str
        Overall figure title
    """

    energies = sorted(diffractions.keys())
    n_energies = len(energies)

    fig, axes = plt.subplots(2, n_energies, figsize=(5*n_energies, 10))

    if n_energies == 1:
        axes = axes.reshape(2, 1)

    # Compute reconstructions
    reconstructions = {}
    for E in energies:
        amp, ph = reconstruct(diffractions[E])
        reconstructions[E] = (amp, ph)

    # Common colorscales
    all_amps = [reconstructions[E][0] for E in energies]
    amp_vmin = min(a.min() for a in all_amps)
    amp_vmax = max(a.max() for a in all_amps)

    for i, E in enumerate(energies):
        amp, phase = reconstructions[E]

        # Amplitude (top row)
        im0 = axes[0, i].imshow(amp, cmap='viridis', origin='lower',
                                 vmin=amp_vmin, vmax=amp_vmax)
        axes[0, i].set_title(f'Amplitude (E={E:.1f} eV)')
        axes[0, i].set_xlabel('x (pixels)')
        axes[0, i].set_ylabel('y (pixels)')

        # Phase (bottom row)
        im1 = axes[1, i].imshow(phase, cmap='twilight', origin='lower',
                                 vmin=-np.pi, vmax=np.pi)
        axes[1, i].set_title(f'Phase (E={E:.1f} eV)')
        axes[1, i].set_xlabel('x (pixels)')
        axes[1, i].set_ylabel('y (pixels)')

    # Add colorbars
    plt.colorbar(im0, ax=axes[0, -1], label='Amplitude')
    plt.colorbar(im1, ax=axes[1, -1], label='Phase (radians)')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


# =============================================================================
# BLOCK ANALYSIS VISUALIZATION
# =============================================================================

def plot_block_analysis(block_result: Dict[str, np.ndarray],
                        block_index: Tuple[int, int] = None) -> None:
    """Plot all views of a single block."""

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    idx_str = f" [{block_index[0]}, {block_index[1]}]" if block_index else ""

    effective_density = block_result.get('effective_density', block_result['block'])
    if np.iscomplexobj(effective_density):
        # For energy-dependent processing, show the magnitude
        display_density = np.abs(effective_density)
    else:
        if effective_density.ndim == 3:
            display_density = np.sum(effective_density, axis=0)
        else:
            display_density = effective_density

    im0 = axes[0].imshow(np.real(display_density), cmap='viridis', origin='lower')
    axes[0].set_title(f'Block{idx_str}')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(block_result['intensity'], cmap='viridis', origin='lower')
    axes[1].set_title(f'Diffraction{idx_str}')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(block_result['amplitude'], cmap='viridis', origin='lower')
    axes[2].set_title(f'Amplitude{idx_str}')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(block_result['phase'], cmap='twilight', origin='lower',
                         vmin=-np.pi, vmax=np.pi)
    axes[3].set_title(f'Phase{idx_str}')
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    plt.tight_layout()
    plt.show()


def plot_block_analysis_multi_energy(
    block: np.ndarray,
    energies: List[float],
    scattering_factors: ScatteringFactors,
    block_index: Tuple[int, int] = None
) -> None:
    """
    Plot block analysis for multiple energies.

    Shows a grid of plots: rows = energies, cols = [density, diffraction, amplitude, phase]

    Parameters
    ----------
    block : np.ndarray
        Composition-resolved block of shape (2, block_size, block_size)

    energies : list of float
        List of energies in eV

    scattering_factors : ScatteringFactors
        ScatteringFactors object

    block_index : tuple, optional
        Block position for labeling
    """

    n_energies = len(energies)

    fig, axes = plt.subplots(n_energies, 4, figsize=(16, 4*n_energies))

    if n_energies == 1:
        axes = axes.reshape(1, -1)

    idx_str = f" [{block_index[0]}, {block_index[1]}]" if block_index else ""

    # Process each energy
    results = []
    for E in energies:
        result = process_block_at_energy(block, E, scattering_factors)
        results.append(result)

    # Get common colorscales
    all_intensities = [r['intensity'] for r in results]
    int_vmin = min(I.min() for I in all_intensities)
    int_vmax = max(I.max() for I in all_intensities)

    all_amplitudes = [r['amplitude'] for r in results]
    amp_vmin = min(a.min() for a in all_amplitudes)
    amp_vmax = max(a.max() for a in all_amplitudes)

    for i, (E, result) in enumerate(zip(energies, results)):
        # Effective density magnitude (it's complex now)
        eff_dens = np.abs(result['effective_density'])
        im0 = axes[i, 0].imshow(eff_dens, cmap='viridis', origin='lower')
        axes[i, 0].set_title(f'|ρ_eff| (E={E:.0f} eV)')
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)

        # Diffraction intensity
        im1 = axes[i, 1].imshow(result['intensity'], cmap='viridis', origin='lower',
                                 vmin=int_vmin, vmax=int_vmax)
        axes[i, 1].set_title(f'Diffraction (E={E:.0f} eV)')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)

        # Amplitude
        im2 = axes[i, 2].imshow(result['amplitude'], cmap='viridis', origin='lower',
                                 vmin=amp_vmin, vmax=amp_vmax)
        axes[i, 2].set_title(f'Amplitude (E={E:.0f} eV)')
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)

        # Phase
        im3 = axes[i, 3].imshow(result['phase'], cmap='twilight', origin='lower',
                                 vmin=-np.pi, vmax=np.pi)
        axes[i, 3].set_title(f'Phase (E={E:.0f} eV)')
        plt.colorbar(im3, ax=axes[i, 3], fraction=0.046)

    fig.suptitle(f'Block{idx_str} Analysis at Multiple Energies', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_all_blocks_summary(blocks: np.ndarray, title: str = "All Blocks") -> None:
    """Plot all blocks in a grid."""

    n_blocks_y, n_blocks_x = blocks.shape[:2]

    fig, axes = plt.subplots(n_blocks_y, n_blocks_x,
                             figsize=(2*n_blocks_x, 2*n_blocks_y))

    is_composition_resolved = (blocks.ndim == 5)

    all_values = []
    for i in range(n_blocks_y):
        for j in range(n_blocks_x):
            block = blocks[i, j]
            if is_composition_resolved:
                block = np.sum(block, axis=0)
            all_values.append(block)

    all_values = np.array(all_values)
    vmin, vmax = all_values.min(), all_values.max()

    for i in range(n_blocks_y):
        for j in range(n_blocks_x):
            ax = axes[i, j]
            block = blocks[i, j]
            if is_composition_resolved:
                block = np.sum(block, axis=0)

            ax.imshow(block, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'[{i},{j}]', fontsize=8)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()
