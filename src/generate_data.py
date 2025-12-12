#!/usr/bin/env python3
"""
generate_training_data.py

Generate synthetic training data for the MAD CNN.

This script creates a diverse dataset of simulated nanoparticle diffraction
patterns with their corresponding ground truth labels. Each particle is
processed through the full workflow:
    1. Create particle with varied shape parameters
    2. Apply layered displacement field (interface + surface + corners)
    3. Add random strain
    4. Optionally add module noise (compositional variations)
    5. Compute multi-energy diffraction
    6. Add Poisson noise
    7. Extract patches and compute ground truth labels

The output is saved as individual .npz files per particle, plus a metadata
file summarizing the dataset.

Output structure:
    output_dir/
    ├── metadata.json          # Dataset summary and parameters
    ├── particle_0000.npz      # First particle
    ├── particle_0001.npz      # Second particle
    ├── ...
    └── particle_NNNN.npz      # Last particle

Each .npz file contains:
    - X: (n_patches, 16, 16, 8) - intensity patches (CNN input)
    - Y: (n_patches, 16, 16, 4) - label patches (CNN output)
    - f_prime: (8,) - f'(E) for Ni at each energy
    - f_double_prime: (8,) - f''(E) for Ni at each energy
    - energies: (8,) - energy values in eV
    - shape_type: str - particle shape
    - params: dict - particle parameters

Author: Claude (Anthropic) + Thomas
Date: December 2024
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import sys

# Import from core modules - use try/except for flexible import paths
try:
    # When running from repository root: python src/generate_data.py
    from src.core_shell import (
        create_particle_with_shape,
        apply_displacement_to_particle,
        add_correlated_module_noise,
        create_layered_displacement_field,
        create_random_strain_field,
        compute_diffraction_oversampled_cropped,
        ScatteringFactors,
        add_poisson_noise,
        compute_ground_truth_labels,
        extract_label_patches,
        DEFAULT_PIXEL_SIZE,
        clamp_radius_to_grid,
        validate_particle_bounds,
    )
except ImportError:
    # When running from src directory: python generate_data.py
    from core_shell import (
        create_particle_with_shape,
        apply_displacement_to_particle,
        add_correlated_module_noise,
        create_layered_displacement_field,
        create_random_strain_field,
        compute_diffraction_oversampled_cropped,
        ScatteringFactors,
        add_poisson_noise,
        compute_ground_truth_labels,
        extract_label_patches,
        DEFAULT_PIXEL_SIZE,
        clamp_radius_to_grid,
        validate_particle_bounds,
    )


# =============================================================================
# CONFIGURATION
# =============================================================================

# Output directory (default, can be overridden with -o flag)
OUTPUT_DIR = Path('synthetic_data')

# Number of particles to generate
N_PARTICLES = 100  # Start small for testing, increase later

# Grid sizes
GRID_SIZE_FFT = 256       # FFT computation grid
GRID_SIZE_OUTPUT = 128    # Cropped output size
PATCH_SIZE = 16           # CNN patch size

# Physical parameters
PIXEL_SIZE = 5.0  # Ångströms

# Energies around Ni K-edge (8320 eV)
ENERGIES = [8313, 8318, 8323, 8328, 8333, 8338, 8343, 8348]  # Centered on Ni K-edge (8333 eV)
N_ENERGIES = len(ENERGIES)

# Bragg peak
Q_BRAGG = 3.09  # Å⁻¹ (Ni 111)

# Noise settings
USE_MODULE_NOISE = True
USE_POISSON_NOISE = True

# Maximum photon count at brightest pixel
# This sets the intensity scale of the training data.
# Should match experimental conditions (~1e6 for typical synchrotron data).
# Previous value (5000) was too low and caused scale mismatch with test data.
POISSON_MAX_COUNTS = 1000000  # 1e6 - realistic synchrotron photon counts

# Shape distribution (probabilities for each shape type)
SHAPE_DISTRIBUTION = {
    'hexagon': 0.30,               # Was 0.35
    'polygon': 0.20,               # Was 0.25
    'polygon_centrosymmetric': 0.20,
    'circle': 0.15,
    'ellipse': 0.15,               # NEW
}

# Composition distribution (probabilities for each composition mode)
COMPOSITION_DISTRIBUTION = {
    'sharp': 0.40,           # Traditional core-shell
    'radial_gradient': 0.18,
    'linear_gradient': 0.10,
    'janus': 0.08,
    'multi_shell': 0.07,
    'uniform': 0.17,         # Pure Ni3Fe, Ni, or Fe (no core-shell)
}

# For 'uniform' mode, sample which pure composition:
UNIFORM_COMPOSITION_OPTIONS = [
    {'Ni': 0.75, 'Fe': 0.25},  # Ni3Fe
    {'Ni': 1.0, 'Fe': 0.0},    # Pure Ni
    {'Ni': 0.0, 'Fe': 1.0},    # Pure Fe
]

# Probability of applying optional geometry modifiers
TRUNCATION_PROBABILITY = 0.15    # 15% of particles get Winterbottom truncation
OFF_CENTER_PROBABILITY = 0.10    # 10% of particles have off-center cores

# Parameter ranges for random variation
PARAM_RANGES = {
    # Particle size (radius in pixels) - EXPANDED
    'outer_radius': (20, 60),       # Was (30, 50)

    # Size scale multiplier
    'size_scale': (0.7, 1.3),       # NEW

    # Core fraction (0.4 = small core, 0.7 = large core)
    'core_fraction': (0.4, 0.7),

    # Hexagon anisotropy (1.0 = regular, 1.3 = elongated)
    'anisotropy': (0.9, 1.3),

    # Ellipse aspect ratio
    'ellipse_aspect_ratio': (1.2, 2.0),  # NEW

    # Polygon vertices
    'n_vertices': (5, 12),

    # Winterbottom truncation fraction
    'truncation_fraction': (0.1, 0.6),   # NEW

    # Gradient transition width (as fraction of radius difference)
    'gradient_transition_width': (0.1, 0.4),  # NEW

    # Multi-shell: number of shells
    'n_shells': (2, 4),              # NEW

    # Displacement amplitudes (Ångströms)
    'interface_amplitude': (0.5, 2.5),
    'surface_amplitude': (0.3, 1.5),
    'random_amplitude': (0.1, 0.5),

    # Random strain correlation length (normalized 0-1)
    'random_correlation': (0.05, 0.15),

    # Module noise amplitude (if enabled)
    'module_noise_amplitude': (0.02, 0.08),
}

# Random seed for reproducibility (set to None for random)
RANDOM_SEED = 42


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def sample_shape_type() -> str:
    """Sample a random shape type according to the distribution."""
    shapes = list(SHAPE_DISTRIBUTION.keys())
    probs = list(SHAPE_DISTRIBUTION.values())
    return np.random.choice(shapes, p=probs)


def sample_composition_mode() -> str:
    """Sample a composition mode according to the distribution."""
    modes = list(COMPOSITION_DISTRIBUTION.keys())
    probs = list(COMPOSITION_DISTRIBUTION.values())
    return np.random.choice(modes, p=probs)


def sample_composition_parameters(
    composition_mode: str,
    outer_radius: float,
    core_fraction: float
) -> Dict[str, Any]:
    """
    Sample mode-specific composition parameters.

    Parameters
    ----------
    composition_mode : str
        One of: 'sharp', 'radial_gradient', 'linear_gradient', 'janus', 'multi_shell', 'uniform'
    outer_radius : float
        Particle outer radius (for radius-dependent parameters)
    core_fraction : float
        Core fraction (for computing shell radii)

    Returns
    -------
    params : dict
        Mode-specific parameters
    """
    # Default compositions
    core_composition = {'Ni': 0.75, 'Fe': 0.25}  # Ni3Fe
    shell_composition = {'Ni': 1.0, 'Fe': 0.0}   # Pure Ni

    if composition_mode == 'sharp':
        # No extra parameters needed - uses core/shell compositions directly
        return {}

    elif composition_mode == 'radial_gradient':
        return {
            'inner_composition': core_composition,
            'outer_composition': shell_composition,
            'transition_width': np.random.uniform(*PARAM_RANGES['gradient_transition_width']),
        }

    elif composition_mode == 'linear_gradient':
        return {
            'composition_start': core_composition,
            'composition_end': shell_composition,
            'gradient_direction': np.random.uniform(0, 2 * np.pi),
            'gradient_width': None,  # Use default (30% of extent)
        }

    elif composition_mode == 'janus':
        # Randomly choose two different compositions for the two halves
        compositions = [
            {'Ni': 0.75, 'Fe': 0.25},  # Ni3Fe
            {'Ni': 1.0, 'Fe': 0.0},    # Pure Ni
            {'Ni': 0.0, 'Fe': 1.0},    # Pure Fe
            {'Ni': 0.5, 'Fe': 0.5},    # NiFe
        ]
        comp_a, comp_b = np.random.choice(len(compositions), size=2, replace=False)
        return {
            'composition_a': compositions[comp_a],
            'composition_b': compositions[comp_b],
            'split_angle': np.random.uniform(0, 2 * np.pi),
            'interface_width': np.random.choice([0.0, 2.0, 5.0]),  # Sharp or smooth
        }

    elif composition_mode == 'multi_shell':
        # Random number of shells (2-4 total regions)
        n_regions = np.random.randint(*PARAM_RANGES['n_shells']) + 1
        n_boundaries = n_regions - 1

        # Generate shell radii at random fractions
        fractions = np.sort(np.random.uniform(0.3, 0.85, size=n_boundaries))
        shell_radii = [outer_radius * f for f in fractions]

        # Generate random compositions for each region
        possible_compositions = [
            {'Ni': 0.75, 'Fe': 0.25},  # Ni3Fe
            {'Ni': 1.0, 'Fe': 0.0},    # Pure Ni
            {'Ni': 0.0, 'Fe': 1.0},    # Pure Fe
            {'Ni': 0.5, 'Fe': 0.5},    # NiFe
        ]
        shell_compositions = [
            possible_compositions[np.random.randint(len(possible_compositions))]
            for _ in range(n_regions)
        ]

        return {
            'shell_radii': shell_radii,
            'shell_compositions': shell_compositions,
            'transition_width': np.random.choice([0.0, 2.0]),  # Sharp or smooth
        }

    elif composition_mode == 'uniform':
        # Choose one of the pure compositions
        composition = UNIFORM_COMPOSITION_OPTIONS[
            np.random.randint(len(UNIFORM_COMPOSITION_OPTIONS))
        ]
        return {
            'composition': composition,
        }

    else:
        return {}


def sample_parameters(shape_type: str, composition_mode: str = None) -> Dict[str, Any]:
    """
    Sample random parameters for a particle.

    Includes boundary validation to ensure particles fit within the grid.
    """
    # Sample composition mode if not provided
    if composition_mode is None:
        composition_mode = sample_composition_mode()

    # Sample base radius and apply size scale
    base_radius = np.random.uniform(*PARAM_RANGES['outer_radius'])
    size_scale = np.random.uniform(*PARAM_RANGES['size_scale'])
    outer_radius = base_radius * size_scale

    # CRITICAL: Clamp radius to ensure particle fits in grid
    center = (GRID_SIZE_FFT / 2, GRID_SIZE_FFT / 2)
    max_valid_radius = clamp_radius_to_grid(center, GRID_SIZE_FFT, margin=2)
    outer_radius = min(outer_radius, max_valid_radius)

    # Ensure minimum viable particle size
    if outer_radius < 15:
        outer_radius = np.random.uniform(15, max_valid_radius)

    core_fraction = np.random.uniform(*PARAM_RANGES['core_fraction'])

    params = {
        'shape_type': shape_type,
        'composition_mode': composition_mode,
        'outer_radius': outer_radius,
        'core_fraction': core_fraction,
        'interface_amplitude': np.random.uniform(*PARAM_RANGES['interface_amplitude']),
        'surface_amplitude': np.random.uniform(*PARAM_RANGES['surface_amplitude']),
        'random_amplitude': np.random.uniform(*PARAM_RANGES['random_amplitude']),
        'random_correlation': np.random.uniform(*PARAM_RANGES['random_correlation']),
    }

    # Shape-specific parameters
    if shape_type == 'hexagon':
        params['anisotropy'] = np.random.uniform(*PARAM_RANGES['anisotropy'])
        params['rotation_angle'] = np.random.uniform(0, 2 * np.pi)
    elif shape_type in ['polygon', 'polygon_centrosymmetric']:
        params['n_vertices'] = np.random.randint(*PARAM_RANGES['n_vertices'])
    elif shape_type == 'ellipse':
        params['ellipse_aspect_ratio'] = np.random.uniform(*PARAM_RANGES['ellipse_aspect_ratio'])
        params['rotation_angle'] = np.random.uniform(0, np.pi)

    # NEW: Random Winterbottom truncation
    if np.random.random() < TRUNCATION_PROBABILITY:
        params['truncation_fraction'] = np.random.uniform(*PARAM_RANGES['truncation_fraction'])
        params['truncation_angle'] = np.random.uniform(0, 2 * np.pi)
    else:
        params['truncation_fraction'] = 0.0
        params['truncation_angle'] = 0.0

    # NEW: Random off-center core (only for sharp/radial_gradient modes)
    if composition_mode in ['sharp', 'radial_gradient'] and np.random.random() < OFF_CENTER_PROBABILITY:
        # Maximum offset keeps core inside particle
        max_offset = outer_radius * core_fraction * 0.3
        params['core_offset'] = (
            np.random.uniform(-max_offset, max_offset),
            np.random.uniform(-max_offset, max_offset)
        )
    else:
        params['core_offset'] = (0.0, 0.0)

    # NEW: Composition mode parameters
    params['composition_params'] = sample_composition_parameters(
        composition_mode, outer_radius, core_fraction
    )

    # Module noise
    if USE_MODULE_NOISE:
        params['module_noise_amplitude'] = np.random.uniform(*PARAM_RANGES['module_noise_amplitude'])

    return params


def extract_intensity_patches(intensity_cube: np.ndarray, patch_size: int) -> np.ndarray:
    """Extract non-overlapping patches from intensity cube."""
    ny, nx, n_energies = intensity_cube.shape
    n_patches_y = ny // patch_size
    n_patches_x = nx // patch_size

    patches = np.zeros((n_patches_y, n_patches_x, patch_size, patch_size, n_energies))

    for i in range(n_patches_y):
        for j in range(n_patches_x):
            y_start = i * patch_size
            x_start = j * patch_size
            patches[i, j] = intensity_cube[
                y_start:y_start + patch_size,
                x_start:x_start + patch_size,
                :
            ]

    return patches


def extract_intensity_patches_overlapping(
    intensity_cube: np.ndarray,
    patch_size: int,
    stride: int
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Extract overlapping patches from intensity cube.

    Parameters
    ----------
    intensity_cube : np.ndarray
        Shape (H, W, n_energies)
    patch_size : int
        Size of each patch
    stride : int
        Step between patch origins (stride < patch_size gives overlap)

    Returns
    -------
    patches : np.ndarray
        Shape (n_patches, patch_size, patch_size, n_energies)
    positions : list of (y, x) tuples
        Origin positions for each patch
    """
    ny, nx, n_energies = intensity_cube.shape
    patches = []
    positions = []

    for y in range(0, ny - patch_size + 1, stride):
        for x in range(0, nx - patch_size + 1, stride):
            patch = intensity_cube[y:y + patch_size, x:x + patch_size, :]
            patches.append(patch)
            positions.append((y, x))

    return np.array(patches), positions


def extract_label_patches_overlapping(
    labels: Dict[str, np.ndarray],
    patch_size: int,
    stride: int,
    label_keys: List[str] = None
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Extract overlapping patches from ground truth label arrays.

    Parameters
    ----------
    labels : dict
        Dictionary of label arrays (each shape H × W)
    patch_size : int
        Size of each patch
    stride : int
        Step between patch origins
    label_keys : list, optional
        Which label keys to extract. Default: ['F_T_mag', 'F_A_mag', 'sin_delta_phi', 'cos_delta_phi']

    Returns
    -------
    patches : np.ndarray
        Shape (n_patches, patch_size, patch_size, n_channels)
    positions : list of (y, x) tuples
        Origin positions for each patch
    """
    if label_keys is None:
        label_keys = ['F_T_mag', 'F_A_mag', 'sin_delta_phi', 'cos_delta_phi']

    # Stack labels into a cube (H, W, n_channels)
    label_cube = np.stack([labels[key] for key in label_keys], axis=-1)
    ny, nx, n_channels = label_cube.shape

    patches = []
    positions = []

    for y in range(0, ny - patch_size + 1, stride):
        for x in range(0, nx - patch_size + 1, stride):
            patch = label_cube[y:y + patch_size, x:x + patch_size, :]
            patches.append(patch)
            positions.append((y, x))

    return np.array(patches), positions


def process_particle(
    params: Dict[str, Any],
    scattering_factors: ScatteringFactors,
    verbose: bool = False,
    use_overlap: bool = False,
    stride: int = 8
) -> Dict[str, Any]:
    """
    Process a single particle through the full workflow.

    Parameters
    ----------
    params : dict
        Particle parameters from sample_parameters()
    scattering_factors : ScatteringFactors
        Scattering factor lookup object
    verbose : bool
        Print debug info
    use_overlap : bool
        If True, extract overlapping patches with given stride.
        If False, extract non-overlapping patches (stride = patch_size).
    stride : int
        Step between patch origins when use_overlap=True.
        Typical: stride=8 for 50% overlap with patch_size=16.

    Returns
    -------
    dict
        Dictionary with patches, labels, and metadata.
    """

    shape_type = params['shape_type']

    # Build shape_params dict
    shape_params = {}
    if shape_type == 'hexagon':
        shape_params['anisotropy'] = params.get('anisotropy', 1.0)
        shape_params['rotation_angle'] = params.get('rotation_angle', 0.0)
    elif shape_type in ['polygon', 'polygon_centrosymmetric']:
        n_vert = params.get('n_vertices', 7)
        shape_params['min_corners'] = n_vert
        shape_params['max_corners'] = n_vert + 1
    elif shape_type == 'ellipse':
        shape_params['aspect_ratio'] = params.get('ellipse_aspect_ratio', 1.5)
        shape_params['rotation_angle'] = params.get('rotation_angle', 0.0)

    # -------------------------------------------------------------------------
    # 1. Create particle (with new composition and geometry options)
    # -------------------------------------------------------------------------

    particle, info = create_particle_with_shape(
        grid_size=GRID_SIZE_FFT,
        shape_type=shape_type,
        outer_radius=params['outer_radius'],
        core_fraction=params['core_fraction'],
        pixel_size=PIXEL_SIZE,
        shape_params=shape_params,
        verbose=False,
        # NEW: Composition mode and parameters
        composition_mode=params.get('composition_mode', 'sharp'),
        composition_params=params.get('composition_params', {}),
        # NEW: Geometry modifiers
        truncation_fraction=params.get('truncation_fraction', 0.0),
        truncation_angle=params.get('truncation_angle', 0.0),
        core_offset=params.get('core_offset', (0.0, 0.0)),
    )
    
    outer_mask = info['outer_mask']
    core_mask = info['core_mask']
    outer_vertices = info.get('outer_vertices', [])
    
    # -------------------------------------------------------------------------
    # 2. Create displacement field
    # -------------------------------------------------------------------------
    
    displacement_analytic, _ = create_layered_displacement_field(
        core_mask=core_mask,
        outer_mask=outer_mask,
        pixel_size=PIXEL_SIZE,
        interface_amplitude=params['interface_amplitude'],
        surface_amplitude=params['surface_amplitude'],
        corner_positions=outer_vertices,
        verbose=False
    )
    
    displacement_random = create_random_strain_field(
        grid_size=GRID_SIZE_FFT,
        pixel_size=PIXEL_SIZE,
        displacement_amplitude=params['random_amplitude'],
        correlation_length=params['random_correlation'],
        verbose=False
    )
    
    displacement = displacement_analytic + displacement_random
    
    # -------------------------------------------------------------------------
    # 3. Apply displacement to particle
    # -------------------------------------------------------------------------
    
    particle_strained = apply_displacement_to_particle(
        particle=particle,
        displacement=displacement,
        q_bragg_magnitude=Q_BRAGG
    )
    
    # -------------------------------------------------------------------------
    # 4. Optional: Add module noise
    # -------------------------------------------------------------------------
    
    if USE_MODULE_NOISE and 'module_noise_amplitude' in params:
        particle_noisy = add_correlated_module_noise(
            particle=particle_strained,
            noise_amplitude=params['module_noise_amplitude'],
            correlation_length=0.15,
            verbose=False
        )
    else:
        particle_noisy = particle_strained
    
    # -------------------------------------------------------------------------
    # 5. Compute multi-energy diffraction
    # -------------------------------------------------------------------------
    
    diffractions = compute_diffraction_oversampled_cropped(
        particle=particle_noisy,
        energies=ENERGIES,
        pixel_size=PIXEL_SIZE,
        scattering_factors=scattering_factors,
        output_size=GRID_SIZE_OUTPUT,
        verbose=False
    )
    
    # Build intensity cube
    intensity_list = []
    for E in ENERGIES:
        intensity = np.abs(diffractions[E])**2
        intensity_list.append(intensity)
    intensity_cube_clean = np.stack(intensity_list, axis=-1)
    
    # -------------------------------------------------------------------------
    # 6. Add Poisson noise
    # -------------------------------------------------------------------------
    
    if USE_POISSON_NOISE:
        intensity_cube = add_poisson_noise(
            intensity_cube_clean,
            max_counts=POISSON_MAX_COUNTS,
            verbose=False
        )
    else:
        intensity_cube = intensity_cube_clean
    
    # -------------------------------------------------------------------------
    # 7. Extract intensity patches
    # -------------------------------------------------------------------------

    if use_overlap:
        # Overlapping patches for boundary-aware training
        intensity_patches, positions = extract_intensity_patches_overlapping(
            intensity_cube, PATCH_SIZE, stride
        )
    else:
        # Non-overlapping patches (original behavior)
        intensity_patches = extract_intensity_patches(intensity_cube, PATCH_SIZE)
        # Reshape from (ny, nx, H, W, E) to (n_patches, H, W, E)
        n_patches_y, n_patches_x = intensity_patches.shape[:2]
        intensity_patches = intensity_patches.reshape(-1, PATCH_SIZE, PATCH_SIZE, N_ENERGIES)

    # -------------------------------------------------------------------------
    # 8. Compute ground truth labels
    # -------------------------------------------------------------------------

    labels = compute_ground_truth_labels(
        particle=particle_noisy,
        pixel_size=PIXEL_SIZE,
        output_size=GRID_SIZE_OUTPUT,
        verbose=False
    )

    if use_overlap:
        # Overlapping label patches (must match intensity patches)
        label_patches, _ = extract_label_patches_overlapping(
            labels=labels,
            patch_size=PATCH_SIZE,
            stride=stride,
            label_keys=['F_T_mag', 'F_A_mag', 'sin_delta_phi', 'cos_delta_phi']
        )
    else:
        # Non-overlapping patches (original behavior)
        label_patches = extract_label_patches(
            labels=labels,
            patch_size=PATCH_SIZE,
            label_keys=['F_T_mag', 'F_A_mag', 'sin_delta_phi', 'cos_delta_phi']
        )
        # Reshape from (ny, nx, H, W, 4) to (n_patches, H, W, 4)
        label_patches = label_patches.reshape(-1, PATCH_SIZE, PATCH_SIZE, 4)
    
    # -------------------------------------------------------------------------
    # 9. Extract f'/f''
    # -------------------------------------------------------------------------
    
    f_prime = np.array([scattering_factors.get_f_prime('Ni', E) for E in ENERGIES])
    f_double_prime = np.array([scattering_factors.get_f_double_prime('Ni', E) for E in ENERGIES])
    
    return {
        'X': intensity_patches.astype(np.float32),
        'Y': label_patches.astype(np.float32),
        'f_prime': f_prime.astype(np.float32),
        'f_double_prime': f_double_prime.astype(np.float32),
        'energies': np.array(ENERGIES, dtype=np.float32),
        'shape_type': shape_type,
        'params': params,
        'n_patches': len(intensity_patches),
    }


def save_particle_data(data: Dict, output_path: Path, particle_idx: int):
    """Save a single particle's data to an .npz file."""
    
    filename = output_path / f'particle_{particle_idx:04d}.npz'
    
    # Convert params dict to JSON string for storage
    params_json = json.dumps(data['params'])
    
    np.savez_compressed(
        filename,
        X=data['X'],
        Y=data['Y'],
        f_prime=data['f_prime'],
        f_double_prime=data['f_double_prime'],
        energies=data['energies'],
        shape_type=data['shape_type'],
        params_json=params_json
    )
    
    return filename


def create_metadata(
    output_path: Path,
    n_particles: int,
    total_patches: int,
    shape_counts: Dict[str, int],
    start_time: float,
    end_time: float,
    use_overlap: bool = False,
    stride: int = 16
) -> Dict:
    """Create and save metadata file."""

    metadata = {
        'dataset_name': output_path.name,
        'created_at': datetime.now().isoformat(),
        'generation_time_seconds': end_time - start_time,

        # Dataset size
        'n_particles': n_particles,
        'total_patches': total_patches,
        'patches_per_particle': total_patches // n_particles if n_particles > 0 else 0,

        # Data shapes
        'patch_shape_X': [PATCH_SIZE, PATCH_SIZE, N_ENERGIES],
        'patch_shape_Y': [PATCH_SIZE, PATCH_SIZE, 4],
        'label_channels': ['F_T_mag', 'F_A_mag', 'sin_delta_phi', 'cos_delta_phi'],

        # Patch extraction settings
        'use_overlap': use_overlap,
        'stride': stride if use_overlap else PATCH_SIZE,
        'overlap_fraction': 1 - stride / PATCH_SIZE if use_overlap else 0,

        # Physics parameters
        'grid_size_fft': GRID_SIZE_FFT,
        'grid_size_output': GRID_SIZE_OUTPUT,
        'pixel_size_angstrom': PIXEL_SIZE,
        'energies_eV': ENERGIES,
        'q_bragg_inv_angstrom': Q_BRAGG,

        # Shape distribution
        'shape_distribution_target': SHAPE_DISTRIBUTION,
        'shape_counts_actual': shape_counts,

        # Parameter ranges
        'parameter_ranges': PARAM_RANGES,

        # Noise settings
        'use_module_noise': USE_MODULE_NOISE,
        'use_poisson_noise': USE_POISSON_NOISE,
        'poisson_max_counts': POISSON_MAX_COUNTS,

        # Reproducibility
        'random_seed': RANDOM_SEED,
    }
    
    # Save metadata
    metadata_path = output_path / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================

def generate_dataset(
    output_dir: Path,
    n_particles: int,
    scattering_factors_dir: str = '.',
    verbose: bool = True,
    use_overlap: bool = False,
    stride: int = 8
):
    """
    Generate the complete training dataset.

    Parameters
    ----------
    output_dir : Path
        Directory to save output files
    n_particles : int
        Number of particles to generate
    scattering_factors_dir : str
        Directory containing Nickel.f1f2 and Iron.f1f2 files
    verbose : bool
        Print progress information
    use_overlap : bool
        If True, extract overlapping patches with given stride.
        This helps the CNN learn consistent predictions at patch boundaries.
    stride : int
        Step between patch origins when use_overlap=True.
        Default: 8 (50% overlap with patch_size=16).
        Smaller stride = more overlap = more patches per particle.
    """
    
    start_time = time.time()
    
    # =========================================================================
    # SETUP
    # =========================================================================
    
    print("\n" + "=" * 78)
    print("NANOMAD TRAINING DATA GENERATION")
    print("=" * 78)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Set random seed
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)
        print(f"Random seed: {RANDOM_SEED}")
    
    # Load scattering factors
    print(f"\nLoading scattering factors from: {scattering_factors_dir}")
    sf = ScatteringFactors(data_dir=scattering_factors_dir)
    
    # Print configuration
    print("\n" + "-" * 78)
    print("CONFIGURATION")
    print("-" * 78)
    print(f"  Particles to generate: {n_particles}")
    print(f"  FFT grid size: {GRID_SIZE_FFT} × {GRID_SIZE_FFT}")
    print(f"  Output grid size: {GRID_SIZE_OUTPUT} × {GRID_SIZE_OUTPUT}")
    print(f"  Patch size: {PATCH_SIZE} × {PATCH_SIZE}")
    if use_overlap:
        # Calculate expected patches with overlap
        n_patches_per_axis = (GRID_SIZE_OUTPUT - PATCH_SIZE) // stride + 1
        patches_per_particle = n_patches_per_axis ** 2
        print(f"  Overlapping patches: ON (stride={stride}, {100*(1 - stride/PATCH_SIZE):.0f}% overlap)")
        print(f"  Patches per particle: {patches_per_particle}")
    else:
        print(f"  Overlapping patches: OFF")
        print(f"  Patches per particle: {(GRID_SIZE_OUTPUT // PATCH_SIZE) ** 2}")
    print(f"  Energies: {N_ENERGIES} ({ENERGIES[0]} - {ENERGIES[-1]} eV)")
    print(f"  Module noise: {'ON' if USE_MODULE_NOISE else 'OFF'}")
    print(f"  Poisson noise: {'ON' if USE_POISSON_NOISE else 'OFF'}")
    
    print("\n  Shape distribution:")
    for shape, prob in SHAPE_DISTRIBUTION.items():
        print(f"    {shape}: {prob:.0%}")
    
    print("\n  Parameter ranges:")
    for param, (lo, hi) in PARAM_RANGES.items():
        print(f"    {param}: [{lo}, {hi}]")
    
    # =========================================================================
    # GENERATION LOOP
    # =========================================================================
    
    print("\n" + "-" * 78)
    print("GENERATING PARTICLES")
    print("-" * 78)
    
    total_patches = 0
    shape_counts = {shape: 0 for shape in SHAPE_DISTRIBUTION}
    failed_particles = []
    
    for i in range(n_particles):
        particle_start = time.time()
        
        try:
            # Sample parameters
            shape_type = sample_shape_type()
            params = sample_parameters(shape_type)
            
            # Process particle
            data = process_particle(
                params, sf, verbose=False,
                use_overlap=use_overlap, stride=stride
            )
            
            # Save to disk
            save_particle_data(data, output_dir, i)
            
            # Update counts
            total_patches += data['n_patches']
            shape_counts[shape_type] += 1
            
            # Progress output
            particle_time = time.time() - particle_start
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (n_particles - i - 1)
            
            if verbose:
                print(f"  [{i+1:4d}/{n_particles}] {shape_type:25s} "
                      f"| {data['n_patches']} patches "
                      f"| {particle_time:.2f}s "
                      f"| ETA: {eta/60:.1f} min")
        
        except Exception as e:
            print(f"  [{i+1:4d}/{n_particles}] ERROR: {e}")
            failed_particles.append((i, str(e)))
            continue
    
    # =========================================================================
    # SAVE METADATA
    # =========================================================================
    
    end_time = time.time()
    
    print("\n" + "-" * 78)
    print("SAVING METADATA")
    print("-" * 78)
    
    metadata = create_metadata(
        output_path=output_dir,
        n_particles=n_particles - len(failed_particles),
        total_patches=total_patches,
        shape_counts=shape_counts,
        start_time=start_time,
        end_time=end_time,
        use_overlap=use_overlap,
        stride=stride
    )
    
    print(f"  Metadata saved to: {output_dir / 'metadata.json'}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 78)
    print("GENERATION COMPLETE")
    print("=" * 78)
    
    print(f"\n  Total particles: {n_particles - len(failed_particles)}")
    print(f"  Failed particles: {len(failed_particles)}")
    print(f"  Total patches: {total_patches}")
    print(f"  Generation time: {(end_time - start_time)/60:.1f} minutes")
    print(f"  Average time per particle: {(end_time - start_time)/(n_particles - len(failed_particles)):.2f} seconds")
    
    print("\n  Shape distribution (actual):")
    for shape, count in shape_counts.items():
        pct = count / (n_particles - len(failed_particles)) * 100 if n_particles > len(failed_particles) else 0
        print(f"    {shape}: {count} ({pct:.1f}%)")
    
    # Estimate file sizes
    x_size = total_patches * PATCH_SIZE * PATCH_SIZE * N_ENERGIES * 4 / 1e9
    y_size = total_patches * PATCH_SIZE * PATCH_SIZE * 4 * 4 / 1e9
    print(f"\n  Estimated data size:")
    print(f"    X (intensities): ~{x_size:.2f} GB")
    print(f"    Y (labels): ~{y_size:.2f} GB")
    print(f"    Total: ~{x_size + y_size:.2f} GB")
    
    print(f"\n  Output directory: {output_dir}")
    
    if failed_particles:
        print(f"\n  Failed particles:")
        for idx, error in failed_particles[:10]:  # Show first 10
            print(f"    Particle {idx}: {error}")
        if len(failed_particles) > 10:
            print(f"    ... and {len(failed_particles) - 10} more")
    
    print("\n" + "=" * 78)
    print("Done!")
    print("=" * 78)
    
    return metadata


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate synthetic training data for MAD CNN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python generate_training_data.py                    # Use defaults
  python generate_training_data.py -n 500             # Generate 500 particles
  python generate_training_data.py -o /path/to/output # Custom output directory
  python generate_training_data.py --sf-dir /path/to/scattering_factors

  # Generate overlapping patches for boundary-aware training:
  python generate_training_data.py -n 500 --overlap --stride 8
        """
    )
    
    parser.add_argument(
        '-n', '--n-particles',
        type=int,
        default=N_PARTICLES,
        help=f'Number of particles to generate (default: {N_PARTICLES})'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default=str(OUTPUT_DIR),
        help=f'Output directory (default: {OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--sf-dir',
        type=str,
        default='.',
        help='Directory containing scattering factor files (default: current directory)'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=RANDOM_SEED,
        help=f'Random seed (default: {RANDOM_SEED})'
    )

    parser.add_argument(
        '--overlap',
        action='store_true',
        help='Extract overlapping patches for boundary-aware training'
    )

    parser.add_argument(
        '--stride',
        type=int,
        default=8,
        help='Stride for overlapping patches (default: 8 = 50%% overlap with 16x16 patches)'
    )

    args = parser.parse_args()
    
    # Update global seed if specified
    if args.seed is not None:
        RANDOM_SEED = args.seed
    
    # Run generation
    generate_dataset(
        output_dir=Path(args.output_dir),
        n_particles=args.n_particles,
        scattering_factors_dir=args.sf_dir,
        verbose=not args.quiet,
        use_overlap=args.overlap,
        stride=args.stride
    )
