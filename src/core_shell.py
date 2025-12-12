"""
core_shell.py - 2D Core-Shell Particle Simulation for BCDI (Composition-Resolved)

This module provides functions to:
1. Create 2D circular core-shell particles with SEPARATE species density maps
2. Compute diffraction patterns via FFT with FULL scattering factor physics
3. Reconstruct objects via inverse FFT (amplitude and phase separation)
4. Extract and process sub-blocks for ML training data preparation
5. Visualize all results

=============================================================================
KEY CONCEPT: COMPOSITION-RESOLVED DENSITY MAPS
=============================================================================

Instead of storing a single "effective electron density" value at each pixel,
we store SEPARATE density maps for each atomic species (Ni and Fe).

Why this matters:
-----------------
At different X-ray energies AND scattering angles, each element scatters 
differently. The full scattering factor for each element is:

    f(Q, E) = f₀(Q) + f'(E) + i·f''(E)

Where:
- f₀(Q): Thomson scattering - depends on scattering vector magnitude |Q|
         Parameterized using IT92 Gaussian coefficients from International Tables
         At Q=0: f₀(0) ≈ Z (atomic number)
         Decreases smoothly with increasing Q
         
- f'(E): Real part of anomalous/resonant scattering (energy-dependent)
         Changes dramatically near absorption edges
         Can be negative (reduces scattering below Thomson value)
         
- f''(E): Imaginary part (absorption, energy-dependent)
          Always positive
          Related to absorption cross-section
          Jumps up at absorption edges

Near an absorption edge (like Ni K-edge at ~8.333 keV), f' and f'' change
dramatically, providing elemental contrast.

By storing separate Ni and Fe maps, we can:
1. Apply Q- and E-dependent complex scattering factors
2. Sum contributions from each species to get the total scattering
3. Generate multi-energy datasets efficiently

Data Structure:
---------------
The particle is stored as a 3D array of shape (2, grid_size, grid_size):
    - Index 0: Nickel (Ni) density map
    - Index 1: Iron (Fe) density map

Physical units are tracked via pixel_size (in Ångströms), which allows
proper computation of the Q-grid for scattering factor calculations.

=============================================================================

Author: Claude (Anthropic) for Thomas's BCDI ML research
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional, List, Union
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve

# Import the IT92 Thomson scattering coefficients
# These are Gaussian parameterizations of f₀(Q) from the International Tables
# for Crystallography, Volume C (1992)
try:
    # When running from repository root
    from data.thomson_factors import d_fthomson_IT92
except ImportError:
    # Fallback for direct script execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.thomson_factors import d_fthomson_IT92


# =============================================================================
# CONSTANTS AND SPECIES DEFINITIONS
# =============================================================================

# Species indices in the composition array
# We use a fixed ordering so code can reliably access each species
SPECIES_NI = 0  # Nickel is at index 0
SPECIES_FE = 1  # Iron is at index 1
SPECIES_NAMES = ['Ni', 'Fe']  # For labeling and printing
N_SPECIES = 2   # Total number of species we track

# Default pixel size in Ångströms
# For a 128×128 grid, this gives a real-space extent of 1280 Å = 128 nm
# which is appropriate for 100-500 nm nanoparticles
DEFAULT_PIXEL_SIZE = 10.0  # Å

# Atomic numbers - used as fallback f₀ values when Q-dependence is disabled
# Note: With Q-dependent scattering enabled, we use IT92 coefficients instead
ATOMIC_NUMBER = {
    'Ni': 28,
    'Fe': 26
}


# =============================================================================
# Q-GRID AND THOMSON SCATTERING (f₀)
# =============================================================================

def compute_q_grid(grid_size: int, pixel_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Q-grid (reciprocal space coordinates) for an FFT.
    
    When you FFT a real-space image with pixel size Δx, the resulting
    reciprocal space has:
    - Q spacing: ΔQ = 2π / (N × Δx) = 2π / L  (where L is total size)
    - Q range: -Q_max to +Q_max where Q_max = π / Δx
    
    After fftshift, Q=0 is at the center of the array.
    
    Parameters
    ----------
    grid_size : int
        Size of the grid (N × N pixels)
        
    pixel_size : float
        Real-space pixel size in Ångströms
        
    Returns
    -------
    qx : np.ndarray
        2D array of Qx values (Å⁻¹), shape (grid_size, grid_size)
        
    qy : np.ndarray
        2D array of Qy values (Å⁻¹), shape (grid_size, grid_size)
        
    q_magnitude : np.ndarray
        2D array of |Q| = sqrt(Qx² + Qy²) values (Å⁻¹)
        
    Notes
    -----
    The Q-grid is centered at Q=0 (form factor geometry).
    For BCDI around a Bragg peak, you would add Q_Bragg to shift the center.
    
    Example
    -------
    >>> qx, qy, q_mag = compute_q_grid(128, 10.0)  # 128×128, 10 Å pixels
    >>> print(f"Q range: 0 to {q_mag.max():.3f} Å⁻¹")
    Q range: 0 to 0.444 Å⁻¹
    """
    
    # Compute the frequency array for the FFT
    # np.fft.fftfreq gives frequencies in cycles per pixel
    # Multiply by 2π/pixel_size to convert to Q in Å⁻¹
    
    freq = np.fft.fftfreq(grid_size)  # in cycles per pixel, range [-0.5, 0.5)
    
    # Convert to Q: Q = 2π × freq / pixel_size = 2π × (cycles/pixel) / (Å/pixel) = Å⁻¹
    # But fftfreq gives cycles per sample, so Q = 2π × freq / pixel_size
    q_1d = 2 * np.pi * freq / pixel_size  # Å⁻¹
    
    # Shift so Q=0 is at center (matching fftshift of diffraction pattern)
    q_1d_shifted = np.fft.fftshift(q_1d)
    
    # Create 2D grids
    qx, qy = np.meshgrid(q_1d_shifted, q_1d_shifted, indexing='xy')
    
    # Compute magnitude
    q_magnitude = np.sqrt(qx**2 + qy**2)
    
    return qx, qy, q_magnitude


def compute_f0_thomson(element: str, q_magnitude: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute the Thomson scattering factor f₀(Q) using IT92 Gaussian parameterization.
    
    The International Tables for Crystallography (1992) parameterize f₀ as:
    
        f₀(s) = Σᵢ aᵢ exp(-bᵢ s²) + c
        
    where s = sin(θ)/λ = |Q|/(4π).
    
    At s=0 (forward scattering, Q=0): f₀(0) = Σaᵢ + c ≈ Z (atomic number)
    As Q increases, f₀ decreases because electrons appear "smaller" at 
    larger scattering angles.
    
    Parameters
    ----------
    element : str
        Element symbol ('Ni', 'Fe', or any element in IT92 table)
        
    q_magnitude : float or np.ndarray
        Magnitude of scattering vector |Q| in Å⁻¹
        Can be a single value or an array (e.g., from compute_q_grid)
        
    Returns
    -------
    f0 : float or np.ndarray
        Thomson scattering factor f₀(Q)
        Same shape as input q_magnitude
        
    Raises
    ------
    ValueError
        If element is not in the IT92 table
        
    Example
    -------
    >>> f0_Ni_0 = compute_f0_thomson('Ni', 0.0)  # Forward scattering
    >>> print(f"f₀_Ni(Q=0) = {f0_Ni_0:.4f}")  # Should be ~28
    f₀_Ni(Q=0) = 27.9875
    
    >>> f0_Ni_3 = compute_f0_thomson('Ni', 3.0)  # At Q=3 Å⁻¹
    >>> print(f"f₀_Ni(Q=3) = {f0_Ni_3:.4f}")  # Lower due to Q-dependence
    f₀_Ni(Q=3) = 20.7862
    """
    
    if element not in d_fthomson_IT92:
        raise ValueError(
            f"Element '{element}' not found in IT92 table. "
            f"Available elements include: Ni, Fe, Co, Cu, ..."
        )
    
    # Get coefficients: ([a1, a2, a3, a4], [b1, b2, b3, b4], c)
    a_coeffs, b_coeffs, c = d_fthomson_IT92[element]
    a_coeffs = np.array(a_coeffs)
    b_coeffs = np.array(b_coeffs)
    
    # Convert Q to s: s = |Q| / (4π)
    s = np.asarray(q_magnitude) / (4 * np.pi)
    s_squared = s ** 2
    
    # Compute f₀(s) = Σᵢ aᵢ exp(-bᵢ s²) + c
    # Handle both scalar and array inputs
    if s_squared.ndim == 0:
        # Scalar input
        f0 = sum(a * np.exp(-b * s_squared) for a, b in zip(a_coeffs, b_coeffs)) + c
    else:
        # Array input - need to broadcast properly
        # Shape of s_squared: (Ny, Nx) or similar
        # Shape of a_coeffs, b_coeffs: (4,)
        # We want to sum over the 4 Gaussians for each pixel
        
        f0 = np.zeros_like(s_squared, dtype=float)
        for a, b in zip(a_coeffs, b_coeffs):
            f0 += a * np.exp(-b * s_squared)
        f0 += c
    
    return f0


# =============================================================================
# DISPLACEMENT FIELD AND STRAIN
# =============================================================================
#
# In BCDI, we measure a complex electron density:
#
#     ρ(r) = |ρ(r)| × exp(i·φ(r))
#
# The PHASE φ(r) encodes the displacement field u(r) via:
#
#     φ(r) = Q_Bragg · u(r)
#
# where:
#   - u(r) is the displacement vector (how much atoms have moved from ideal positions)
#   - Q_Bragg is the scattering vector of the Bragg reflection being measured
#   - The dot product projects the 3D displacement onto the measurement direction
#
# For a specific reflection (hkl), you're sensitive to displacements along Q_hkl.
#
# UNITS:
#   - Q_Bragg in Å⁻¹
#   - u(r) in Å
#   - φ(r) is dimensionless (radians)
#
# For this 2D simulation, we work with scalar displacement fields u_Q(r),
# representing the component of displacement along the Q direction.
# =============================================================================

# Lattice parameters for common materials (in Å)
LATTICE_PARAMS = {
    'Ni': 3.524,      # FCC Ni lattice parameter
    'Fe': 2.867,      # BCC Fe lattice parameter  
    'Ni3Fe': 3.545,   # Approximate for Ni₃Fe (slightly larger than pure Ni)
}


def compute_q_bragg(hkl: Tuple[int, int, int], lattice_param: float) -> float:
    """
    Compute the magnitude of the Bragg vector for a cubic crystal.
    
    For a cubic crystal with lattice parameter a, the d-spacing for (hkl) is:
    
        d_hkl = a / sqrt(h² + k² + l²)
    
    And the Bragg vector magnitude is:
    
        |Q_Bragg| = 2π / d_hkl = 2π × sqrt(h² + k² + l²) / a
    
    Parameters
    ----------
    hkl : tuple of (h, k, l)
        Miller indices of the reflection
        
    lattice_param : float
        Cubic lattice parameter in Å
        
    Returns
    -------
    q_bragg : float
        Magnitude of Q_Bragg in Å⁻¹
        
    Example
    -------
    >>> q_111 = compute_q_bragg((1, 1, 1), 3.524)  # Ni (111)
    >>> print(f"|Q_111| = {q_111:.4f} Å⁻¹")
    |Q_111| = 3.0880 Å⁻¹
    """
    
    h, k, l = hkl
    d_hkl = lattice_param / np.sqrt(h**2 + k**2 + l**2)
    q_bragg = 2 * np.pi / d_hkl
    
    return q_bragg


def create_displacement_field(
    grid_size: int,
    pixel_size: float,
    pattern: str = 'core_shell_mismatch',
    center: Tuple[float, float] = None,
    core_radius: float = None,
    outer_radius: float = None,
    smooth_transition: bool = True,
    transition_width: float = None,
    **kwargs
) -> np.ndarray:
    """
    Create a scalar displacement field u_Q(r) representing displacement along Q.
    
    The displacement field is used to compute the phase via φ(r) = Q_Bragg × u_Q(r).
    Different patterns simulate different physical situations.
    
    Parameters
    ----------
    grid_size : int
        Size of the grid (N × N pixels)
        
    pixel_size : float
        Pixel size in Å
        
    pattern : str
        Type of displacement pattern:
        - 'core_shell_mismatch': Lattice mismatch between core and shell
        - 'radial': Radial strain increasing from center
        - 'uniform': Uniform strain gradient along x
        - 'zero': No displacement (for testing)
        
    center : tuple, optional
        Center position (x, y) in pixels. Default is grid center.
        
    core_radius : float, optional
        Core radius in pixels (needed for 'core_shell_mismatch')
        
    outer_radius : float, optional
        Outer radius in pixels (needed for some patterns)
        
    smooth_transition : bool, optional
        If True (default), use a smooth (error function) transition between
        core and shell regions instead of a sharp boundary. This creates
        physically more realistic strain distributions.
        
    transition_width : float, optional
        Width of the smooth transition region in pixels.
        Default: 5% of core_radius, minimum 2 pixels.
        Only used if smooth_transition=True.
        
    **kwargs : dict
        Pattern-specific parameters:
        
        For 'core_shell_mismatch':
            - lattice_mismatch : float
                Relative lattice mismatch (a_core - a_shell) / a_shell
                Typical values: 0.001 to 0.02 (0.1% to 2%)
                Default: 0.006 (~0.6%, realistic for Ni₃Fe/Ni)
                
        For 'radial':
            - max_strain : float
                Maximum strain at outer radius. Default: 0.01 (1%)
                
        For 'uniform':
            - strain : float
                Uniform strain value. Default: 0.01 (1%)
        
    Returns
    -------
    u_Q : np.ndarray
        2D array of displacement values in Å, shape (grid_size, grid_size)
        This represents the displacement component along the Q direction.
        
    Notes
    -----
    The 'core_shell_mismatch' pattern is most relevant for Ni₃Fe particles:
    
    - The Ni₃Fe core has a slightly larger lattice parameter than pure Ni shell
    - This creates compressive strain in the core and tensile strain in the shell
    - The displacement field is approximately radial
    
    For Ni₃Fe (a ≈ 3.545 Å) surrounded by Ni shell (a ≈ 3.524 Å):
        mismatch = (3.545 - 3.524) / 3.524 ≈ 0.006 (0.6%)
    
    With smooth_transition=True, the strain transition at the interface is
    spread over a few nanometers, which is more physically realistic than
    an atomically sharp boundary.
    
    Example
    -------
    >>> u_Q = create_displacement_field(
    ...     grid_size=128, pixel_size=10.0,
    ...     pattern='core_shell_mismatch',
    ...     core_radius=30, outer_radius=50,
    ...     lattice_mismatch=0.006,
    ...     smooth_transition=True
    ... )
    >>> print(f"Max displacement: {np.abs(u_Q).max():.3f} Å")
    """
    from scipy.special import erf
    
    # -------------------------------------------------------------------------
    # Set defaults
    # -------------------------------------------------------------------------
    
    if center is None:
        center = (grid_size / 2, grid_size / 2)
    
    # -------------------------------------------------------------------------
    # Create coordinate arrays
    # -------------------------------------------------------------------------
    
    y_idx, x_idx = np.indices((grid_size, grid_size))
    cx, cy = center
    
    # Distance from center in pixels
    r_pixels = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2)
    
    # Distance from center in Å
    r_angstrom = r_pixels * pixel_size
    
    # -------------------------------------------------------------------------
    # Generate displacement field based on pattern
    # -------------------------------------------------------------------------
    
    if pattern == 'zero':
        # No displacement - useful for testing
        u_Q = np.zeros((grid_size, grid_size), dtype=float)
    
    elif pattern == 'core_shell_mismatch':
        # Lattice mismatch between core (Ni₃Fe) and shell (Ni)
        #
        # Physics:
        # - Core has larger lattice parameter than shell
        # - At the interface, both must match → strain
        # - Core is compressed, shell is stretched
        # - Displacement is approximately radial and proportional to r
        #
        # Elastic model:
        # - In the core: u(r) ≈ -ε_core × r (compression)
        # - In the shell: u(r) ≈ +ε_shell × (r - R_core) + u(R_core)
        # - The strain partitions based on elastic properties
        #
        # With smooth_transition=True, we use an error function to smoothly
        # interpolate between core and shell strain values at the interface.
        
        if core_radius is None:
            raise ValueError("core_radius required for 'core_shell_mismatch' pattern")
        if outer_radius is None:
            outer_radius = grid_size / 2
        
        # Get lattice mismatch (default: Ni₃Fe/Ni mismatch)
        lattice_mismatch = kwargs.get('lattice_mismatch', 0.006)
        
        # Convert radii to Å
        core_radius_angstrom = core_radius * pixel_size
        outer_radius_angstrom = outer_radius * pixel_size
        
        # Strain values in core and shell
        # Total mismatch is accommodated roughly 50/50
        strain_core = -lattice_mismatch / 2   # Negative = compression
        strain_shell = +lattice_mismatch / 2  # Positive = tension
        
        if smooth_transition:
            # Smooth transition using error function
            #
            # We define a smooth "shell fraction" that goes from 0 (pure core)
            # to 1 (pure shell) with a smooth transition at r = core_radius.
            #
            # shell_frac(r) = 0.5 * (1 + erf((r - R_core) / w))
            # where w controls the transition width
            
            if transition_width is None:
                # Default: 5% of core radius, minimum 2 pixels
                transition_width = max(2.0, 0.05 * core_radius)
            
            transition_width_angstrom = transition_width * pixel_size
            
            # Compute smooth shell fraction (0 in core center, 1 in outer shell)
            shell_frac = 0.5 * (1.0 + erf((r_angstrom - core_radius_angstrom) / transition_width_angstrom))
            
            # Compute local strain as weighted average of core and shell strain
            local_strain = (1.0 - shell_frac) * strain_core + shell_frac * strain_shell
            
            # Displacement is integral of strain: u(r) = ∫₀ʳ ε(r') dr'
            # For smoothly varying strain, we approximate:
            # u(r) ≈ strain_avg(r) × r
            # where strain_avg is the average strain from 0 to r
            #
            # More accurate: cumulative integral
            # But for simplicity and smoothness, we use:
            # u(r) = local_strain(r) × r × correction_factor
            #
            # Actually, let's compute it properly:
            # u(r) = ∫₀ʳ [(1-f(r'))×ε_core + f(r')×ε_shell] dr'
            #      = ε_core × r + (ε_shell - ε_core) × ∫₀ʳ f(r') dr'
            #
            # For f(r) = 0.5(1 + erf((r-R)/w)), the integral is complex.
            # Instead, use the displacement formulation directly:
            
            # In pure core region: u = ε_core × r
            # In pure shell region: u = u(R_core) + ε_shell × (r - R_core)
            # In transition: smooth interpolation
            
            # Displacement at the interface (pure core value)
            u_at_interface = strain_core * core_radius_angstrom
            
            # Core displacement contribution: ε_core × r (for all r)
            u_core_contrib = strain_core * r_angstrom
            
            # Shell displacement contribution: starts from interface
            # u_shell_contrib = ε_shell × (r - R_core) for r > R_core
            r_from_interface = np.maximum(r_angstrom - core_radius_angstrom, 0)
            u_shell_contrib = strain_shell * r_from_interface
            
            # Smoothly blend between core-only and core+shell contributions
            # In core (shell_frac ≈ 0): u = ε_core × r
            # In shell (shell_frac ≈ 1): u = ε_core × R_core + ε_shell × (r - R_core)
            #                           = u_at_interface + ε_shell × (r - R_core)
            
            # We can write: u = u_core_contrib + shell_frac × adjustment
            # where adjustment makes it match the shell formula
            #
            # For r > R_core:
            # shell formula: u_at_interface + ε_shell × (r - R_core)
            # core formula: ε_core × r
            # difference = u_at_interface + ε_shell × (r - R_core) - ε_core × r
            #            = ε_core × R_core + ε_shell × r - ε_shell × R_core - ε_core × r
            #            = (ε_shell - ε_core) × (r - R_core)
            
            adjustment = (strain_shell - strain_core) * r_from_interface
            
            u_Q = u_core_contrib + shell_frac * adjustment
            
        else:
            # Sharp transition (original implementation)
            # Initialize displacement field
            u_Q = np.zeros((grid_size, grid_size), dtype=float)
            
            # Masks for regions
            core_mask = r_pixels <= core_radius
            shell_mask = (r_pixels > core_radius) & (r_pixels <= outer_radius)
            
            # Core: displacement proportional to r, with core strain
            u_Q[core_mask] = strain_core * r_angstrom[core_mask]
            
            # Shell: displacement must be continuous at interface
            # u_shell(r) = u_core(R_core) + strain_shell × (r - R_core)
            u_at_interface = strain_core * core_radius_angstrom
            u_Q[shell_mask] = u_at_interface + strain_shell * (r_angstrom[shell_mask] - core_radius_angstrom)
    
    elif pattern == 'radial':
        # Simple radial strain field: ε(r) increases linearly from center
        # u(r) = ε(r) × r where ε(r) = max_strain × (r / r_max)
        # This gives u(r) ∝ r²
        
        max_strain = kwargs.get('max_strain', 0.01)
        
        if outer_radius is None:
            outer_radius = grid_size / 2
        
        r_max = outer_radius * pixel_size
        
        # Strain increases linearly with r
        epsilon_r = max_strain * (r_angstrom / r_max)
        
        # Displacement = strain × distance
        u_Q = epsilon_r * r_angstrom
        
        # Zero outside particle
        outside_mask = r_pixels > outer_radius
        u_Q[outside_mask] = 0.0
    
    elif pattern == 'uniform':
        # Uniform strain gradient along x direction
        # ε = du/dx = constant, so u = ε × (x - x_center)
        
        strain = kwargs.get('strain', 0.01)
        
        x_angstrom = (x_idx - cx) * pixel_size
        u_Q = strain * x_angstrom
        
        # Optionally zero outside particle
        if outer_radius is not None:
            outside_mask = r_pixels > outer_radius
            u_Q[outside_mask] = 0.0
    
    else:
        raise ValueError(
            f"Unknown displacement pattern '{pattern}'. "
            f"Available: 'zero', 'core_shell_mismatch', 'radial', 'uniform'"
        )
    
    return u_Q


# =============================================================================
# LAYERED DISPLACEMENT FIELD (Shape-Aware Strain)
# =============================================================================
#
# The basic create_displacement_field() uses radial coordinates, which produces
# circular strain patterns regardless of actual particle shape.
#
# This section provides a more realistic approach where displacement follows
# the actual geometry of the particle. Multiple strain sources can be layered:
#
#   u_total = u_interface + u_surface + u_random + ...
#            (enhanced by corner concentration factor)
#
# Strain sources:
# 1. Interface strain: Based on signed distance from core-shell boundary
# 2. Surface relaxation: Based on distance from outer particle boundary  
# 3. Corner enhancement: Multiplicative factor that intensifies strain near corners
# 4. Random strain: Correlated noise (already implemented elsewhere)
#
# =============================================================================

def compute_signed_distance_field(
    mask: np.ndarray,
    pixel_size: float = 1.0,
    verbose: bool = False
) -> np.ndarray:
    """
    Compute signed distance field from a binary mask boundary.
    
    The signed distance is:
    - Negative inside the mask (distance to nearest boundary, going inward)
    - Positive outside the mask (distance to nearest boundary, going outward)
    - Zero exactly on the boundary
    
    Parameters
    ----------
    mask : np.ndarray
        2D boolean array where True = inside the region
        
    pixel_size : float
        Pixel size for converting to physical units (default 1.0 = pixels)
        
    verbose : bool
        Print statistics
        
    Returns
    -------
    signed_distance : np.ndarray
        2D array of signed distances (same shape as mask).
        Units match pixel_size.
        
    Notes
    -----
    Uses scipy.ndimage.distance_transform_edt for efficient computation.
    The signed distance field is useful for creating displacement patterns
    that follow the actual shape of regions rather than radial patterns.
    """
    from scipy.ndimage import distance_transform_edt
    
    # Distance from boundary for points OUTSIDE the mask
    dist_outside = distance_transform_edt(~mask)
    
    # Distance from boundary for points INSIDE the mask
    dist_inside = distance_transform_edt(mask)
    
    # Signed distance: negative inside, positive outside
    signed_distance = dist_outside - dist_inside
    
    # Convert to physical units
    signed_distance = signed_distance * pixel_size
    
    if verbose:
        inside_vals = signed_distance[mask]
        outside_vals = signed_distance[~mask]
        print(f"    Signed distance field computed:")
        print(f"      Inside range: [{inside_vals.min():.1f}, {inside_vals.max():.1f}] (negative)")
        print(f"      Outside range: [{outside_vals.min():.1f}, {outside_vals.max():.1f}] (positive)")
    
    return signed_distance


def detect_shape_corners(
    mask: np.ndarray,
    corner_threshold: float = 0.3,
    min_distance: int = None,
    max_corners: int = 12,
    verbose: bool = False
) -> List[Tuple[int, int]]:
    """
    Detect corners (high curvature points) on the boundary of a shape.
    
    Uses the Harris corner detector on the mask boundary, with filtering
    to select only the most prominent corners.
    
    Parameters
    ----------
    mask : np.ndarray
        2D boolean array defining the shape
        
    corner_threshold : float
        Threshold for corner detection (relative to max response).
        Higher = fewer, more prominent corners.
        Default 0.3 (more selective than 0.1).
        
    min_distance : int, optional
        Minimum distance between detected corners in pixels.
        Default: 5% of grid size (ensures well-separated corners).
        
    max_corners : int
        Maximum number of corners to return.
        The most prominent corners are kept if more are detected.
        Default 12 (typical polygon won't have more).
        
    verbose : bool
        Print corner information
        
    Returns
    -------
    corners : list of (y, x) tuples
        Pixel coordinates of detected corners
    """
    from scipy.ndimage import gaussian_filter, maximum_filter
    
    grid_size = mask.shape[0]
    
    # Default min_distance based on grid size
    if min_distance is None:
        min_distance = max(grid_size // 20, 15)  # At least 15 pixels apart
    
    # Convert mask to float and find edges
    mask_float = mask.astype(float)
    
    # Compute gradients
    gy, gx = np.gradient(mask_float)
    
    # Harris corner detector components
    Ixx = gaussian_filter(gx * gx, sigma=2)
    Iyy = gaussian_filter(gy * gy, sigma=2)
    Ixy = gaussian_filter(gx * gy, sigma=2)
    
    # Harris response: det(M) - k * trace(M)^2
    k = 0.04
    det = Ixx * Iyy - Ixy * Ixy
    trace = Ixx + Iyy
    harris = det - k * trace * trace
    
    # Threshold
    threshold = corner_threshold * harris.max()
    
    # Non-maximum suppression
    harris_max = maximum_filter(harris, size=min_distance)
    corners_mask = (harris == harris_max) & (harris > threshold)
    
    # Get coordinates and sort by response strength
    corner_coords = np.argwhere(corners_mask)
    if len(corner_coords) == 0:
        corners = []
    else:
        # Get Harris response at each corner for ranking
        corner_responses = [harris[c[0], c[1]] for c in corner_coords]
        
        # Sort by response (strongest first) and keep top max_corners
        sorted_indices = np.argsort(corner_responses)[::-1]  # Descending
        top_indices = sorted_indices[:max_corners]
        
        corners = [(int(corner_coords[i][0]), int(corner_coords[i][1])) for i in top_indices]
    
    if verbose:
        print(f"    Detected {len(corners)} corners (max {max_corners}, min_dist={min_distance}px)")
        if len(corners) > 0 and len(corners) <= 12:
            for i, (y, x) in enumerate(corners):
                print(f"      Corner {i+1}: ({y}, {x})")
    
    return corners


def compute_corner_enhancement_field(
    grid_size: int,
    corners: List[Tuple[int, int]],
    enhancement_amplitude: float = 0.5,
    enhancement_width: float = 20.0,
    verbose: bool = False
) -> np.ndarray:
    """
    Compute a multiplicative enhancement field that intensifies near corners.
    
    The enhancement factor is:
        factor = 1 + amplitude × Σ exp(-distance_to_corner² / (2×width²))
    
    This creates Gaussian "hot spots" around each corner that multiply
    (enhance) the existing strain field.
    
    Parameters
    ----------
    grid_size : int
        Size of the output array (grid_size × grid_size)
        
    corners : list of (y, x) tuples
        Corner positions in pixel coordinates
        
    enhancement_amplitude : float
        Maximum additional enhancement at corner center.
        E.g., 0.5 means strain is enhanced by up to 50% at corners.
        
    enhancement_width : float
        Width (sigma) of the Gaussian enhancement in pixels.
        Larger = broader enhancement region.
        
    verbose : bool
        Print field statistics
        
    Returns
    -------
    enhancement : np.ndarray
        2D array of shape (grid_size, grid_size).
        Values ≥ 1.0 everywhere (1.0 = no enhancement).
        Use as: u_enhanced = u_base × enhancement
    """
    
    y_coords, x_coords = np.indices((grid_size, grid_size))
    
    # Start with no enhancement (factor = 1)
    enhancement = np.ones((grid_size, grid_size), dtype=float)
    
    if len(corners) == 0:
        if verbose:
            print(f"    No corners provided, enhancement field = 1.0 everywhere")
        return enhancement
    
    # Add Gaussian contribution from each corner
    for (cy, cx) in corners:
        distance_sq = (y_coords - cy)**2 + (x_coords - cx)**2
        gaussian = np.exp(-distance_sq / (2 * enhancement_width**2))
        enhancement += enhancement_amplitude * gaussian
    
    if verbose:
        print(f"    Corner enhancement field computed:")
        print(f"      Number of corners: {len(corners)}")
        print(f"      Enhancement amplitude: {enhancement_amplitude:.2f}")
        print(f"      Enhancement width: {enhancement_width:.1f} pixels")
        print(f"      Enhancement range: [{enhancement.min():.3f}, {enhancement.max():.3f}]")
    
    return enhancement


def compute_surface_relaxation_field(
    outer_mask: np.ndarray,
    pixel_size: float,
    relaxation_depth: float = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Compute surface relaxation displacement field.
    
    Atoms near the surface relax due to missing bonds. This creates
    a displacement that decays exponentially going inward from the surface.
    
    Parameters
    ----------
    outer_mask : np.ndarray
        2D boolean array where True = inside particle
        
    pixel_size : float
        Pixel size in Ångströms
        
    relaxation_depth : float, optional
        Characteristic decay length in Ångströms.
        Default: 50 Å (about 5 nm, a few unit cells)
        
    verbose : bool
        Print field statistics
        
    Returns
    -------
    surface_displacement : np.ndarray
        2D array of displacement values in Ångströms.
        Maximum at surface, decaying inward.
        Zero outside particle.
    """
    from scipy.ndimage import distance_transform_edt
    
    if relaxation_depth is None:
        relaxation_depth = 50.0  # Å, about 5 nm
    
    # Distance from surface (going inward)
    dist_from_surface = distance_transform_edt(outer_mask) * pixel_size
    
    # Exponential decay from surface
    # u_surface ∝ exp(-distance / relaxation_depth)
    surface_displacement = np.exp(-dist_from_surface / relaxation_depth)
    
    # Zero outside particle
    surface_displacement[~outer_mask] = 0.0
    
    # Normalize so maximum is 1.0 (will be scaled by amplitude later)
    if surface_displacement.max() > 0:
        surface_displacement = surface_displacement / surface_displacement.max()
    
    if verbose:
        inside_vals = surface_displacement[outer_mask]
        print(f"    Surface relaxation field computed:")
        print(f"      Relaxation depth: {relaxation_depth:.1f} Å")
        print(f"      Field range (inside): [{inside_vals.min():.3f}, {inside_vals.max():.3f}]")
    
    return surface_displacement


def create_layered_displacement_field(
    core_mask: np.ndarray,
    outer_mask: np.ndarray,
    pixel_size: float,
    lattice_mismatch: float = 0.006,
    interface_amplitude: float = None,
    surface_amplitude: float = 0.2,
    surface_relaxation_depth: float = 50.0,
    corner_enhancement: float = 0.3,
    corner_width: float = None,
    corner_positions: List[Tuple[int, int]] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Create a physically realistic displacement field with multiple strain layers.
    
    This produces displacement that follows the actual particle geometry rather
    than simple radial patterns. Multiple strain sources are combined:
    
    1. Interface strain: From core-shell lattice mismatch, following core boundary
    2. Surface relaxation: From missing bonds at particle surface
    3. Corner enhancement: Multiplicative factor intensifying strain at corners
    
    The total displacement is:
        u_total = (u_interface + u_surface) × corner_enhancement
    
    Parameters
    ----------
    core_mask : np.ndarray
        2D boolean array where True = inside core region
        
    outer_mask : np.ndarray
        2D boolean array where True = inside particle (core + shell)
        
    pixel_size : float
        Pixel size in Ångströms
        
    lattice_mismatch : float
        Relative lattice mismatch between core and shell.
        Default 0.006 (0.6%) for Ni₃Fe/Ni.
        
    interface_amplitude : float, optional
        Amplitude scaling for interface displacement.
        If None, computed from lattice_mismatch and geometry.
        
    surface_amplitude : float
        Amplitude of surface relaxation in Ångströms.
        Default 0.2 Å.
        
    surface_relaxation_depth : float
        Decay length for surface relaxation in Ångströms.
        Default 50 Å.
        
    corner_enhancement : float
        How much to enhance strain at corners (multiplicative).
        E.g., 0.3 means up to 30% enhancement at corners.
        Set to 0 to disable.
        
    corner_width : float, optional
        Width of corner enhancement region in pixels.
        Default: 10% of grid size.
        
    corner_positions : list, optional
        Vertex positions as [(y1, x1), (y2, x2), ...] in pixel coordinates.
        These should come from particle creation (info['outer_vertices']).
        If None and corner_enhancement > 0, no corner enhancement is applied.
        
    verbose : bool
        Print detailed information about each layer.
        
    Returns
    -------
    displacement : np.ndarray
        2D array of total displacement in Ångströms.
        
    info : dict
        Dictionary containing intermediate fields and parameters for debugging:
        - 'u_interface': Interface displacement component
        - 'u_surface': Surface displacement component
        - 'enhancement': Corner enhancement field
        - 'corners': Corner positions used
        - Various amplitude and parameter values
        
    Example
    -------
    >>> particle, pinfo = create_particle_with_shape(shape_type='hexagon', ...)
    >>> core_mask = pinfo['core_mask']
    >>> outer_mask = pinfo['outer_mask']
    >>> corners = pinfo['outer_vertices']  # Known vertex positions!
    >>> 
    >>> displacement, dinfo = create_layered_displacement_field(
    ...     core_mask, outer_mask, pixel_size=10.0,
    ...     lattice_mismatch=0.006,
    ...     corner_enhancement=0.3,
    ...     corner_positions=corners
    ... )
    """
    
    grid_size = core_mask.shape[0]
    
    if verbose:
        print(f"\n  Creating layered displacement field (shape-aware)...")
        print(f"    Grid size: {grid_size}×{grid_size}")
        print(f"    Pixel size: {pixel_size} Å")
    
    # -------------------------------------------------------------------------
    # Layer 1: Interface displacement (core-shell mismatch)
    # -------------------------------------------------------------------------
    #
    # Displacement is based on signed distance from core boundary.
    # - Inside core: negative distance → compression (negative displacement)
    # - In shell: positive distance → tension (positive displacement)
    #
    # The magnitude is scaled by lattice mismatch.
    
    if verbose:
        print(f"\n    Layer 1: Interface strain (core-shell mismatch)")
        print(f"      Lattice mismatch: {lattice_mismatch*100:.2f}%")
    
    # Signed distance from core boundary (negative inside, positive outside)
    signed_dist_core = compute_signed_distance_field(core_mask, pixel_size=pixel_size, verbose=verbose)
    
    # Compute interface amplitude if not provided
    # The displacement at the interface should accommodate the mismatch
    # For strain ε and characteristic length L: u ~ ε × L
    if interface_amplitude is None:
        # Use mean distance from center to interface as characteristic length
        core_distances = -signed_dist_core[core_mask]  # Positive values inside core
        if len(core_distances) > 0:
            characteristic_length = core_distances.mean()
        else:
            characteristic_length = grid_size * pixel_size / 4
        interface_amplitude = lattice_mismatch * characteristic_length
    
    # Scale signed distance to get displacement
    # The strain is approximately: ε = du/dr ≈ interface_amplitude / characteristic_length
    # So: u = ε × signed_distance (approximately)
    u_interface = (lattice_mismatch / 2) * signed_dist_core
    
    # Only apply within the particle
    u_interface[~outer_mask] = 0.0
    
    if verbose:
        inside_vals = u_interface[outer_mask]
        print(f"      Interface displacement range: [{inside_vals.min():.3f}, {inside_vals.max():.3f}] Å")
    
    # -------------------------------------------------------------------------
    # Layer 2: Surface relaxation
    # -------------------------------------------------------------------------
    
    if surface_amplitude > 0:
        if verbose:
            print(f"\n    Layer 2: Surface relaxation")
            print(f"      Amplitude: {surface_amplitude:.2f} Å")
        
        surface_field = compute_surface_relaxation_field(
            outer_mask, pixel_size, 
            relaxation_depth=surface_relaxation_depth,
            verbose=verbose
        )
        u_surface = surface_amplitude * surface_field
    else:
        u_surface = np.zeros_like(u_interface)
        if verbose:
            print(f"\n    Layer 2: Surface relaxation (DISABLED)")
    
    # -------------------------------------------------------------------------
    # Corner detection and enhancement
    # -------------------------------------------------------------------------
    
    if corner_enhancement > 0:
        if verbose:
            print(f"\n    Corner enhancement (multiplicative)")
            print(f"      Enhancement amplitude: {corner_enhancement:.2f}")
        
        # Use provided corner positions (from particle creation)
        corners = corner_positions if corner_positions else []
        
        if len(corners) == 0:
            if verbose:
                print(f"      No corner positions provided - skipping corner enhancement")
            enhancement = np.ones((grid_size, grid_size))
        else:
            if corner_width is None:
                corner_width = grid_size * 0.1  # 10% of grid size
            
            if verbose:
                print(f"      Using {len(corners)} provided vertices")
            
            enhancement = compute_corner_enhancement_field(
                grid_size, corners,
                enhancement_amplitude=corner_enhancement,
                enhancement_width=corner_width,
                verbose=verbose
            )
    else:
        corners = []
        enhancement = np.ones((grid_size, grid_size))
        if verbose:
            print(f"\n    Corner enhancement (DISABLED)")
    
    # -------------------------------------------------------------------------
    # Combine layers
    # -------------------------------------------------------------------------
    #
    # Total displacement = (interface + surface) × corner_enhancement
    #
    # The corner enhancement is multiplicative - it intensifies existing strain
    # at corners rather than adding a separate displacement.
    
    u_base = u_interface + u_surface
    displacement = u_base * enhancement
    
    # Ensure zero outside particle
    displacement[~outer_mask] = 0.0
    
    if verbose:
        inside_vals = displacement[outer_mask]
        print(f"\n    Combined displacement field:")
        print(f"      Total range: [{inside_vals.min():.3f}, {inside_vals.max():.3f}] Å")
        print(f"      RMS displacement: {np.std(inside_vals):.3f} Å")
    
    # -------------------------------------------------------------------------
    # Return displacement and debug info
    # -------------------------------------------------------------------------
    
    info = {
        'u_interface': u_interface,
        'u_surface': u_surface,
        'enhancement': enhancement,
        'signed_dist_core': signed_dist_core,
        'corners': corners,
        'lattice_mismatch': lattice_mismatch,
        'interface_amplitude': interface_amplitude,
        'surface_amplitude': surface_amplitude,
        'surface_relaxation_depth': surface_relaxation_depth,
        'corner_enhancement': corner_enhancement,
        'corner_width': corner_width,
    }
    
    return displacement, info


def apply_displacement_to_particle(
    particle: np.ndarray,
    displacement: np.ndarray,
    q_bragg_magnitude: float
) -> np.ndarray:
    """
    Apply a displacement field to a particle, adding phase.
    
    This converts a real-valued particle to a complex-valued particle
    with phase encoding the displacement:
    
        ρ_complex(r) = ρ(r) × exp(i × φ(r))
        
    where φ(r) = Q_Bragg × u_Q(r).
    
    Parameters
    ----------
    particle : np.ndarray
        Real-valued particle array. Can be:
        - 2D array (Ny, Nx): single density map
        - 3D array (N_species, Ny, Nx): composition-resolved particle
        
    displacement : np.ndarray
        2D array of displacement values u_Q(r) in Å, shape (Ny, Nx)
        This is the displacement component along the Q direction.
        
    q_bragg_magnitude : float
        Magnitude of the Bragg vector |Q_Bragg| in Å⁻¹
        
    Returns
    -------
    particle_with_phase : np.ndarray
        Complex-valued particle with phase = Q_Bragg × displacement.
        Same shape as input particle.
        
    Notes
    -----
    The phase is the same for all species (Ni and Fe) because displacement
    is a property of the lattice position, not the atom type.
    
    For multi-energy calculations, apply this BEFORE computing FFTs.
    The phase does NOT depend on energy, only on geometry.
    
    Example
    -------
    >>> particle, info = create_core_shell_particle(...)
    >>> u_Q = create_displacement_field(...)
    >>> q_bragg = compute_q_bragg((1, 1, 1), LATTICE_PARAMS['Ni'])
    >>> particle_strained = apply_displacement_to_particle(particle, u_Q, q_bragg)
    >>> # Now use with any diffraction function:
    >>> diff = compute_diffraction(particle_strained)
    """
    
    # -------------------------------------------------------------------------
    # Compute phase field: φ(r) = Q_Bragg × u_Q(r)
    # Units: Å⁻¹ × Å = dimensionless (radians)
    # -------------------------------------------------------------------------
    
    phase = q_bragg_magnitude * displacement
    
    # -------------------------------------------------------------------------
    # Create phase factor: exp(i × φ)
    # -------------------------------------------------------------------------
    
    phase_factor = np.exp(1j * phase)
    
    # -------------------------------------------------------------------------
    # Apply phase to particle
    # -------------------------------------------------------------------------
    
    if particle.ndim == 2:
        # Single 2D density map
        particle_with_phase = particle * phase_factor
        
    elif particle.ndim == 3:
        # Composition-resolved: (N_species, Ny, Nx)
        # The same phase applies to all species
        # Broadcasting: phase_factor[None, :, :] gives shape (1, Ny, Nx)
        particle_with_phase = particle * phase_factor[None, :, :]
        
    else:
        raise ValueError(
            f"particle must be 2D or 3D, got shape {particle.shape}"
        )
    
    return particle_with_phase


def get_displacement_info(
    displacement: np.ndarray,
    q_bragg_magnitude: float,
    pixel_size: float
) -> Dict[str, float]:
    """
    Compute summary statistics for a displacement field.
    
    Parameters
    ----------
    displacement : np.ndarray
        2D displacement field in Å
        
    q_bragg_magnitude : float
        |Q_Bragg| in Å⁻¹
        
    pixel_size : float
        Pixel size in Å (for strain calculation)
        
    Returns
    -------
    info : dict
        Dictionary with displacement and phase statistics
    """
    
    # Displacement statistics
    u_max = np.abs(displacement).max()
    u_mean = np.abs(displacement).mean()
    u_rms = np.sqrt(np.mean(displacement**2))
    
    # Phase statistics (φ = Q × u)
    phase = q_bragg_magnitude * displacement
    phase_max = np.abs(phase).max()
    phase_range = phase.max() - phase.min()
    
    # Estimate strain from displacement gradient
    # ε ≈ du/dr ≈ Δu / Δr
    grad_y, grad_x = np.gradient(displacement, pixel_size)
    strain_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    strain_max = strain_magnitude.max()
    strain_mean = strain_magnitude.mean()
    
    return {
        'displacement_max_angstrom': u_max,
        'displacement_mean_angstrom': u_mean,
        'displacement_rms_angstrom': u_rms,
        'phase_max_radians': phase_max,
        'phase_range_radians': phase_range,
        'phase_range_pi': phase_range / np.pi,
        'strain_max': strain_max,
        'strain_mean': strain_mean,
    }


def print_displacement_info(
    displacement: np.ndarray,
    q_bragg_magnitude: float,
    pixel_size: float
) -> None:
    """Print summary of displacement field statistics."""
    
    info = get_displacement_info(displacement, q_bragg_magnitude, pixel_size)
    
    print("=" * 60)
    print("DISPLACEMENT FIELD STATISTICS")
    print("=" * 60)
    print(f"Displacement (|u_Q|):")
    print(f"  Max:  {info['displacement_max_angstrom']:.4f} Å")
    print(f"  Mean: {info['displacement_mean_angstrom']:.4f} Å")
    print(f"  RMS:  {info['displacement_rms_angstrom']:.4f} Å")
    print("-" * 60)
    print(f"Phase (φ = Q × u, with |Q| = {q_bragg_magnitude:.4f} Å⁻¹):")
    print(f"  Max:   {info['phase_max_radians']:.4f} rad ({info['phase_max_radians']/np.pi:.3f}π)")
    print(f"  Range: {info['phase_range_radians']:.4f} rad ({info['phase_range_pi']:.3f}π)")
    print("-" * 60)
    print(f"Strain (estimated from gradient):")
    print(f"  Max:  {info['strain_max']:.6f} ({info['strain_max']*100:.4f}%)")
    print(f"  Mean: {info['strain_mean']:.6f} ({info['strain_mean']*100:.4f}%)")
    print("=" * 60)


# =============================================================================
# SCATTERING FACTOR DATA LOADING AND INTERPOLATION
# =============================================================================

class ScatteringFactors:
    """
    Class to load and interpolate f'(E) and f''(E) scattering factor data.
    
    This class handles:
    1. Loading tabulated f' and f'' data from files (Henke/Sasaki format)
    2. Interpolating to get values at any energy within the data range
    3. Computing the full complex scattering factor f(E) = f₀ + f'(E) + i·f''(E)
    
    Physics Background:
    -------------------
    The atomic scattering factor describes how strongly an atom scatters X-rays.
    It has three components:
    
        f(Q, E) = f₀(Q) + f'(E) + i·f''(E)
    
    - f₀(Q): Thomson scattering, depends on scattering vector Q
             At Q=0 (forward scattering), f₀(0) = Z (atomic number)
             We use this approximation: f₀ ≈ Z (constant)
             
    - f'(E): Real part of anomalous scattering
             Changes dramatically near absorption edges
             Can be negative (reduces scattering below Thomson value)
             
    - f''(E): Imaginary part (absorption)
              Always positive
              Related to absorption cross-section
              Jumps up at absorption edges
    
    Near an absorption edge, f' typically shows a sharp negative dip while
    f'' shows a sharp increase (the "white line").
    
    File Format:
    ------------
    Expected format (Demeter/Ifeffit style):
        # Comment lines start with #
        # energy    f'        f''
        2000.0000  -0.119    5.527
        2005.0000  -0.115    5.505
        ...
    
    Energies in eV, f' and f'' are dimensionless.
    """
    
    def __init__(self, data_dir: Union[str, Path] = None):
        """
        Initialize the ScatteringFactors object.

        Parameters
        ----------
        data_dir : str or Path, optional
            Directory containing the .f1f2 data files.
            If None, auto-detects from common locations:
            - data/ (when running from repo root)
            - ../data/ (when running from src/ or notebooks/)
            - . (current directory, fallback)
            Expected files: Nickel.f1f2, Iron.f1f2
        """

        if data_dir is None:
            # Auto-detect data directory by looking for Nickel.f1f2
            candidates = [
                Path("data"),           # Running from repo root
                Path("../data"),        # Running from src/ or notebooks/
                Path("."),              # Current directory (fallback)
            ]
            data_dir = None
            for candidate in candidates:
                if (candidate / "Nickel.f1f2").exists():
                    data_dir = candidate
                    break
            if data_dir is None:
                # Default to 'data' and let it fail later with clear error
                data_dir = Path("data")
        else:
            data_dir = Path(data_dir)
        
        self.data_dir = data_dir
        
        # Storage for loaded data and interpolators
        # Keys are element names ('Ni', 'Fe')
        self._energy = {}      # Energy arrays
        self._f_prime = {}     # f' arrays
        self._f_double_prime = {}  # f'' arrays
        self._interp_f_prime = {}  # Interpolation functions for f'
        self._interp_f_double_prime = {}  # Interpolation functions for f''
        
        # Track which elements have been loaded
        self._loaded_elements = set()
        
        # Try to load data files on initialization
        self._load_data_files()
    
    def _load_data_files(self):
        """
        Load scattering factor data from files.
        
        Looks for files named: Nickel.f1f2, Iron.f1f2
        """
        
        # Map element symbols to file names
        element_files = {
            'Ni': 'Nickel.f1f2',
            'Fe': 'Iron.f1f2'
        }
        
        for element, filename in element_files.items():
            filepath = self.data_dir / filename
            
            if filepath.exists():
                self._load_single_file(element, filepath)
            else:
                print(f"Warning: Could not find {filepath}")
                print(f"         Scattering factors for {element} will not be available.")
    
    def _load_single_file(self, element: str, filepath: Path):
        """
        Load f' and f'' data from a single file.
        
        Parameters
        ----------
        element : str
            Element symbol ('Ni' or 'Fe')
        filepath : Path
            Path to the data file
        """
        
        print(f"Loading scattering factors for {element} from {filepath}...")
        
        # Read the file, skipping comment lines
        energies = []
        f_primes = []
        f_double_primes = []
        
        with open(filepath, 'r') as f:
            for line in f:
                # Skip comment lines
                if line.strip().startswith('#'):
                    continue
                
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Parse data line
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        energy = float(parts[0])
                        f_prime = float(parts[1])
                        f_double_prime = float(parts[2])
                        
                        energies.append(energy)
                        f_primes.append(f_prime)
                        f_double_primes.append(f_double_prime)
                    except ValueError:
                        # Skip lines that can't be parsed
                        continue
        
        # Convert to numpy arrays
        self._energy[element] = np.array(energies)
        self._f_prime[element] = np.array(f_primes)
        self._f_double_prime[element] = np.array(f_double_primes)
        
        # Create interpolation functions
        # Using linear interpolation (could use cubic for smoother results)
        self._interp_f_prime[element] = interp1d(
            self._energy[element], 
            self._f_prime[element],
            kind='linear',
            bounds_error=True  # Raise error if outside data range
        )
        
        self._interp_f_double_prime[element] = interp1d(
            self._energy[element],
            self._f_double_prime[element],
            kind='linear',
            bounds_error=True
        )
        
        self._loaded_elements.add(element)
        
        # Print summary
        e_min = self._energy[element].min()
        e_max = self._energy[element].max()
        n_points = len(self._energy[element])
        print(f"  Loaded {n_points} data points")
        print(f"  Energy range: {e_min:.1f} - {e_max:.1f} eV")
    
    def get_f_prime(self, element: str, energy: float) -> float:
        """
        Get f'(E) for an element at a specific energy.
        
        Parameters
        ----------
        element : str
            Element symbol ('Ni' or 'Fe')
        energy : float
            X-ray energy in eV
            
        Returns
        -------
        f_prime : float
            The f' value at this energy
        """
        
        if element not in self._loaded_elements:
            raise ValueError(f"No data loaded for element '{element}'")
        
        return float(self._interp_f_prime[element](energy))
    
    def get_f_double_prime(self, element: str, energy: float) -> float:
        """
        Get f''(E) for an element at a specific energy.
        
        Parameters
        ----------
        element : str
            Element symbol ('Ni' or 'Fe')
        energy : float
            X-ray energy in eV
            
        Returns
        -------
        f_double_prime : float
            The f'' value at this energy
        """
        
        if element not in self._loaded_elements:
            raise ValueError(f"No data loaded for element '{element}'")
        
        return float(self._interp_f_double_prime[element](energy))
    
    def get_scattering_factor(self, element: str, energy: float) -> complex:
        """
        Get the full complex scattering factor f(E) = f₀ + f'(E) + i·f''(E).
        
        This version uses constant f₀ ≈ Z (atomic number), ignoring Q-dependence.
        For Q-dependent scattering, use get_scattering_factor_q() instead.
        
        Parameters
        ----------
        element : str
            Element symbol ('Ni' or 'Fe')
        energy : float
            X-ray energy in eV
            
        Returns
        -------
        f : complex
            The complex scattering factor f = f₀ + f' + i·f''
            
        Example
        -------
        >>> sf = ScatteringFactors()
        >>> f_Ni = sf.get_scattering_factor('Ni', 8333)  # At Ni K-edge
        >>> print(f"f_Ni = {f_Ni.real:.2f} + {f_Ni.imag:.2f}i")
        f_Ni = 20.00 + 3.81i
        """
        
        # Get the three components
        f0 = ATOMIC_NUMBER[element]  # Approximation: f₀ ≈ Z
        f_prime = self.get_f_prime(element, energy)
        f_double_prime = self.get_f_double_prime(element, energy)
        
        # Combine into complex scattering factor
        # f = f₀ + f' + i·f''
        f = complex(f0 + f_prime, f_double_prime)
        
        return f
    
    def get_scattering_factor_q(
        self, 
        element: str, 
        energy: float, 
        q_magnitude: np.ndarray
    ) -> np.ndarray:
        """
        Get the full Q-dependent complex scattering factor f(Q, E).
        
        This is the COMPLETE scattering factor with proper Q-dependence:
        
            f(Q, E) = f₀(Q) + f'(E) + i·f''(E)
        
        Where f₀(Q) is computed using IT92 Gaussian parameterization, which
        correctly accounts for the decrease in scattering at higher Q values.
        
        Parameters
        ----------
        element : str
            Element symbol ('Ni' or 'Fe')
            
        energy : float
            X-ray energy in eV
            
        q_magnitude : np.ndarray
            2D array of |Q| values in Å⁻¹ (from compute_q_grid)
            
        Returns
        -------
        f : np.ndarray
            2D complex array of scattering factors, same shape as q_magnitude.
            f[i,j] = f₀(Q[i,j]) + f'(E) + i·f''(E)
            
        Example
        -------
        >>> sf = ScatteringFactors()
        >>> qx, qy, q_mag = compute_q_grid(128, 10.0)
        >>> f_Ni = sf.get_scattering_factor_q('Ni', 8333, q_mag)
        >>> print(f"f_Ni at center: {f_Ni[64,64]:.2f}")  # Q=0
        >>> print(f"f_Ni at edge: {f_Ni[0,64]:.2f}")     # Q=Q_max
        
        Notes
        -----
        The returned array has the same shape as q_magnitude, with each pixel
        containing the complex scattering factor appropriate for that Q value.
        
        At Q=0 (center), f₀ ≈ Z, so results match get_scattering_factor().
        At higher Q, f₀ decreases, reducing the overall scattering.
        """
        
        if element not in self._loaded_elements:
            raise ValueError(f"No data loaded for element '{element}'")
        
        # Get Q-dependent Thomson scattering factor f₀(Q)
        # This returns a 2D array the same shape as q_magnitude
        f0_q = compute_f0_thomson(element, q_magnitude)
        
        # Get energy-dependent anomalous terms (scalars, broadcast to array)
        f_prime = self.get_f_prime(element, energy)
        f_double_prime = self.get_f_double_prime(element, energy)
        
        # Combine into complex scattering factor array
        # f(Q, E) = f₀(Q) + f'(E) + i·f''(E)
        # The real part varies with Q, imaginary part is constant
        f_real = f0_q + f_prime        # 2D array
        f_imag = np.full_like(f0_q, f_double_prime)  # 2D array of constant value
        
        f = f_real + 1j * f_imag
        
        return f
    
    def get_all_scattering_factors(self, energy: float) -> Dict[str, complex]:
        """
        Get scattering factors for all loaded elements at a given energy.
        
        Parameters
        ----------
        energy : float
            X-ray energy in eV
            
        Returns
        -------
        factors : dict
            Dictionary mapping element symbols to complex scattering factors
            
        Example
        -------
        >>> sf = ScatteringFactors()
        >>> factors = sf.get_all_scattering_factors(8333)
        >>> print(f"Ni: {factors['Ni']}, Fe: {factors['Fe']}")
        """
        
        factors = {}
        for element in self._loaded_elements:
            factors[element] = self.get_scattering_factor(element, energy)
        
        return factors
    
    def print_scattering_factors(self, energy: float):
        """
        Print a nicely formatted summary of scattering factors at a given energy.
        
        Parameters
        ----------
        energy : float
            X-ray energy in eV
        """
        
        print(f"\nScattering factors at E = {energy:.1f} eV:")
        print("-" * 50)
        print(f"{'Element':<10} {'f0 (≈Z)':<10} {'f_prime':<12} {'f_dblprm':<12} {'|f|':<10}")
        print("-" * 50)
        
        for element in sorted(self._loaded_elements):
            f0 = ATOMIC_NUMBER[element]
            fp = self.get_f_prime(element, energy)
            fpp = self.get_f_double_prime(element, energy)
            f = self.get_scattering_factor(element, energy)
            
            print(f"{element:<10} {f0:<10} {fp:<12.4f} {fpp:<12.4f} {abs(f):<10.4f}")
        
        print("-" * 50)
    
    def get_energy_range(self, element: str) -> Tuple[float, float]:
        """
        Get the valid energy range for an element.
        
        Parameters
        ----------
        element : str
            Element symbol
            
        Returns
        -------
        (e_min, e_max) : tuple of float
            Minimum and maximum energies in eV
        """
        
        if element not in self._loaded_elements:
            raise ValueError(f"No data loaded for element '{element}'")
        
        return (self._energy[element].min(), self._energy[element].max())


# =============================================================================
# PARTICLE SHAPE GENERATION (POLYGONS, HEXAGONS, ETC.)
# =============================================================================
# Note: Legacy create_core_shell_particle() removed.
# Use create_particle_with_shape(shape_type='circle', ...) for circular particles.
#
# Beyond simple circles, real nanoparticles have faceted shapes determined by
# crystallography. This section provides functions to generate various 2D
# particle shapes for more realistic simulations.
#
# Shape types:
# - 'circle': Simple circle (original)
# - 'polygon_centrosymmetric': Random convex polygon with centrosymmetry
# - 'polygon': Random convex polygon (non-centrosymmetric)
# - 'hexagon': Hexagonal shape with random rotation and anisotropy
# - 'winterbottom': Half-dome (circle with bottom truncated)
#
# All shapes are returned as binary masks that can be used to define
# core and shell regions for composition-resolved particles.
# =============================================================================

def create_polygon_mask_centrosymmetric(
    grid_size: int,
    center: Tuple[float, float] = None,
    base_radius: float = 0.3,
    noise_level: float = 0.1,
    min_corners: int = 6,
    max_corners: int = 12,
    min_relative_volume: float = 0.01,
    max_relative_volume: float = 0.3,
    max_aspect_ratio: float = 1.5,
    max_attempts: int = 1000,
    seed: int = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Create a centrosymmetric random convex polygon mask.
    
    Generates roundish polygons by placing vertices in polar coordinates,
    then enforcing centrosymmetry by reflecting points. The convex hull
    is computed to ensure a valid convex shape.
    
    Parameters
    ----------
    grid_size : int
        Size of the output mask (grid_size × grid_size)
        
    center : tuple, optional
        Center of the polygon in normalized coordinates (0 to 1).
        Default is (0.5, 0.5) = center of grid.
        
    base_radius : float
        Base radius of the polygon in normalized units (0 to 1).
        Default 0.3 gives a polygon filling ~60% of the grid diameter.
        
    noise_level : float
        Amount of radial variation in vertices. Higher = more irregular.
        Default 0.1.
        
    min_corners, max_corners : int
        Range for number of polygon vertices.
        
    min_relative_volume, max_relative_volume : float
        Acceptable range for polygon area / grid area.
        
    max_aspect_ratio : float
        Maximum width/height ratio. Reject elongated shapes.
        
    max_attempts : int
        Maximum generation attempts before giving up.
        
    seed : int, optional
        Random seed for reproducibility
        
    verbose : bool
        Print generation info
        
    Returns
    -------
    mask : np.ndarray
        Boolean mask of shape (grid_size, grid_size), True inside polygon.
    """
    from scipy.spatial import ConvexHull
    from matplotlib.path import Path
    
    if seed is not None:
        np.random.seed(seed)
    
    if center is None:
        center = (0.5, 0.5)
    
    relative_volume = -1.0
    attempts = 0
    
    while (relative_volume < min_relative_volume or 
           relative_volume > max_relative_volume) and attempts < max_attempts:
        attempts += 1
        
        # Generate half the points (will reflect for centrosymmetry)
        n_points_half = np.random.randint(min_corners // 2, max_corners // 2 + 1)
        
        # Random angles in [0, π] for half-circle
        angles = np.sort(np.random.rand(n_points_half) * np.pi)
        
        # Radii with some noise
        radii = base_radius + noise_level * (np.random.rand(n_points_half) - 0.5)
        
        # Create points in Cartesian coordinates
        points_half = np.column_stack([
            radii * np.cos(angles),
            radii * np.sin(angles)
        ])
        
        # Reflect for centrosymmetry: point (x, y) → (-x, -y)
        points_reflected = -points_half
        
        # Combine and shift to center
        points = np.vstack([points_half, points_reflected]) + np.array([center[0], center[1]])
        
        try:
            hull = ConvexHull(points)
        except:
            continue
        
        # Check aspect ratio
        hull_points = points[hull.vertices]
        min_x, min_y = hull_points.min(axis=0)
        max_x, max_y = hull_points.max(axis=0)
        width = max_x - min_x
        height = max_y - min_y
        aspect_ratio = max(width / (height + 1e-10), height / (width + 1e-10))
        
        if aspect_ratio > max_aspect_ratio:
            continue
        
        # Create mask
        x = np.linspace(0, 1, grid_size)
        y = np.linspace(0, 1, grid_size)
        xx, yy = np.meshgrid(x, y)
        pixels = np.column_stack([xx.flatten(), yy.flatten()])
        
        path = Path(points[hull.vertices])
        mask_flat = path.contains_points(pixels)
        mask = mask_flat.reshape(grid_size, grid_size)
        
        relative_volume = mask.sum() / grid_size**2
    
    if attempts >= max_attempts:
        if verbose:
            print(f"  Warning: Max attempts ({max_attempts}) reached. Using last shape.")
    
    # Convert hull vertices to pixel coordinates (y, x) format
    hull_points = points[hull.vertices]  # In normalized (0-1) coordinates
    vertices_pixel = []
    for point in hull_points:
        # point is (x_norm, y_norm) in 0-1 range
        x_pixel = point[0] * grid_size
        y_pixel = point[1] * grid_size
        vertices_pixel.append((int(y_pixel), int(x_pixel)))  # (y, x) format
    
    if verbose:
        print(f"  Centrosymmetric polygon created:")
        print(f"    Vertices: {len(hull.vertices)}")
        print(f"    Relative volume: {relative_volume:.3f}")
        print(f"    Aspect ratio: {aspect_ratio:.2f}")
        print(f"    Attempts: {attempts}")
    
    return mask.astype(bool), vertices_pixel


def create_polygon_mask(
    grid_size: int,
    center: Tuple[float, float] = None,
    base_radius: float = 0.3,
    noise_level: float = 0.1,
    min_corners: int = 6,
    max_corners: int = 12,
    min_relative_volume: float = 0.01,
    max_relative_volume: float = 0.3,
    max_aspect_ratio: float = 1.5,
    max_attempts: int = 1000,
    seed: int = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Create a random convex polygon mask (non-centrosymmetric).
    
    Similar to centrosymmetric version but allows asymmetric shapes.
    
    Parameters
    ----------
    (Same as create_polygon_mask_centrosymmetric)
        
    Returns
    -------
    mask : np.ndarray
        Boolean mask of shape (grid_size, grid_size), True inside polygon.
    """
    from scipy.spatial import ConvexHull
    from matplotlib.path import Path
    
    if seed is not None:
        np.random.seed(seed)
    
    if center is None:
        center = (0.5, 0.5)
    
    relative_volume = -1.0
    attempts = 0
    
    while (relative_volume < min_relative_volume or 
           relative_volume > max_relative_volume) and attempts < max_attempts:
        attempts += 1
        
        # Generate points around full circle
        n_points = np.random.randint(min_corners, max_corners + 1)
        
        # Random angles in [0, 2π]
        angles = np.sort(np.random.rand(n_points) * 2 * np.pi)
        
        # Radii with some noise
        radii = base_radius + noise_level * (np.random.rand(n_points) - 0.5)
        
        # Create points in Cartesian coordinates, shifted to center
        points = np.column_stack([
            radii * np.cos(angles) + center[0],
            radii * np.sin(angles) + center[1]
        ])
        
        try:
            hull = ConvexHull(points)
        except:
            continue
        
        # Check aspect ratio
        hull_points = points[hull.vertices]
        min_x, min_y = hull_points.min(axis=0)
        max_x, max_y = hull_points.max(axis=0)
        width = max_x - min_x
        height = max_y - min_y
        aspect_ratio = max(width / (height + 1e-10), height / (width + 1e-10))
        
        if aspect_ratio > max_aspect_ratio:
            continue
        
        # Create mask
        x = np.linspace(0, 1, grid_size)
        y = np.linspace(0, 1, grid_size)
        xx, yy = np.meshgrid(x, y)
        pixels = np.column_stack([xx.flatten(), yy.flatten()])
        
        path = Path(points[hull.vertices])
        mask_flat = path.contains_points(pixels)
        mask = mask_flat.reshape(grid_size, grid_size)
        
        relative_volume = mask.sum() / grid_size**2
    
    if attempts >= max_attempts:
        if verbose:
            print(f"  Warning: Max attempts ({max_attempts}) reached. Using last shape.")
    
    # Convert hull vertices to pixel coordinates (y, x) format
    hull_points = points[hull.vertices]  # In normalized (0-1) coordinates
    vertices_pixel = []
    for point in hull_points:
        # point is (x_norm, y_norm) in 0-1 range
        x_pixel = point[0] * grid_size
        y_pixel = point[1] * grid_size
        vertices_pixel.append((int(y_pixel), int(x_pixel)))  # (y, x) format
    
    if verbose:
        print(f"  Non-centrosymmetric polygon created:")
        print(f"    Vertices: {len(hull.vertices)}")
        print(f"    Relative volume: {relative_volume:.3f}")
        print(f"    Aspect ratio: {aspect_ratio:.2f}")
        print(f"    Attempts: {attempts}")
    
    return mask.astype(bool), vertices_pixel


def create_hexagon_mask(
    grid_size: int,
    center: Tuple[float, float] = None,
    base_radius: float = None,
    rotation_angle: float = None,
    anisotropy: float = None,
    seed: int = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Create a hexagonal mask with random rotation and anisotropy.
    
    Hexagons are common shapes for FCC metal nanoparticles viewed along
    certain crystallographic directions (e.g., [111]).
    
    Parameters
    ----------
    grid_size : int
        Size of the output mask (grid_size × grid_size)
        
    center : tuple, optional
        Center in normalized coordinates. Default (0.5, 0.5).
        
    base_radius : float, optional
        Base radius in normalized units. If None, random 0.25-0.4.
        
    rotation_angle : float, optional
        Rotation angle in radians. If None, random 0-π.
        
    anisotropy : float, optional
        Stretch factor (1.0 = regular hexagon). If None, random 0.8-1.2.
        
    seed : int, optional
        Random seed for reproducibility
        
    verbose : bool
        Print generation info
        
    Returns
    -------
    mask : np.ndarray
        Boolean mask of shape (grid_size, grid_size), True inside hexagon.
    """
    if seed is not None:
        np.random.seed(seed)
    
    if center is None:
        center = (0.5, 0.5)
    
    if base_radius is None:
        base_radius = np.random.uniform(0.25, 0.4)
    
    if rotation_angle is None:
        rotation_angle = np.random.rand() * np.pi
    
    if anisotropy is None:
        anisotropy = 1.0 + 0.4 * (2.0 * np.random.rand() - 1)  # 0.8 to 1.2
    
    # Create coordinate grid centered at (0, 0)
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    xx, yy = np.meshgrid(x, y)
    
    # Shift to center
    xx_centered = xx - (2 * center[0] - 1)
    yy_centered = yy - (2 * center[1] - 1)
    
    # Apply rotation
    cos_a = np.cos(rotation_angle)
    sin_a = np.sin(rotation_angle)
    x_rot = xx_centered * cos_a - yy_centered * sin_a
    y_rot = xx_centered * sin_a + yy_centered * cos_a
    
    # Hexagon is intersection of 3 pairs of half-planes
    # Each pair is perpendicular to one of the 3 axes at 60° intervals
    beta = 60 * np.pi / 180  # 60 degrees
    
    # The three "x-coordinates" in rotated frames
    x1 = x_rot
    x2 = x_rot * np.cos(beta) - y_rot * np.sin(beta)
    x3 = x_rot * np.cos(2 * beta) - y_rot * np.sin(2 * beta)
    
    # Half-width of hexagon at each orientation
    half_width = base_radius * np.cos(beta / 2)
    
    # Create mask: inside all three constraints
    mask = np.ones((grid_size, grid_size), dtype=bool)
    mask &= np.abs(x1) <= anisotropy * half_width
    mask &= np.abs(x2) <= half_width
    mask &= np.abs(x3) <= half_width
    
    # Compute actual vertex positions in pixel coordinates
    # Hexagon vertices are at angles 0, 60, 120, 180, 240, 300 degrees
    # adjusted by rotation_angle
    vertices_pixel = []
    for i in range(6):
        angle = rotation_angle + i * np.pi / 3  # 60 degree increments
        # Vertex in normalized coordinates relative to center
        vx_norm = base_radius * np.cos(angle)
        vy_norm = base_radius * np.sin(angle)
        # Apply anisotropy (stretch along x after rotation)
        # For simplicity, apply anisotropy to the first axis
        vx_rot = vx_norm * anisotropy
        vy_rot = vy_norm
        # Convert to pixel coordinates
        # center is in normalized coords (0-1), need to convert to pixels
        vx_pixel = (center[0] + vx_rot / 2) * grid_size
        vy_pixel = (center[1] + vy_rot / 2) * grid_size
        vertices_pixel.append((int(vy_pixel), int(vx_pixel)))  # (y, x) format
    
    if verbose:
        relative_volume = mask.sum() / grid_size**2
        print(f"  Hexagon created:")
        print(f"    Base radius: {base_radius:.3f}")
        print(f"    Rotation: {rotation_angle * 180 / np.pi:.1f}°")
        print(f"    Anisotropy: {anisotropy:.2f}")
        print(f"    Relative volume: {relative_volume:.3f}")
        print(f"    Vertices: {len(vertices_pixel)}")
    
    return mask, vertices_pixel


# =============================================================================
# ELLIPSE MASK
# =============================================================================

def create_ellipse_mask(
    grid_size: int,
    center: Tuple[float, float] = None,
    semi_major: float = None,
    semi_minor: float = None,
    rotation_angle: float = None,
    seed: int = None,
    verbose: bool = False
) -> Tuple[np.ndarray, List]:
    """
    Create an elliptical mask with specified semi-axes and rotation.

    Parameters
    ----------
    grid_size : int
        Size of the output mask (grid_size × grid_size)

    center : tuple, optional
        Center in normalized coordinates (0-1). Default (0.5, 0.5).

    semi_major : float, optional
        Semi-major axis in normalized units (0-1). If None, random 0.25-0.4.

    semi_minor : float, optional
        Semi-minor axis in normalized units (0-1). If None, derived from semi_major
        with random aspect ratio 1.2-2.0.

    rotation_angle : float, optional
        Rotation angle of major axis in radians. If None, random 0-π.

    seed : int, optional
        Random seed for reproducibility

    verbose : bool
        Print generation info

    Returns
    -------
    mask : np.ndarray
        Boolean mask of shape (grid_size, grid_size), True inside ellipse.

    vertices : list
        Empty list (ellipses have no discrete vertices)
    """
    if seed is not None:
        np.random.seed(seed)

    if center is None:
        center = (0.5, 0.5)

    if semi_major is None:
        semi_major = np.random.uniform(0.25, 0.4)

    if semi_minor is None:
        # Default aspect ratio between 1.2 and 2.0
        aspect_ratio = np.random.uniform(1.2, 2.0)
        semi_minor = semi_major / aspect_ratio

    if rotation_angle is None:
        rotation_angle = np.random.rand() * np.pi

    # Create coordinate grid in normalized units (-1 to 1)
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    xx, yy = np.meshgrid(x, y)

    # Shift to center (convert center from 0-1 to -1 to 1 range)
    cx = 2 * center[0] - 1
    cy = 2 * center[1] - 1
    xx_centered = xx - cx
    yy_centered = yy - cy

    # Apply inverse rotation to test points
    cos_a = np.cos(-rotation_angle)
    sin_a = np.sin(-rotation_angle)
    x_rot = xx_centered * cos_a - yy_centered * sin_a
    y_rot = xx_centered * sin_a + yy_centered * cos_a

    # Ellipse equation: (x/a)² + (y/b)² <= 1
    # In normalized coords, semi_major and semi_minor are fractions of half-grid
    ellipse_value = (x_rot / semi_major)**2 + (y_rot / semi_minor)**2
    mask = ellipse_value <= 1.0

    if verbose:
        relative_volume = mask.sum() / grid_size**2
        actual_aspect_ratio = semi_major / semi_minor
        print(f"  Ellipse created:")
        print(f"    Semi-major: {semi_major:.3f}")
        print(f"    Semi-minor: {semi_minor:.3f}")
        print(f"    Aspect ratio: {actual_aspect_ratio:.2f}")
        print(f"    Rotation: {rotation_angle * 180 / np.pi:.1f}°")
        print(f"    Relative volume: {relative_volume:.3f}")

    # Ellipses have no discrete vertices
    return mask.astype(bool), []


# =============================================================================
# PARTICLE BOUNDARY VALIDATION
# =============================================================================

def validate_particle_bounds(
    outer_radius: float,
    center: Tuple[float, float],
    grid_size: int,
    margin: int = 2
) -> bool:
    """
    Check that a particle with given radius fits entirely within the grid.

    This is critical for avoiding edge artifacts in FFT-based computations.
    Particles that extend beyond the grid boundary will cause wraparound
    artifacts in the diffraction pattern.

    Parameters
    ----------
    outer_radius : float
        Particle radius in pixels

    center : tuple
        Center position (cy, cx) in pixels

    grid_size : int
        Size of the grid (grid_size × grid_size)

    margin : int
        Minimum distance from particle edge to grid boundary in pixels.
        Default is 2 pixels to ensure clean edges.

    Returns
    -------
    valid : bool
        True if particle fits within grid with margin, False otherwise.
    """
    cy, cx = center

    # Check all four edges
    if cx - outer_radius < margin:
        return False
    if cx + outer_radius > grid_size - margin:
        return False
    if cy - outer_radius < margin:
        return False
    if cy + outer_radius > grid_size - margin:
        return False

    return True


def clamp_radius_to_grid(
    center: Tuple[float, float],
    grid_size: int,
    margin: int = 2
) -> float:
    """
    Return the maximum valid radius for a particle at the given center.

    This ensures the particle fits entirely within the grid with a safety margin.

    Parameters
    ----------
    center : tuple
        Center position (cy, cx) in pixels

    grid_size : int
        Size of the grid (grid_size × grid_size)

    margin : int
        Minimum distance from particle edge to grid boundary in pixels.

    Returns
    -------
    max_radius : float
        Maximum radius in pixels that keeps particle within bounds.
        Guaranteed to be at least 10 pixels (minimum viable particle).
    """
    cy, cx = center

    # Maximum radius is limited by closest edge
    max_radius = min(
        cx - margin,
        grid_size - margin - cx,
        cy - margin,
        grid_size - margin - cy
    )

    # Ensure minimum viable particle size
    return max(max_radius, 10.0)


# =============================================================================
# WINTERBOTTOM TRUNCATION
# =============================================================================

def apply_winterbottom_truncation(
    mask: np.ndarray,
    center: Tuple[float, float],
    truncation_fraction: float,
    truncation_angle: float = 0.0
) -> np.ndarray:
    """
    Apply Winterbottom truncation (flat cut) to any particle shape.

    This models particles supported on a substrate, which truncates the
    equilibrium crystal shape. The Winterbottom construction determines
    how much of the particle is "cut off" based on interfacial energies.

    Parameters
    ----------
    mask : np.ndarray
        Original particle mask (boolean, 2D)

    center : tuple
        Center position (cy, cx) in pixels

    truncation_fraction : float
        Fraction of particle diameter to remove (0.0 to 1.0).
        - 0.0 = no truncation (full particle)
        - 0.5 = hemisphere (half removed)
        - 0.8 = thin slice (80% removed, only 20% remains)

    truncation_angle : float
        Angle of the truncation plane in radians.
        - 0.0 = horizontal cut from bottom
        - π/2 = vertical cut from left
        - π = horizontal cut from top

    Returns
    -------
    truncated_mask : np.ndarray
        Boolean mask with truncation applied

    Notes
    -----
    The truncation is applied as a half-plane that removes all pixels
    on one side of a line passing through the particle at a distance
    determined by truncation_fraction from the center.
    """
    if truncation_fraction <= 0.0:
        return mask.copy()

    if truncation_fraction >= 1.0:
        return np.zeros_like(mask)

    grid_size = mask.shape[0]
    cy, cx = center

    # Find the particle extent to determine truncation distance
    # Use the mask to find actual particle radius
    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0:
        return mask.copy()

    # Compute distances from center to all particle pixels
    distances = np.sqrt((y_indices - cy)**2 + (x_indices - cx)**2)
    max_radius = distances.max()

    # The truncation line is at this distance from center
    # truncation_fraction=0.5 means the line passes through center
    # truncation_fraction=0.2 means line is 0.8*radius from center (removes 20%)
    truncation_distance = max_radius * (1.0 - 2.0 * truncation_fraction)

    # Create coordinate grid
    y_coords = np.arange(grid_size)
    x_coords = np.arange(grid_size)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')

    # Compute signed distance to truncation plane
    # The plane normal points in direction (sin(angle), cos(angle))
    # (angle=0 means normal points "down", cutting from bottom)
    nx = np.sin(truncation_angle)
    ny = np.cos(truncation_angle)

    # Signed distance from each point to the truncation plane
    # Points with positive distance are kept
    signed_dist = (xx - cx) * nx + (yy - cy) * ny

    # Keep points above the truncation plane
    keep_mask = signed_dist >= truncation_distance

    # Apply truncation
    truncated_mask = mask & keep_mask

    return truncated_mask


# =============================================================================
# OFF-CENTER CORE
# =============================================================================

def compute_off_center_core_mask(
    outer_mask: np.ndarray,
    center: Tuple[float, float],
    core_fraction: float,
    core_offset: Tuple[float, float],
    shape_type: str = 'circle'
) -> np.ndarray:
    """
    Compute a core mask that is offset from the particle center.

    This creates eccentric core-shell particles where the core is not
    centered within the shell.

    Parameters
    ----------
    outer_mask : np.ndarray
        Boolean mask defining the outer particle boundary

    center : tuple
        Center of outer particle (cy, cx) in pixels

    core_fraction : float
        Size of core relative to outer (0-1)

    core_offset : tuple
        Offset of core center from particle center (dy, dx) in pixels

    shape_type : str
        Shape type for the core ('circle' or 'ellipse').
        For polygon/hexagon shapes, uses circle approximation.

    Returns
    -------
    core_mask : np.ndarray
        Boolean mask for the offset core, constrained to be within outer_mask
    """
    grid_size = outer_mask.shape[0]
    cy, cx = center
    dy, dx = core_offset

    # New core center
    core_cy = cy + dy
    core_cx = cx + dx

    # Estimate outer radius from mask
    y_indices, x_indices = np.where(outer_mask)
    if len(y_indices) == 0:
        return np.zeros_like(outer_mask)

    distances = np.sqrt((y_indices - cy)**2 + (x_indices - cx)**2)
    outer_radius = distances.max()
    core_radius = outer_radius * core_fraction

    # Create core mask centered at offset position
    y_coords = np.arange(grid_size)
    x_coords = np.arange(grid_size)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')

    distance_from_core_center = np.sqrt((yy - core_cy)**2 + (xx - core_cx)**2)
    core_mask = distance_from_core_center <= core_radius

    # Constrain core to be within outer boundary
    core_mask = core_mask & outer_mask

    return core_mask


# =============================================================================
# COMPOSITION MODE FUNCTIONS
# =============================================================================

def apply_sharp_composition(
    particle: np.ndarray,
    core_mask: np.ndarray,
    shell_mask: np.ndarray,
    core_composition: Dict[str, float],
    shell_composition: Dict[str, float]
) -> None:
    """
    Apply sharp (binary) core-shell composition to a particle.

    This is the traditional core-shell model with an abrupt composition
    change at the core-shell interface.

    Parameters
    ----------
    particle : np.ndarray
        Particle array of shape (2, grid_size, grid_size) to fill in-place.
        Index 0 = Ni, Index 1 = Fe.

    core_mask : np.ndarray
        Boolean mask for core region

    shell_mask : np.ndarray
        Boolean mask for shell region

    core_composition : dict
        {'Ni': fraction, 'Fe': fraction} for core

    shell_composition : dict
        {'Ni': fraction, 'Fe': fraction} for shell
    """
    # Fill core
    particle[SPECIES_NI, core_mask] = core_composition['Ni']
    particle[SPECIES_FE, core_mask] = core_composition['Fe']

    # Fill shell
    particle[SPECIES_NI, shell_mask] = shell_composition['Ni']
    particle[SPECIES_FE, shell_mask] = shell_composition['Fe']


def apply_radial_gradient(
    particle: np.ndarray,
    outer_mask: np.ndarray,
    center: Tuple[float, float],
    inner_radius: float,
    outer_radius: float,
    inner_composition: Dict[str, float],
    outer_composition: Dict[str, float],
    transition_width: float = 0.2
) -> None:
    """
    Apply radial gradient composition from center to edge.

    Uses an error function (erf) to create a smooth transition between
    inner and outer compositions based on radial distance.

    Parameters
    ----------
    particle : np.ndarray
        Particle array of shape (2, grid_size, grid_size) to fill in-place.

    outer_mask : np.ndarray
        Boolean mask defining particle boundary

    center : tuple
        Center position (cy, cx) in pixels

    inner_radius : float
        Radius at which composition is purely inner_composition (pixels)

    outer_radius : float
        Radius at which composition is purely outer_composition (pixels)

    inner_composition : dict
        {'Ni': fraction, 'Fe': fraction} at center

    outer_composition : dict
        {'Ni': fraction, 'Fe': fraction} at edge

    transition_width : float
        Width of transition zone as fraction of (outer_radius - inner_radius).
        Smaller values = sharper transition.
    """
    from scipy.special import erf

    grid_size = particle.shape[1]
    cy, cx = center

    # Create distance map from center
    y_coords = np.arange(grid_size)
    x_coords = np.arange(grid_size)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
    distance = np.sqrt((yy - cy)**2 + (xx - cx)**2)

    # Compute transition midpoint and width
    r_mid = (inner_radius + outer_radius) / 2
    width = (outer_radius - inner_radius) * transition_width
    width = max(width, 1.0)  # Prevent division by zero

    # Smooth transition using error function
    # f = 0 at inner_radius, f = 1 at outer_radius
    outer_fraction = 0.5 * (1.0 + erf((distance - r_mid) / width))
    inner_fraction = 1.0 - outer_fraction

    # Apply gradient within mask
    particle[SPECIES_NI, outer_mask] = (
        inner_fraction[outer_mask] * inner_composition['Ni'] +
        outer_fraction[outer_mask] * outer_composition['Ni']
    )
    particle[SPECIES_FE, outer_mask] = (
        inner_fraction[outer_mask] * inner_composition['Fe'] +
        outer_fraction[outer_mask] * outer_composition['Fe']
    )


def apply_linear_gradient(
    particle: np.ndarray,
    outer_mask: np.ndarray,
    center: Tuple[float, float],
    gradient_direction: float,
    composition_start: Dict[str, float],
    composition_end: Dict[str, float],
    gradient_width: float = None
) -> None:
    """
    Apply linear gradient along one axis.

    Creates a smooth composition transition along a specified direction,
    similar to a Janus particle but with a gradual change.

    Parameters
    ----------
    particle : np.ndarray
        Particle array of shape (2, grid_size, grid_size) to fill in-place.

    outer_mask : np.ndarray
        Boolean mask defining particle boundary

    center : tuple
        Center position (cy, cx) in pixels

    gradient_direction : float
        Direction of gradient in radians.
        - 0 = gradient from left to right
        - π/2 = gradient from bottom to top

    composition_start : dict
        Composition at negative side of gradient axis

    composition_end : dict
        Composition at positive side of gradient axis

    gradient_width : float, optional
        Width over which transition occurs (pixels).
        If None, spans the full particle extent.
    """
    from scipy.special import erf

    grid_size = particle.shape[1]
    cy, cx = center

    # Create coordinate grid
    y_coords = np.arange(grid_size)
    x_coords = np.arange(grid_size)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')

    # Project coordinates onto gradient direction
    # direction=0 means gradient along x (left to right)
    dx = np.cos(gradient_direction)
    dy = np.sin(gradient_direction)
    projected = (xx - cx) * dx + (yy - cy) * dy

    # Find extent of particle along gradient direction
    if gradient_width is None:
        particle_projected = projected[outer_mask]
        if len(particle_projected) > 0:
            extent = particle_projected.max() - particle_projected.min()
            gradient_width = extent * 0.3  # Default: transition over 30% of extent
        else:
            gradient_width = 10.0

    # Smooth transition
    # Normalize to [-1, 1] range roughly
    if gradient_width > 0:
        normalized = projected / gradient_width
        end_fraction = 0.5 * (1.0 + erf(normalized))
    else:
        end_fraction = (projected > 0).astype(float)

    start_fraction = 1.0 - end_fraction

    # Apply gradient within mask
    particle[SPECIES_NI, outer_mask] = (
        start_fraction[outer_mask] * composition_start['Ni'] +
        end_fraction[outer_mask] * composition_end['Ni']
    )
    particle[SPECIES_FE, outer_mask] = (
        start_fraction[outer_mask] * composition_start['Fe'] +
        end_fraction[outer_mask] * composition_end['Fe']
    )


def apply_janus_composition(
    particle: np.ndarray,
    outer_mask: np.ndarray,
    center: Tuple[float, float],
    split_angle: float,
    composition_a: Dict[str, float],
    composition_b: Dict[str, float],
    interface_width: float = 0.0
) -> None:
    """
    Apply Janus (two-faced) composition with sharp or smooth split.

    Divides the particle into two halves along a line through the center.

    Parameters
    ----------
    particle : np.ndarray
        Particle array of shape (2, grid_size, grid_size) to fill in-place.

    outer_mask : np.ndarray
        Boolean mask defining particle boundary

    center : tuple
        Center position (cy, cx) in pixels

    split_angle : float
        Angle of the dividing line (radians).
        - 0 = vertical split (left/right)
        - π/2 = horizontal split (bottom/top)

    composition_a : dict
        Composition on negative side of split

    composition_b : dict
        Composition on positive side of split

    interface_width : float
        Width of interface in pixels. 0 = perfectly sharp.
    """
    grid_size = particle.shape[1]
    cy, cx = center

    # Create coordinate grid
    y_coords = np.arange(grid_size)
    x_coords = np.arange(grid_size)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')

    # Compute signed distance to split line
    # Normal direction: (cos(angle), sin(angle))
    nx = np.cos(split_angle)
    ny = np.sin(split_angle)
    signed_dist = (xx - cx) * nx + (yy - cy) * ny

    if interface_width > 0:
        # Smooth interface
        from scipy.special import erf
        b_fraction = 0.5 * (1.0 + erf(signed_dist / interface_width))
    else:
        # Sharp interface
        b_fraction = (signed_dist >= 0).astype(float)

    a_fraction = 1.0 - b_fraction

    # Apply compositions
    particle[SPECIES_NI, outer_mask] = (
        a_fraction[outer_mask] * composition_a['Ni'] +
        b_fraction[outer_mask] * composition_b['Ni']
    )
    particle[SPECIES_FE, outer_mask] = (
        a_fraction[outer_mask] * composition_a['Fe'] +
        b_fraction[outer_mask] * composition_b['Fe']
    )


def apply_multi_shell(
    particle: np.ndarray,
    outer_mask: np.ndarray,
    center: Tuple[float, float],
    shell_radii: List[float],
    shell_compositions: List[Dict[str, float]],
    transition_width: float = 0.0
) -> None:
    """
    Apply multi-shell (onion-like) composition with 3+ regions.

    Parameters
    ----------
    particle : np.ndarray
        Particle array of shape (2, grid_size, grid_size) to fill in-place.

    outer_mask : np.ndarray
        Boolean mask defining particle boundary

    center : tuple
        Center position (cy, cx) in pixels

    shell_radii : list of float
        Radii of shell boundaries (pixels), in increasing order.
        N radii define N+1 regions.

    shell_compositions : list of dict
        Compositions for each region, from innermost to outermost.
        Must have len(shell_radii) + 1 elements.

    transition_width : float
        Width of smooth transitions between shells (pixels).
        0 = sharp boundaries.

    Example
    -------
    # Three-region particle: Ni3Fe core, NiFe middle, Ni outer
    shell_radii = [20, 35]
    shell_compositions = [
        {'Ni': 0.75, 'Fe': 0.25},  # Innermost (r < 20)
        {'Ni': 0.50, 'Fe': 0.50},  # Middle (20 < r < 35)
        {'Ni': 1.00, 'Fe': 0.00},  # Outermost (r > 35)
    ]
    """
    from scipy.special import erf

    grid_size = particle.shape[1]
    cy, cx = center
    n_shells = len(shell_compositions)

    assert len(shell_radii) == n_shells - 1, \
        f"Need {n_shells - 1} radii for {n_shells} shells"

    # Create distance map from center
    y_coords = np.arange(grid_size)
    x_coords = np.arange(grid_size)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
    distance = np.sqrt((yy - cy)**2 + (xx - cx)**2)

    # Compute weight for each shell
    # Shell i has weight 1 where r is in its range, 0 elsewhere
    shell_weights = []

    for i in range(n_shells):
        if i == 0:
            # Innermost shell: r < shell_radii[0]
            r_boundary = shell_radii[0]
            if transition_width > 0:
                weight = 0.5 * (1.0 - erf((distance - r_boundary) / transition_width))
            else:
                weight = (distance < r_boundary).astype(float)
        elif i == n_shells - 1:
            # Outermost shell: r > shell_radii[-1]
            r_boundary = shell_radii[-1]
            if transition_width > 0:
                weight = 0.5 * (1.0 + erf((distance - r_boundary) / transition_width))
            else:
                weight = (distance >= r_boundary).astype(float)
        else:
            # Middle shells: shell_radii[i-1] < r < shell_radii[i]
            r_inner = shell_radii[i - 1]
            r_outer = shell_radii[i]
            if transition_width > 0:
                inner_weight = 0.5 * (1.0 + erf((distance - r_inner) / transition_width))
                outer_weight = 0.5 * (1.0 - erf((distance - r_outer) / transition_width))
                weight = inner_weight * outer_weight
            else:
                weight = ((distance >= r_inner) & (distance < r_outer)).astype(float)

        shell_weights.append(weight)

    # Normalize weights (they should sum to ~1 but may not exactly due to transitions)
    total_weight = sum(shell_weights)
    total_weight = np.maximum(total_weight, 1e-10)  # Avoid division by zero

    # Apply compositions
    ni_map = np.zeros((grid_size, grid_size))
    fe_map = np.zeros((grid_size, grid_size))

    for weight, comp in zip(shell_weights, shell_compositions):
        normalized_weight = weight / total_weight
        ni_map += normalized_weight * comp['Ni']
        fe_map += normalized_weight * comp['Fe']

    particle[SPECIES_NI, outer_mask] = ni_map[outer_mask]
    particle[SPECIES_FE, outer_mask] = fe_map[outer_mask]


def apply_uniform_composition(
    particle: np.ndarray,
    outer_mask: np.ndarray,
    composition: Dict[str, float]
) -> None:
    """
    Apply uniform (single) composition throughout the particle.

    This creates pure-phase particles with no core-shell structure:
    - Pure Ni (composition = {'Ni': 1.0, 'Fe': 0.0})
    - Pure Fe (composition = {'Ni': 0.0, 'Fe': 1.0})
    - Pure Ni3Fe (composition = {'Ni': 0.75, 'Fe': 0.25})

    Parameters
    ----------
    particle : np.ndarray
        Particle array of shape (2, grid_size, grid_size) to fill in-place.

    outer_mask : np.ndarray
        Boolean mask defining particle boundary

    composition : dict
        {'Ni': fraction, 'Fe': fraction} for entire particle
    """
    particle[SPECIES_NI, outer_mask] = composition['Ni']
    particle[SPECIES_FE, outer_mask] = composition['Fe']


def create_particle_with_shape(
    grid_size: int = 128,
    shape_type: str = 'circle',
    outer_radius: float = 50.0,
    core_fraction: float = 0.6,
    pixel_size: float = DEFAULT_PIXEL_SIZE,
    core_composition: Dict[str, float] = None,
    shell_composition: Dict[str, float] = None,
    shell_fe_contamination: float = None,
    center: Tuple[float, float] = None,
    shape_params: Dict[str, Any] = None,
    seed: int = None,
    verbose: bool = True,
    # NEW PARAMETERS for expanded variety
    composition_mode: str = 'sharp',
    composition_params: Dict[str, Any] = None,
    truncation_fraction: float = 0.0,
    truncation_angle: float = 0.0,
    core_offset: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Create a 2D particle with a specified shape type.
    
    This is a unified function that supports multiple particle shapes beyond
    simple circles, including faceted polygons and hexagons that are more
    representative of real nanocrystals.
    
    Parameters
    ----------
    grid_size : int
        Size of the 2D grid (grid_size × grid_size pixels).
        
    shape_type : str
        Type of particle shape:
        - 'circle': Simple circular particle
        - 'polygon': Random convex polygon (non-centrosymmetric)
        - 'polygon_centrosymmetric': Random convex polygon with centrosymmetry
        - 'hexagon': Hexagonal shape with random rotation/anisotropy
        
    outer_radius : float
        For 'circle': radius in pixels.
        For other shapes: determines base_radius in normalized units as
        outer_radius / (grid_size / 2).
        
    core_fraction : float
        Fraction of particle radius that is core (0 to 1).
        The core has the same shape as the outer boundary, just scaled down.
        Default 0.6 means core radius = 0.6 × outer radius.
        
    pixel_size : float
        Size of each pixel in Ångströms.
        
    core_composition : dict, optional
        Composition of the core region as {'Ni': fraction, 'Fe': fraction}.
        Default: {'Ni': 0.75, 'Fe': 0.25} representing Ni₃Fe.
        
    shell_composition : dict, optional
        Composition of the shell region.
        Default: {'Ni': 1.0, 'Fe': 0.0} representing pure Ni.
        
    shell_fe_contamination : float, optional
        If specified, adds Fe contamination to shell.
        
    center : tuple, optional
        Center position. If None, uses grid center.
        
    shape_params : dict, optional
        Additional parameters for specific shape types:
        - For 'polygon'/'polygon_centrosymmetric':
            - 'min_corners', 'max_corners': int
            - 'noise_level': float
        - For 'hexagon':
            - 'rotation_angle': float (radians)
            - 'anisotropy': float
            
    seed : int, optional
        Random seed for reproducibility
        
    verbose : bool
        Print creation info
        
    Returns
    -------
    particle : np.ndarray
        3D array of shape (2, grid_size, grid_size) containing species densities:
        - particle[0, :, :] = Ni density map
        - particle[1, :, :] = Fe density map
        
    info : dict
        Dictionary containing particle parameters and derived quantities.
        
    Example
    -------
    >>> # Create a hexagonal particle
    >>> particle, info = create_particle_with_shape(
    ...     grid_size=256,
    ...     shape_type='hexagon',
    ...     outer_radius=50,
    ...     core_fraction=0.6
    ... )
    
    >>> # Create a random polygon
    >>> particle, info = create_particle_with_shape(
    ...     grid_size=256,
    ...     shape_type='polygon',
    ...     outer_radius=50,
    ...     shape_params={'min_corners': 8, 'max_corners': 12}
    ... )
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    if shape_params is None:
        shape_params = {}
    
    # Set default compositions
    if core_composition is None:
        core_composition = {'Ni': 0.75, 'Fe': 0.25}
    
    if shell_fe_contamination is not None:
        shell_composition = {'Ni': 1.0 - shell_fe_contamination, 'Fe': shell_fe_contamination}
    elif shell_composition is None:
        shell_composition = {'Ni': 1.0, 'Fe': 0.0}
    
    if center is None:
        center = (grid_size / 2, grid_size / 2)
    
    # Normalized center for polygon/hexagon functions (0 to 1)
    center_normalized = (center[0] / grid_size, center[1] / grid_size)
    
    # Base radius in normalized units
    base_radius_normalized = outer_radius / (grid_size / 2) * 0.5
    
    if verbose:
        print(f"\n  Creating {shape_type} particle...")
        print(f"    Grid size: {grid_size} × {grid_size}")
        print(f"    Outer radius: {outer_radius} pixels")
        print(f"    Core fraction: {core_fraction:.1%}")
    
    # Initialize vertices list (will be populated for non-circle shapes)
    outer_vertices = []
    core_vertices = []
    
    # Create outer boundary mask based on shape type
    if shape_type == 'circle':
        # Simple circle - no vertices (or infinite vertices)
        y_coords = np.arange(grid_size)
        x_coords = np.arange(grid_size)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
        distance = np.sqrt((xx - center[0])**2 + (yy - center[1])**2)
        outer_mask = distance <= outer_radius
        core_mask = distance <= (outer_radius * core_fraction)
        outer_vertices = []  # Circles don't have discrete vertices
        core_vertices = []
        
    elif shape_type == 'polygon_centrosymmetric':
        outer_mask, outer_vertices = create_polygon_mask_centrosymmetric(
            grid_size=grid_size,
            center=center_normalized,
            base_radius=base_radius_normalized,
            noise_level=shape_params.get('noise_level', 0.1),
            min_corners=shape_params.get('min_corners', 6),
            max_corners=shape_params.get('max_corners', 12),
            verbose=verbose
        )
        # Core is same shape, scaled down
        core_mask, core_vertices = create_polygon_mask_centrosymmetric(
            grid_size=grid_size,
            center=center_normalized,
            base_radius=base_radius_normalized * core_fraction,
            noise_level=shape_params.get('noise_level', 0.1) * core_fraction,
            min_corners=shape_params.get('min_corners', 6),
            max_corners=shape_params.get('max_corners', 12),
            verbose=False
        )
        
    elif shape_type == 'polygon':
        outer_mask, outer_vertices = create_polygon_mask(
            grid_size=grid_size,
            center=center_normalized,
            base_radius=base_radius_normalized,
            noise_level=shape_params.get('noise_level', 0.1),
            min_corners=shape_params.get('min_corners', 6),
            max_corners=shape_params.get('max_corners', 12),
            verbose=verbose
        )
        # Core is same shape, scaled down (use same seed offset for consistency)
        core_mask, core_vertices = create_polygon_mask(
            grid_size=grid_size,
            center=center_normalized,
            base_radius=base_radius_normalized * core_fraction,
            noise_level=shape_params.get('noise_level', 0.1) * core_fraction,
            min_corners=shape_params.get('min_corners', 6),
            max_corners=shape_params.get('max_corners', 12),
            verbose=False
        )
        
    elif shape_type == 'hexagon':
        rotation = shape_params.get('rotation_angle', None)
        anisotropy = shape_params.get('anisotropy', None)

        outer_mask, outer_vertices = create_hexagon_mask(
            grid_size=grid_size,
            center=center_normalized,
            base_radius=base_radius_normalized,
            rotation_angle=rotation,
            anisotropy=anisotropy,
            verbose=verbose
        )
        # Core hexagon with same rotation/anisotropy
        # Need to store the values used for outer mask
        # For simplicity, regenerate with scaled radius
        core_mask, core_vertices = create_hexagon_mask(
            grid_size=grid_size,
            center=center_normalized,
            base_radius=base_radius_normalized * core_fraction,
            rotation_angle=rotation,
            anisotropy=anisotropy,
            verbose=False
        )

    elif shape_type == 'ellipse':
        # Elliptical particle with independent semi-axes
        aspect_ratio = shape_params.get('aspect_ratio', None)
        rotation = shape_params.get('rotation_angle', None)

        # If aspect_ratio provided, derive semi_minor from semi_major
        if aspect_ratio is not None:
            semi_major = base_radius_normalized
            semi_minor = semi_major / aspect_ratio
        else:
            semi_major = shape_params.get('semi_major', base_radius_normalized)
            semi_minor = shape_params.get('semi_minor', None)

        outer_mask, outer_vertices = create_ellipse_mask(
            grid_size=grid_size,
            center=center_normalized,
            semi_major=semi_major,
            semi_minor=semi_minor,
            rotation_angle=rotation,
            verbose=verbose
        )
        # Core ellipse with same proportions, scaled down
        core_mask, core_vertices = create_ellipse_mask(
            grid_size=grid_size,
            center=center_normalized,
            semi_major=semi_major * core_fraction if semi_major else None,
            semi_minor=semi_minor * core_fraction if semi_minor else None,
            rotation_angle=rotation,
            verbose=False
        )

    else:
        raise ValueError(
            f"Unknown shape_type '{shape_type}'. "
            f"Available: 'circle', 'polygon', 'polygon_centrosymmetric', 'hexagon', 'ellipse'"
        )
    
    # Apply Winterbottom truncation if requested
    if truncation_fraction > 0.0:
        outer_mask = apply_winterbottom_truncation(
            outer_mask, center, truncation_fraction, truncation_angle
        )
        # Also truncate core if it extends into truncated region
        core_mask = core_mask & outer_mask
        if verbose:
            print(f"    Winterbottom truncation: {truncation_fraction:.0%} removed")

    # Apply off-center core if requested
    if core_offset != (0.0, 0.0) and core_offset != (0, 0):
        core_mask = compute_off_center_core_mask(
            outer_mask, center, core_fraction, core_offset, shape_type
        )
        if verbose:
            print(f"    Off-center core: offset ({core_offset[0]:.1f}, {core_offset[1]:.1f}) pixels")

    # Shell is outer minus core
    shell_mask = outer_mask & ~core_mask

    # Create composition-resolved particle
    particle = np.zeros((N_SPECIES, grid_size, grid_size), dtype=np.float64)

    # Initialize composition_params if not provided
    if composition_params is None:
        composition_params = {}

    # Apply composition based on composition_mode
    if composition_mode == 'sharp':
        # Traditional sharp core-shell boundary
        apply_sharp_composition(
            particle, core_mask, shell_mask,
            core_composition, shell_composition
        )

    elif composition_mode == 'radial_gradient':
        # Smooth radial gradient from center to edge
        inner_comp = composition_params.get('inner_composition', core_composition)
        outer_comp = composition_params.get('outer_composition', shell_composition)
        transition_width = composition_params.get('transition_width', 0.2)
        apply_radial_gradient(
            particle, outer_mask, center,
            inner_radius=outer_radius * core_fraction,
            outer_radius=outer_radius,
            inner_composition=inner_comp,
            outer_composition=outer_comp,
            transition_width=transition_width
        )
        if verbose:
            print(f"    Composition mode: radial_gradient (transition_width={transition_width:.2f})")

    elif composition_mode == 'linear_gradient':
        # Linear gradient along one axis
        comp_start = composition_params.get('composition_start', core_composition)
        comp_end = composition_params.get('composition_end', shell_composition)
        gradient_dir = composition_params.get('gradient_direction', 0.0)
        gradient_width = composition_params.get('gradient_width', None)
        apply_linear_gradient(
            particle, outer_mask, center,
            gradient_direction=gradient_dir,
            composition_start=comp_start,
            composition_end=comp_end,
            gradient_width=gradient_width
        )
        if verbose:
            print(f"    Composition mode: linear_gradient (direction={gradient_dir*180/np.pi:.0f}°)")

    elif composition_mode == 'janus':
        # Sharp or smooth two-faced split
        comp_a = composition_params.get('composition_a', core_composition)
        comp_b = composition_params.get('composition_b', shell_composition)
        split_angle = composition_params.get('split_angle', 0.0)
        interface_width = composition_params.get('interface_width', 0.0)
        apply_janus_composition(
            particle, outer_mask, center,
            split_angle=split_angle,
            composition_a=comp_a,
            composition_b=comp_b,
            interface_width=interface_width
        )
        if verbose:
            print(f"    Composition mode: janus (split_angle={split_angle*180/np.pi:.0f}°)")

    elif composition_mode == 'multi_shell':
        # Onion-like multi-shell structure
        shell_radii = composition_params.get('shell_radii', [outer_radius * 0.4, outer_radius * 0.7])
        shell_compositions = composition_params.get('shell_compositions', [
            core_composition,
            {'Ni': 0.5, 'Fe': 0.5},
            shell_composition
        ])
        transition_width = composition_params.get('transition_width', 0.0)
        apply_multi_shell(
            particle, outer_mask, center,
            shell_radii=shell_radii,
            shell_compositions=shell_compositions,
            transition_width=transition_width
        )
        if verbose:
            print(f"    Composition mode: multi_shell ({len(shell_compositions)} shells)")

    elif composition_mode == 'uniform':
        # Single uniform composition throughout
        composition = composition_params.get('composition', core_composition)
        apply_uniform_composition(particle, outer_mask, composition)
        if verbose:
            print(f"    Composition mode: uniform (Ni={composition['Ni']:.0%}, Fe={composition['Fe']:.0%})")

    else:
        raise ValueError(
            f"Unknown composition_mode '{composition_mode}'. "
            f"Available: 'sharp', 'radial_gradient', 'linear_gradient', 'janus', 'multi_shell', 'uniform'"
        )
    
    # Calculate statistics
    core_area = core_mask.sum()
    shell_area = shell_mask.sum()
    total_area = outer_mask.sum()
    
    # Physical dimensions
    physical_extent = grid_size * pixel_size
    physical_diameter_approx = 2 * outer_radius * pixel_size
    
    info = {
        # Grid parameters
        'grid_size': grid_size,
        'center': center,
        'pixel_size': pixel_size,

        # Shape info
        'shape_type': shape_type,
        'outer_radius': outer_radius,
        'core_fraction': core_fraction,
        'shape_params': shape_params,

        # Vertex positions (for corner enhancement in displacement)
        # Format: list of (y, x) tuples in pixel coordinates
        'outer_vertices': outer_vertices,
        'core_vertices': core_vertices,

        # Physical dimensions
        'physical_extent_angstrom': physical_extent,
        'physical_extent_nm': physical_extent / 10.0,
        'physical_diameter_approx_nm': physical_diameter_approx / 10.0,

        # Q-space info
        'q_max': np.pi / pixel_size,

        # Composition
        'core_composition': core_composition.copy(),
        'shell_composition': shell_composition.copy(),
        'shell_fe_contamination': shell_fe_contamination,

        # NEW: Composition mode info
        'composition_mode': composition_mode,
        'composition_params': composition_params.copy() if composition_params else {},

        # NEW: Geometry modifiers
        'truncation_fraction': truncation_fraction,
        'truncation_angle': truncation_angle,
        'core_offset': core_offset,

        # Areas
        'core_area': int(core_area),
        'shell_area': int(shell_area),
        'total_area': int(total_area),
        'core_area_fraction': core_area / total_area if total_area > 0 else 0,
        'shell_area_fraction': shell_area / total_area if total_area > 0 else 0,

        # Masks
        'outer_mask': outer_mask,
        'core_mask': core_mask,
        'shell_mask': shell_mask,
    }
    
    if verbose:
        print(f"    Total area: {total_area} pixels")
        print(f"    Core area: {core_area} pixels ({100*core_area/total_area:.1f}%)")
        print(f"    Shell area: {shell_area} pixels ({100*shell_area/total_area:.1f}%)")
        print(f"    Core: Ni={core_composition['Ni']:.0%}, Fe={core_composition['Fe']:.0%}")
        print(f"    Shell: Ni={shell_composition['Ni']:.1%}, Fe={shell_composition['Fe']:.1%}")
    
    return particle, info


# =============================================================================
# DIFFRACTION (FFT) - BASIC (NO ENERGY DEPENDENCE)
# =============================================================================

def compute_diffraction(particle: np.ndarray) -> np.ndarray:
    """
    Compute the diffraction pattern of a 2D particle using FFT.
    
    This is the BASIC version that works with either:
    1. A single 2D density array (old interface)
    2. A composition-resolved (2, Ny, Nx) array, summed to effective density
    
    For energy-dependent calculations, use compute_diffraction_at_energy() instead.
    
    Parameters
    ----------
    particle : np.ndarray
        Either 2D array (Ny, Nx) or 3D array (2, Ny, Nx)
        
    Returns
    -------
    diffraction : np.ndarray
        2D complex array of diffraction amplitude
    """
    
    if particle.ndim == 3:
        effective_density = np.sum(particle, axis=0)
    elif particle.ndim == 2:
        effective_density = particle
    else:
        raise ValueError(f"particle must be 2D or 3D, got shape {particle.shape}")
    
    fft_result = np.fft.fft2(effective_density)
    diffraction = np.fft.fftshift(fft_result)
    
    return diffraction


# =============================================================================
# DIFFRACTION (FFT) - FULL Q AND ENERGY DEPENDENCE
# =============================================================================
# Note: Removed legacy functions (compute_diffraction_at_energy,
# compute_diffraction_multi_energy, compute_diffraction_q_dependent).
# Use compute_diffraction_multi_energy_q_dependent for full physics.

def compute_diffraction_multi_energy_q_dependent(
    particle: np.ndarray,
    energies: List[float],
    pixel_size: float,
    scattering_factors: ScatteringFactors
) -> Dict[float, np.ndarray]:
    """
    Compute Q-dependent diffraction patterns at multiple energies efficiently.
    
    This combines the multi-energy efficiency optimization with full Q-dependence:
    
    1. Compute FFT(ρ_Ni) and FFT(ρ_Fe) once (2 FFTs total)
    2. Compute Q-grid once
    3. For each energy: compute f(Q, E) arrays and combine with precomputed FFTs
    
    The Q-dependent scattering factors f₀(Q) are computed from IT92 coefficients,
    while f'(E) and f''(E) come from the tabulated data files.
    
    Parameters
    ----------
    particle : np.ndarray
        Composition-resolved particle array of shape (2, Ny, Nx)
        
    energies : list of float
        List of X-ray energies in eV
        
    pixel_size : float
        Real-space pixel size in Ångströms
        
    scattering_factors : ScatteringFactors
        ScatteringFactors object with loaded f' and f'' data
        
    Returns
    -------
    diffractions : dict
        Dictionary mapping energy (float) to diffraction pattern (2D complex array)
        
    Example
    -------
    >>> particle, info = create_core_shell_particle(pixel_size=10.0)
    >>> sf = ScatteringFactors(data_dir='.')
    >>> energies = [8283, 8333, 8383]
    >>> diffs = compute_diffraction_multi_energy_q_dependent(
    ...     particle, energies, 10.0, sf
    ... )
    >>> for E, diff in diffs.items():
    ...     print(f"E={E} eV: max intensity = {np.abs(diff).max()**2:.2e}")
    """
    
    if particle.ndim != 3 or particle.shape[0] != N_SPECIES:
        raise ValueError(
            f"particle must have shape (2, Ny, Nx), got {particle.shape}"
        )
    
    grid_size = particle.shape[1]
    
    # -------------------------------------------------------------------------
    # Precompute things that don't depend on energy
    # -------------------------------------------------------------------------
    
    # Q-grid (same for all energies)
    _, _, q_magnitude = compute_q_grid(grid_size, pixel_size)
    
    # Precompute f₀(Q) arrays (Q-dependent part is energy-independent!)
    # This is a key optimization: f₀(Q) only needs to be computed once
    f0_Ni_q = compute_f0_thomson('Ni', q_magnitude)
    f0_Fe_q = compute_f0_thomson('Fe', q_magnitude)
    
    # Precompute FFTs of species maps
    F_Ni = np.fft.fftshift(np.fft.fft2(particle[SPECIES_NI]))
    F_Fe = np.fft.fftshift(np.fft.fft2(particle[SPECIES_FE]))
    
    # -------------------------------------------------------------------------
    # Loop over energies
    # -------------------------------------------------------------------------
    
    diffractions = {}
    
    for energy in energies:
        # Get energy-dependent terms (scalars)
        fp_Ni = scattering_factors.get_f_prime('Ni', energy)
        fpp_Ni = scattering_factors.get_f_double_prime('Ni', energy)
        fp_Fe = scattering_factors.get_f_prime('Fe', energy)
        fpp_Fe = scattering_factors.get_f_double_prime('Fe', energy)
        
        # Build full Q-dependent scattering factor arrays
        # f(Q, E) = f₀(Q) + f'(E) + i·f''(E)
        f_Ni_q = (f0_Ni_q + fp_Ni) + 1j * fpp_Ni
        f_Fe_q = (f0_Fe_q + fp_Fe) + 1j * fpp_Fe
        
        # Combine with FFTs
        diffraction = f_Ni_q * F_Ni + f_Fe_q * F_Fe
        
        diffractions[energy] = diffraction
    
    return diffractions


def get_diffraction_intensity(diffraction: np.ndarray, log_scale: bool = True) -> np.ndarray:
    """
    Get the intensity of a diffraction pattern for visualization.
    
    Parameters
    ----------
    diffraction : np.ndarray
        Complex diffraction amplitude from compute_diffraction()
        
    log_scale : bool
        If True, return log10(intensity + epsilon) for better visualization.
        
    Returns
    -------
    intensity : np.ndarray
        Real-valued intensity array suitable for plotting.
    """
    
    intensity = np.abs(diffraction)**2
    
    if log_scale:
        intensity = np.log10(intensity + 1e-10)
    
    return intensity


# =============================================================================
# OVERSAMPLING AND CENTER CROP
# =============================================================================
#
# Oversampling is critical for proper BCDI simulation. The oversampling ratio σ
# is defined as:
#
#     σ = L_grid / D_object = (N_pixels × pixel_size) / particle_diameter
#
# For proper sampling of diffraction fringes (Nyquist criterion in Q-space),
# we need σ ≥ 2. Typical values are 2.5-4.
#
# Strategy:
# 1. Create particle on a LARGER grid (e.g., 256×256) for proper oversampling
# 2. Compute FFT on this larger grid
# 3. Center-crop the diffraction pattern to the desired output size (e.g., 128×128)
#
# This mirrors experimental practice where the detector captures a large Q-range
# but you crop to the region of interest around the Bragg peak.
# =============================================================================

def compute_oversampling_ratio(
    grid_size: int,
    particle_diameter_pixels: float,
) -> float:
    """
    Compute the oversampling ratio for a given grid and particle size.
    
    Parameters
    ----------
    grid_size : int
        Size of the FFT grid in pixels
        
    particle_diameter_pixels : float
        Diameter of the particle in pixels
        
    Returns
    -------
    sigma : float
        Oversampling ratio (should be ≥ 2 for proper sampling)
    """
    return grid_size / particle_diameter_pixels


def center_crop_2d(array: np.ndarray, output_size: int) -> np.ndarray:
    """
    Extract the central region of a 2D array.
    
    Parameters
    ----------
    array : np.ndarray
        Input 2D array
        
    output_size : int
        Size of the output (output_size × output_size)
        
    Returns
    -------
    cropped : np.ndarray
        Center-cropped array of shape (output_size, output_size)
    """
    if array.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {array.shape}")
    
    ny, nx = array.shape
    
    if output_size > min(ny, nx):
        raise ValueError(
            f"output_size ({output_size}) cannot be larger than array dimensions ({ny}, {nx})"
        )
    
    start_y = (ny - output_size) // 2
    start_x = (nx - output_size) // 2
    
    return array[start_y:start_y + output_size, start_x:start_x + output_size]


def compute_diffraction_oversampled_cropped(
    particle: np.ndarray,
    energies: List[float],
    pixel_size: float,
    scattering_factors: 'ScatteringFactors',
    output_size: int = 128,
    verbose: bool = True
) -> Dict[float, np.ndarray]:
    """
    Compute multi-energy diffraction with proper oversampling, then center-crop.
    
    This function:
    1. Computes FFT on the full (oversampled) particle grid
    2. Center-crops the result to output_size × output_size
    
    This ensures the diffraction fringes are properly resolved before cropping
    to the region of interest around the Bragg peak.
    
    Parameters
    ----------
    particle : np.ndarray
        Composition-resolved particle array of shape (2, N_fft, N_fft)
        where N_fft is the larger FFT grid size
        
    energies : list of float
        List of X-ray energies in eV
        
    pixel_size : float
        Real-space pixel size in Ångströms
        
    scattering_factors : ScatteringFactors
        ScatteringFactors object with loaded f' and f'' data
        
    output_size : int
        Size of the output diffraction patterns after center cropping.
        Default is 128.
        
    verbose : bool
        If True, print information about the computation
        
    Returns
    -------
    diffractions_cropped : dict
        Dictionary mapping energy (float) to cropped diffraction pattern
        (2D complex array of shape output_size × output_size)
        
    Example
    -------
    >>> # Create particle on 256×256 grid for σ ≈ 2.5
    >>> particle, info = create_core_shell_particle(grid_size=256, outer_radius=50)
    >>> # Compute diffraction and crop to 128×128
    >>> diffs = compute_diffraction_oversampled_cropped(
    ...     particle, energies, pixel_size=10.0, scattering_factors=sf, output_size=128
    ... )
    """
    
    grid_size = particle.shape[1]
    
    if verbose:
        print(f"\n  Computing oversampled diffraction:")
        print(f"    FFT grid size: {grid_size} × {grid_size}")
        print(f"    Output size (after crop): {output_size} × {output_size}")
        print(f"    Pixel size: {pixel_size} Å")
        print(f"    Grid extent: {grid_size * pixel_size / 10:.1f} nm")
    
    # Compute full diffraction patterns on the larger grid
    diffractions_full = compute_diffraction_multi_energy_q_dependent(
        particle=particle,
        energies=energies,
        pixel_size=pixel_size,
        scattering_factors=scattering_factors
    )
    
    # Center-crop each energy slice
    diffractions_cropped = {}
    for energy, diff_full in diffractions_full.items():
        diffractions_cropped[energy] = center_crop_2d(diff_full, output_size)
    
    if verbose:
        print(f"    Computed {len(diffractions_cropped)} diffraction patterns")
        print(f"    Each pattern cropped from {grid_size}×{grid_size} to {output_size}×{output_size}")
    
    return diffractions_cropped


# =============================================================================
# NOISE MODELS
# =============================================================================
#
# Two types of noise are implemented:
#
# 1. POISSON NOISE (Counting Statistics)
#    - Fundamental noise from photon counting at the detector
#    - sqrt(N) statistics: higher intensity → higher absolute noise but lower relative noise
#    - Applied to intensities I = |A|²
#    - Scale factor determines the maximum photon count
#
# 2. CORRELATED MODULE NOISE (Real-Space Inhomogeneity)
#    - Smooth spatial variations in the real-space electron density
#    - Models inhomogeneous composition, surface roughness, defects
#    - Applied to the real-space amplitude |ρ(r)| before FFT
#    - Uses Robinson's convolution method: convolve white noise with Gaussian kernel
#
# Both can be toggled independently for debugging and understanding their effects.
# =============================================================================


# =============================================================================
# GROUND TRUTH LABEL COMPUTATION FOR ML TRAINING
# =============================================================================
#
# These functions compute the ground truth labels for training the ML model
# to replace NanoMAD's chi-squared fitting.
#
# The MAD equation that NanoMAD fits is:
#
#   I(Q, E) = |F_T|² + (f'² + f''²)|F_A/f₀|² + 2|F_T||F_A|/f₀ × [f'·cos(Δφ) + f''·sin(Δφ)]
#
# Where:
#   F_T = Total structure factor (F_Ni + F_Fe in our case)
#   F_A = Anomalous structure factor (F_Ni at Ni K-edge)
#   F_N = Non-anomalous structure factor (F_Fe at Ni K-edge), derived as F_T - F_A
#   Δφ = φ_T - φ_A = Phase difference between F_T and F_A
#
# NanoMAD fits |F_T|, |F_A|, and Δφ from multi-energy intensity data.
# Our ML model will predict these same three quantities directly.
#
# From |F_T|, |F_A|, and Δφ, we can derive F_N:
#   Re(F_N) = |F_T|·cos(Δφ) - |F_A|
#   Im(F_N) = |F_T|·sin(Δφ)
#   |F_N| = √(Re² + Im²)
#
# =============================================================================

def compute_ground_truth_labels(
    particle: np.ndarray,
    pixel_size: float = DEFAULT_PIXEL_SIZE,
    output_size: int = None,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Compute ground truth labels (|F_T|, |F_A|, Δφ) for ML training.
    
    These are the quantities that NanoMAD fits from multi-energy data.
    Our ML model will learn to predict these directly from intensity patches.
    
    IMPORTANT: This function computes structure factors weighted by the
    Thomson scattering factor f₀(Q), NOT the full energy-dependent scattering
    factor. This matches NanoMAD's convention where:
    
        F_T = f₀_Ni(Q)·FFT(ρ_Ni) + f₀_Fe(Q)·FFT(ρ_Fe)
        F_A = f₀_Ni(Q)·FFT(ρ_Ni)
        
    The energy-dependent f'(E) and f''(E) are handled separately in the
    MAD equation fitting, not baked into F_T and F_A.
    
    Parameters
    ----------
    particle : np.ndarray
        Composition-resolved particle of shape (2, Ny, Nx).
        Can be complex (if displacement/phase has been applied).
        particle[0] = Ni density (anomalous at Ni K-edge)
        particle[1] = Fe density (non-anomalous at Ni K-edge)
        
    pixel_size : float
        Pixel size in Ångströms. Needed to compute Q values for f₀(Q).
        
    output_size : int, optional
        If specified, center-crop the output to this size.
        Use this to match the cropped diffraction patterns.
        
    verbose : bool
        Print label statistics
        
    Returns
    -------
    labels : dict
        Dictionary containing:
        - 'F_T_mag': |F_T| array (Thomson-weighted)
        - 'F_A_mag': |F_A| array (Thomson-weighted)
        - 'delta_phi': Δφ array in radians, range [-π, π]
        - 'cos_delta_phi': cos(Δφ) for ML output (avoids discontinuity)
        - 'sin_delta_phi': sin(Δφ) for ML output (avoids discontinuity)
        - 'F_N_mag': |F_N| array (derived, for validation)
        - 'f0_Ni': f₀_Ni(Q) at each pixel (for optional normalization)
        - 'f0_Fe': f₀_Fe(Q) at each pixel
        
    Notes
    -----
    At the Ni K-edge:
    - F_A corresponds to Ni (the anomalous element)
    - F_N corresponds to Fe (non-anomalous at this edge)
    
    The CNN should predict 4 outputs per pixel:
    - |F_T| (use softplus activation → positive)
    - |F_A| (use softplus activation → positive)
    - sin(Δφ) (use tanh activation → range [-1, 1])
    - cos(Δφ) (use tanh activation → range [-1, 1])
    
    Then recover Δφ = atan2(sin, cos) in post-processing.
    
    Consider adding regularization: sin²(Δφ) + cos²(Δφ) ≈ 1
    """
    
    if particle.ndim != 3 or particle.shape[0] != N_SPECIES:
        raise ValueError(f"particle must have shape (2, Ny, Nx), got {particle.shape}")
    
    grid_size = particle.shape[1]
    
    # Extract species maps
    rho_Ni = particle[SPECIES_NI]  # Anomalous at Ni K-edge
    rho_Fe = particle[SPECIES_FE]  # Non-anomalous at Ni K-edge
    
    # -------------------------------------------------------------------------
    # Compute Q-grid for Thomson factor calculation
    # -------------------------------------------------------------------------
    
    # Reciprocal space coordinates
    q_max = np.pi / pixel_size  # Maximum Q at Nyquist
    q_1d = np.fft.fftfreq(grid_size, d=pixel_size) * 2 * np.pi
    q_1d_shifted = np.fft.fftshift(q_1d)
    qy, qx = np.meshgrid(q_1d_shifted, q_1d_shifted, indexing='ij')
    q_magnitude = np.sqrt(qx**2 + qy**2)
    
    # -------------------------------------------------------------------------
    # Compute Thomson scattering factors f₀(Q) for both elements
    # -------------------------------------------------------------------------
    # 
    # CRITICAL: Use ONLY f₀(Q), NOT f₀ + f' + if''
    # The energy-dependent terms are handled separately in the MAD equation
    #
    
    f0_Ni = compute_f0_thomson('Ni', q_magnitude)
    f0_Fe = compute_f0_thomson('Fe', q_magnitude)
    
    if verbose:
        print(f"\n  Computing ground truth labels (Thomson-weighted)...")
        print(f"    f₀_Ni(Q=0) = {compute_f0_thomson('Ni', 0.0):.2f} (atomic number ~28)")
        print(f"    f₀_Fe(Q=0) = {compute_f0_thomson('Fe', 0.0):.2f} (atomic number ~26)")
        print(f"    Q_max = {q_max:.2f} Å⁻¹")
    
    # -------------------------------------------------------------------------
    # Compute structure factors: F = f₀(Q) × FFT(ρ)
    # -------------------------------------------------------------------------
    #
    # This is the key fix: we must weight by f₀(Q) to match NanoMAD convention
    #
    
    # FFT of density maps (these include phase from strain if particle is complex)
    FFT_rho_Ni = np.fft.fftshift(np.fft.fft2(rho_Ni))
    FFT_rho_Fe = np.fft.fftshift(np.fft.fft2(rho_Fe))
    
    # Weight by Thomson factor (f₀ only, NOT f' or f'')
    F_Ni = f0_Ni * FFT_rho_Ni  # Anomalous element contribution
    F_Fe = f0_Fe * FFT_rho_Fe  # Non-anomalous element contribution
    
    # Total structure factor (Thomson-weighted)
    F_T = F_Ni + F_Fe
    
    # Anomalous structure factor (just Ni at Ni K-edge)
    F_A = F_Ni
    
    # Non-anomalous structure factor (just Fe at Ni K-edge)
    F_N = F_Fe
    
    # -------------------------------------------------------------------------
    # Center crop if requested
    # -------------------------------------------------------------------------
    
    if output_size is not None:
        F_T = center_crop_2d(F_T, output_size)
        F_A = center_crop_2d(F_A, output_size)
        F_N = center_crop_2d(F_N, output_size)
        f0_Ni = center_crop_2d(f0_Ni, output_size)
        f0_Fe = center_crop_2d(f0_Fe, output_size)
    
    # -------------------------------------------------------------------------
    # Compute the quantities NanoMAD fits
    # -------------------------------------------------------------------------
    
    F_T_mag = np.abs(F_T)
    F_A_mag = np.abs(F_A)
    F_N_mag = np.abs(F_N)
    
    # Phase difference: Δφ = φ_T - φ_A
    phi_T = np.angle(F_T)
    phi_A = np.angle(F_A)
    delta_phi = phi_T - phi_A
    
    # Wrap to [-π, π]
    delta_phi = np.angle(np.exp(1j * delta_phi))
    
    # ML output representation: sin and cos (avoids discontinuity at ±π)
    cos_delta_phi = np.cos(delta_phi)
    sin_delta_phi = np.sin(delta_phi)
    
    # -------------------------------------------------------------------------
    # Derive F_N from (F_T, F_A, Δφ) for validation
    # -------------------------------------------------------------------------
    # This tests the derivation that NanoMAD uses post-fitting:
    # F_N = F_T - F_A (as complex vectors)
    # In frame where φ_A = 0:
    #   Re(F_N) = |F_T|·cos(Δφ) - |F_A|
    #   Im(F_N) = |F_T|·sin(Δφ)
    
    Re_F_N = F_T_mag * cos_delta_phi - F_A_mag
    Im_F_N = F_T_mag * sin_delta_phi
    F_N_mag_derived = np.sqrt(Re_F_N**2 + Im_F_N**2)
    
    if verbose:
        print(f"    |F_T| range: [{F_T_mag.min():.2e}, {F_T_mag.max():.2e}]")
        print(f"    |F_A| range: [{F_A_mag.min():.2e}, {F_A_mag.max():.2e}]")
        print(f"    |F_A|/|F_T| at max: {F_A_mag.max()/F_T_mag.max():.2%}")
        print(f"    Δφ range: [{delta_phi.min():.3f}, {delta_phi.max():.3f}] rad")
        print(f"    Δφ range: [{np.degrees(delta_phi.min()):.1f}°, {np.degrees(delta_phi.max()):.1f}°]")
        print(f"    |F_N| (direct) range: [{F_N_mag.min():.2e}, {F_N_mag.max():.2e}]")
        print(f"    |F_N| (derived) range: [{F_N_mag_derived.min():.2e}, {F_N_mag_derived.max():.2e}]")
        
        # Check consistency between direct and derived F_N
        F_N_diff = np.abs(F_N_mag_derived - F_N_mag)
        rel_error = F_N_diff.max() / (F_N_mag.max() + 1e-10)
        print(f"    |F_N| derivation error (max relative): {rel_error:.2e}")
    
    return {
        # Primary outputs (what CNN predicts)
        'F_T_mag': F_T_mag,
        'F_A_mag': F_A_mag,
        'delta_phi': delta_phi,
        'cos_delta_phi': cos_delta_phi,
        'sin_delta_phi': sin_delta_phi,
        
        # Derived/validation outputs
        'F_N_mag': F_N_mag,
        'F_N_mag_derived': F_N_mag_derived,
        
        # Thomson factors (for optional CNN input or normalization)
        'f0_Ni': f0_Ni,
        'f0_Fe': f0_Fe,
    }


def extract_label_patches(
    labels: Dict[str, np.ndarray],
    patch_size: int = 16,
    label_keys: List[str] = None
) -> np.ndarray:
    """
    Extract non-overlapping patches from ground truth label arrays.
    
    Parameters
    ----------
    labels : dict
        Dictionary of label arrays from compute_ground_truth_labels()
        
    patch_size : int
        Size of each patch (patch_size × patch_size)
        
    label_keys : list of str, optional
        Which labels to include in the output.
        Default: ['F_T_mag', 'F_A_mag', 'delta_phi'] (the 3 NanoMAD quantities)
        Alternative: ['F_T_mag', 'F_A_mag', 'cos_delta_phi', 'sin_delta_phi'] (4 outputs)
        
    Returns
    -------
    label_patches : np.ndarray
        Array of shape (n_patches_y, n_patches_x, patch_size, patch_size, n_labels)
        where n_labels = len(label_keys)
        
    Example
    -------
    >>> labels = compute_ground_truth_labels(particle_complex, output_size=128)
    >>> label_patches = extract_label_patches(labels, patch_size=16)
    >>> print(f"Label patches shape: {label_patches.shape}")  # (8, 8, 16, 16, 3)
    """
    
    if label_keys is None:
        label_keys = ['F_T_mag', 'F_A_mag', 'delta_phi']
    
    # Get array shape from first label
    first_key = label_keys[0]
    ny, nx = labels[first_key].shape
    
    n_patches_y = ny // patch_size
    n_patches_x = nx // patch_size
    n_labels = len(label_keys)
    
    # Initialize output array
    label_patches = np.zeros((n_patches_y, n_patches_x, patch_size, patch_size, n_labels))
    
    # Extract patches for each label
    for label_idx, key in enumerate(label_keys):
        label_array = labels[key]
        
        for i in range(n_patches_y):
            for j in range(n_patches_x):
                y_start = i * patch_size
                x_start = j * patch_size
                
                label_patches[i, j, :, :, label_idx] = label_array[
                    y_start:y_start + patch_size,
                    x_start:x_start + patch_size
                ]
    
    return label_patches


def compute_F_N_from_predictions(
    F_T_mag: np.ndarray,
    F_A_mag: np.ndarray,
    delta_phi: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Derive F_N from predicted/fitted MAD parameters.
    
    This is the same derivation NanoMAD uses after fitting.
    Use this to convert ML predictions into the non-anomalous structure factor.
    
    Parameters
    ----------
    F_T_mag : np.ndarray
        Magnitude of total structure factor |F_T|
        
    F_A_mag : np.ndarray
        Magnitude of anomalous structure factor |F_A|
        
    delta_phi : np.ndarray
        Phase difference Δφ = φ_T - φ_A in radians
        
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'F_N_mag': |F_N|
        - 'F_N_real': Re(F_N) in frame where φ_A = 0
        - 'F_N_imag': Im(F_N) in frame where φ_A = 0
        - 'phi_N_relative': Phase of F_N relative to F_A
        
    Notes
    -----
    The derivation uses complex vector subtraction F_N = F_T - F_A.
    In the frame where φ_A = 0:
        F_A = |F_A| (real, positive)
        F_T = |F_T| × exp(i·Δφ) = |F_T|·cos(Δφ) + i·|F_T|·sin(Δφ)
        F_N = F_T - F_A = (|F_T|·cos(Δφ) - |F_A|) + i·|F_T|·sin(Δφ)
    """
    
    # F_N components in frame where φ_A = 0
    F_N_real = F_T_mag * np.cos(delta_phi) - F_A_mag
    F_N_imag = F_T_mag * np.sin(delta_phi)
    
    # Magnitude and phase
    F_N_mag = np.sqrt(F_N_real**2 + F_N_imag**2)
    phi_N_relative = np.arctan2(F_N_imag, F_N_real)
    
    return {
        'F_N_mag': F_N_mag,
        'F_N_real': F_N_real,
        'F_N_imag': F_N_imag,
        'phi_N_relative': phi_N_relative,
    }


def add_poisson_noise(
    intensity: np.ndarray,
    max_counts: float = None,
    seed: int = None,
    verbose: bool = True
) -> np.ndarray:
    """
    Add Poisson counting noise to a diffraction intensity pattern.
    
    This simulates the fundamental photon counting statistics of X-ray detection.
    The intensity is scaled so that max_counts is the maximum expected count,
    then Poisson-sampled.
    
    Parameters
    ----------
    intensity : np.ndarray
        Input intensity array I = |A|² (any shape)
        
    max_counts : float, optional
        Maximum expected photon count (at the brightest pixel).
        If None, randomly chosen from 10^(2.9 to 4.0) ≈ 800-10000.
        Higher values = less relative noise.
        
    seed : int, optional
        Random seed for reproducibility
        
    verbose : bool
        If True, print noise statistics
        
    Returns
    -------
    intensity_noisy : np.ndarray
        Poisson-sampled intensity (same shape as input, dtype float64)
        
    Notes
    -----
    The Poisson distribution has mean = variance = λ (the expected counts).
    Signal-to-noise ratio: SNR = λ / sqrt(λ) = sqrt(λ)
    So brighter regions have higher absolute noise but better SNR.
    
    Example
    -------
    >>> I_clean = np.abs(diffraction)**2
    >>> I_noisy = add_poisson_noise(I_clean, max_counts=5000)
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # Choose random max counts if not specified
    if max_counts is None:
        # Random scale: 10^(2.9 to 4.0) → ~800 to 10000 max counts
        log_scale = np.random.uniform(2.9, 4.0)
        max_counts = 10 ** log_scale
    
    # Scale intensity so maximum = max_counts
    I_max = intensity.max()
    if I_max < 1e-20:
        # Avoid division by zero for empty patterns
        if verbose:
            print("    Warning: intensity is essentially zero, returning zeros")
        return np.zeros_like(intensity)
    
    scale_factor = max_counts / I_max
    intensity_scaled = intensity * scale_factor
    
    # Apply Poisson sampling
    # np.random.poisson expects non-negative values
    intensity_noisy = np.random.poisson(lam=intensity_scaled).astype(np.float64)
    
    if verbose:
        # Compute statistics
        total_counts = intensity_noisy.sum()
        nonzero_fraction = (intensity_noisy > 0).sum() / intensity_noisy.size
        
        # Estimate SNR at bright pixels (where signal > 10% of max)
        bright_mask = intensity_noisy > 0.1 * intensity_noisy.max()
        if bright_mask.sum() > 0:
            bright_values = intensity_noisy[bright_mask]
            mean_snr = np.mean(np.sqrt(bright_values))  # SNR ≈ sqrt(counts)
        else:
            mean_snr = 0
        
        print(f"\n  Poisson noise applied:")
        print(f"    Max counts (scale): {max_counts:.0f}")
        print(f"    Total counts: {total_counts:.2e}")
        print(f"    Non-zero pixels: {nonzero_fraction:.1%}")
        print(f"    Mean SNR at bright pixels: {mean_snr:.1f}")
    
    return intensity_noisy


def create_correlated_noise_2d(
    size: int,
    correlation_length: float = None,
    seed: int = None
) -> np.ndarray:
    """
    Create a 2D array of spatially correlated noise using Robinson's method.
    
    This generates smooth, random spatial variations by convolving white noise
    with a Gaussian kernel. The correlation length controls the smoothness.
    
    Named after Robinson's approach used in BCDI phase pattern generation.
    
    Parameters
    ----------
    size : int
        Size of the output array (size × size)
        
    correlation_length : float, optional
        Correlation length in normalized units (0 to 1, where 1 = full array).
        Larger values → smoother variations.
        If None, randomly chosen from 0.1 to 0.3.
        
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    noise : np.ndarray
        2D array of correlated noise, normalized to range [-1, 1]
        
    Notes
    -----
    The correlation length in pixels is: L_pixels = correlation_length × size
    
    Typical values:
    - 0.05-0.1: Fine-scale variations (many features)
    - 0.2-0.3: Medium-scale variations
    - 0.4-0.6: Large-scale variations (few smooth features)
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    if correlation_length is None:
        correlation_length = np.random.uniform(0.1, 0.3)
    
    # Generate white noise
    white_noise = np.random.normal(size=(size, size))
    
    # Create Gaussian kernel for convolution
    # Coordinates in normalized units [-1, 1]
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    
    # Gaussian kernel with specified correlation length
    kernel = np.exp(-(xx**2 + yy**2) / (2 * correlation_length**2))
    
    # Convolve to create correlated noise
    noise = fftconvolve(kernel, white_noise, mode='same')
    
    # Normalize to [-1, 1]
    noise = noise - noise.mean()
    noise = noise / (np.abs(noise).max() + 1e-10)
    
    return noise


def add_correlated_module_noise(
    particle: np.ndarray,
    noise_amplitude: float = None,
    correlation_length: float = None,
    preserve_total_density: bool = True,
    seed: int = None,
    verbose: bool = True
) -> np.ndarray:
    """
    Add spatially correlated noise to the real-space particle amplitude.
    
    This simulates inhomogeneities in the electron density that might arise from:
    - Composition variations not captured by the core-shell model
    - Surface roughness or faceting
    - Lattice defects
    - Thermal vibrations (Debye-Waller effects)
    
    The noise is multiplicative and only applied where the particle exists.
    
    Parameters
    ----------
    particle : np.ndarray
        Composition-resolved particle of shape (2, Ny, Nx), can be complex
        (if displacement/phase has been applied)
        
    noise_amplitude : float, optional
        Amplitude of the noise as a fraction of the local density.
        E.g., 0.2 means ±20% variation.
        If None, randomly chosen from 0.1 to 0.3.
        
    correlation_length : float, optional
        Spatial correlation length (see create_correlated_noise_2d).
        If None, randomly chosen.
        
    preserve_total_density : bool, optional
        If True (default), renormalize so that Ni + Fe = original total at each pixel.
        This ensures atomic fractions still sum to 1.0 inside the particle.
        If False, total density can vary (physical if representing occupancy/DW factor).
        
    seed : int, optional
        Random seed for reproducibility
        
    verbose : bool
        If True, print noise information
        
    Returns
    -------
    particle_noisy : np.ndarray
        Particle with correlated noise added to the amplitude.
        Same shape and dtype as input.
        
    Notes
    -----
    With preserve_total_density=True:
        - The Ni/Fe RATIO varies spatially (compositional inhomogeneity)
        - But Ni + Fe = 1.0 everywhere inside the particle
        - This is physically correct for atomic fractions
        
    With preserve_total_density=False:
        - Total density varies (like a Debye-Waller or occupancy factor)
        - Ni + Fe can exceed 1.0 in some regions
        - Use this for modeling density variations, not composition variations
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    if noise_amplitude is None:
        noise_amplitude = np.random.uniform(0.1, 0.3)
    
    grid_size = particle.shape[1]
    
    # Create correlated noise pattern (normalized to [-1, 1])
    noise = create_correlated_noise_2d(grid_size, correlation_length, seed=None)
    
    # Create multiplicative factor: 1 + amplitude × noise
    noise_factor = 1.0 + noise_amplitude * noise
    
    # Handle both real and complex particles
    if np.iscomplexobj(particle):
        # Separate amplitude and phase
        amplitude = np.abs(particle)
        phase = np.angle(particle)
        
        # Apply noise to amplitude
        amplitude_noisy = amplitude * noise_factor[np.newaxis, :, :]
        
        if preserve_total_density:
            # Renormalize so Ni + Fe = original total at each pixel
            original_total = amplitude.sum(axis=0)  # Shape: (Ny, Nx)
            noisy_total = amplitude_noisy.sum(axis=0)
            
            # Avoid division by zero outside particle
            valid_mask = noisy_total > 1e-10
            
            # Renormalization factor
            renorm = np.ones_like(noisy_total)
            renorm[valid_mask] = original_total[valid_mask] / noisy_total[valid_mask]
            
            amplitude_noisy = amplitude_noisy * renorm[np.newaxis, :, :]
        
        # Ensure no negative amplitudes
        amplitude_noisy = np.clip(amplitude_noisy, 0, None)
        
        # Reconstruct complex particle
        particle_noisy = amplitude_noisy * np.exp(1j * phase)
        
    else:
        # Real particle
        particle_noisy = particle * noise_factor[np.newaxis, :, :]
        
        if preserve_total_density:
            # Renormalize so Ni + Fe = original total at each pixel
            original_total = particle.sum(axis=0)
            noisy_total = particle_noisy.sum(axis=0)
            
            valid_mask = noisy_total > 1e-10
            renorm = np.ones_like(noisy_total)
            renorm[valid_mask] = original_total[valid_mask] / noisy_total[valid_mask]
            
            particle_noisy = particle_noisy * renorm[np.newaxis, :, :]
        
        # Ensure no negative densities
        particle_noisy = np.clip(particle_noisy, 0, None)
    
    if verbose:
        actual_corr_len = correlation_length if correlation_length is not None else "random"
        print(f"\n  Correlated module noise applied:")
        print(f"    Noise amplitude: ±{noise_amplitude*100:.1f}%")
        print(f"    Correlation length: {actual_corr_len}")
        print(f"    Noise factor range: [{noise_factor.min():.3f}, {noise_factor.max():.3f}]")
        print(f"    Preserve total density: {preserve_total_density}")
        
        # Check total density after noise
        if np.iscomplexobj(particle_noisy):
            total_after = np.abs(particle_noisy).sum(axis=0)
        else:
            total_after = particle_noisy.sum(axis=0)
        
        inside_particle = total_after > 0.01
        if inside_particle.sum() > 0:
            print(f"    Total density range (inside particle): [{total_after[inside_particle].min():.3f}, {total_after[inside_particle].max():.3f}]")
    
    return particle_noisy


# =============================================================================
# RANDOM STRAIN FIELD (ROBINSON-STYLE)
# =============================================================================
#
# Beyond the analytic core-shell mismatch model, we can add random strain
# components to create more diverse training data. This uses the same
# convolution approach as the correlated noise, but interprets the result
# as a displacement field.
#
# Physics:
# - The random strain represents additional displacements beyond the
#   ideal core-shell mismatch (defects, relaxation, thermal effects)
# - It's added to the analytic displacement before computing the phase
# - The correlation length controls the "scale" of strain features
# - The strain amplitude controls the magnitude of displacements
# =============================================================================

def create_random_strain_field(
    grid_size: int,
    pixel_size: float,
    displacement_amplitude: float = None,
    correlation_length: float = None,
    seed: int = None,
    verbose: bool = True
) -> np.ndarray:
    """
    Create a smooth random displacement field using Robinson's method.
    
    This generates spatially correlated random displacements that can be added
    to the analytic core-shell mismatch model to create more diverse strain
    patterns for training data.
    
    Parameters
    ----------
    grid_size : int
        Size of the grid (grid_size × grid_size)
        
    pixel_size : float
        Pixel size in Ångströms (for reference in verbose output)
        
    displacement_amplitude : float, optional
        Maximum displacement amplitude in Ångströms.
        If None, randomly chosen from 0.3 to 1.0 Å.
        This should be comparable to or smaller than the analytic core-shell
        mismatch displacement (~0.9 Å for 0.6% mismatch).
        
    correlation_length : float, optional
        Correlation length in normalized units (0 to 1).
        Larger values → larger-scale strain features.
        If None, randomly chosen from 0.15 to 0.35.
        
    seed : int, optional
        Random seed for reproducibility
        
    verbose : bool
        If True, print strain field statistics
        
    Returns
    -------
    displacement : np.ndarray
        2D array of displacement values in Ångströms.
        This is the component of displacement along the Q direction (u_Q).
        
    Notes
    -----
    "Robinson-style" refers to the convolution method for generating smooth
    random fields, commonly used in BCDI simulation. The approach:
    1. Generate white noise
    2. Convolve with Gaussian kernel (smoothing)
    3. Scale to desired displacement amplitude
    
    The physical interpretation is that the crystal has smooth displacement
    variations superimposed on any systematic strain (like core-shell mismatch).
    These might arise from defects, thermal effects, or incomplete relaxation.
    
    The displacement amplitude should typically be comparable to or smaller
    than the systematic strain contribution to avoid unrealistic distortions.
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # Default: random displacement amplitude that adds diversity without overwhelming
    # the analytic strain. The analytic core-shell gives ~0.9 Å max.
    # With Q_Bragg ≈ 3 Å⁻¹, phase = Q × u, so:
    #   0.3 Å → ~1 rad phase variation
    #   1.0 Å → ~3 rad phase variation (can cause wrapping!)
    # We use a smaller range to add texture without phase wrapping.
    if displacement_amplitude is None:
        displacement_amplitude = np.random.uniform(0.1, 0.4)
    
    # Default: moderate correlation length for visible but not overwhelming features
    if correlation_length is None:
        correlation_length = np.random.uniform(0.15, 0.35)
    
    # Create correlated noise pattern (normalized to [-1, 1])
    noise = create_correlated_noise_2d(grid_size, correlation_length, seed=None)
    
    # Scale directly to desired displacement amplitude
    # The noise is in [-1, 1], so multiply by amplitude to get [-amp, +amp]
    displacement = displacement_amplitude * noise
    
    # Compute effective strain for informational purposes
    # strain ≈ du/dr, estimate from displacement gradient
    grad_y, grad_x = np.gradient(displacement, pixel_size)
    strain_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    effective_strain_rms = np.std(strain_magnitude)
    
    if verbose:
        print(f"\n  Random strain field created (Robinson-style):")
        print(f"    Displacement amplitude: ±{displacement_amplitude:.3f} Å")
        print(f"    Correlation length: {correlation_length:.2f} (normalized)")
        print(f"    Correlation length: {correlation_length * grid_size * pixel_size:.1f} Å (physical)")
        print(f"    Displacement range: [{displacement.min():.3f}, {displacement.max():.3f}] Å")
        print(f"    Displacement RMS: {np.std(displacement):.3f} Å")
        print(f"    Effective strain RMS: {effective_strain_rms*100:.4f}%")
    
    return displacement


def combine_displacement_fields(
    displacement_analytic: np.ndarray,
    displacement_random: np.ndarray,
    particle_mask: np.ndarray = None,
    verbose: bool = True
) -> np.ndarray:
    """
    Combine analytic (core-shell) and random displacement fields.
    
    Parameters
    ----------
    displacement_analytic : np.ndarray
        Displacement field from core-shell mismatch model (Å)
        
    displacement_random : np.ndarray
        Random displacement field from Robinson-style generation (Å)
        
    particle_mask : np.ndarray, optional
        Boolean mask indicating particle interior. If provided, the
        random displacement is only applied inside the particle.
        
    verbose : bool
        If True, print combined field statistics
        
    Returns
    -------
    displacement_combined : np.ndarray
        Sum of analytic and random displacements (Å)
    """
    
    if particle_mask is not None:
        # Apply random displacement only inside particle
        displacement_random_masked = displacement_random * particle_mask
    else:
        displacement_random_masked = displacement_random
    
    displacement_combined = displacement_analytic + displacement_random_masked
    
    if verbose:
        print(f"\n  Combined displacement field:")
        print(f"    Analytic contribution range: [{displacement_analytic.min():.3f}, {displacement_analytic.max():.3f}] Å")
        print(f"    Random contribution range: [{displacement_random_masked.min():.3f}, {displacement_random_masked.max():.3f}] Å")
        print(f"    Combined range: [{displacement_combined.min():.3f}, {displacement_combined.max():.3f}] Å")
    
    return displacement_combined


# =============================================================================
# RECONSTRUCTION (INVERSE FFT)
# =============================================================================

def reconstruct(diffraction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct the object from its diffraction pattern using inverse FFT.
    
    Parameters
    ----------
    diffraction : np.ndarray
        Complex diffraction amplitude (with q=0 at center)
        
    Returns
    -------
    amplitude : np.ndarray
        Magnitude of the reconstruction: |IFFT[A(q)]|
        
    phase : np.ndarray
        Phase of the reconstruction: arg(IFFT[A(q)])
        Values are in radians, range [-π, π].
    """
    
    unshifted = np.fft.ifftshift(diffraction)
    reconstruction = np.fft.ifft2(unshifted)
    
    amplitude = np.abs(reconstruction)
    phase = np.angle(reconstruction)
    
    return amplitude, phase


# =============================================================================
# BLOCK EXTRACTION
# =============================================================================

def extract_blocks(particle: np.ndarray, block_size: int = 16) -> np.ndarray:
    """
    Divide a particle into non-overlapping square blocks.
    
    Parameters
    ----------
    particle : np.ndarray
        Either 2D (N, N) or 3D (2, N, N) array
        
    block_size : int
        Size of each square block (default 16)
        
    Returns
    -------
    blocks : np.ndarray
        For 2D input: shape (n_y, n_x, block_size, block_size)
        For 3D input: shape (n_y, n_x, 2, block_size, block_size)
    """
    
    if particle.ndim == 3:
        n_species, n_rows, n_cols = particle.shape
        is_composition_resolved = True
    elif particle.ndim == 2:
        n_rows, n_cols = particle.shape
        is_composition_resolved = False
    else:
        raise ValueError(f"particle must be 2D or 3D, got shape {particle.shape}")
    
    if n_rows % block_size != 0 or n_cols % block_size != 0:
        raise ValueError(
            f"Particle spatial shape ({n_rows}, {n_cols}) must be divisible by "
            f"block_size ({block_size})"
        )
    
    n_blocks_y = n_rows // block_size
    n_blocks_x = n_cols // block_size
    
    if is_composition_resolved:
        blocks = particle.reshape(n_species, n_blocks_y, block_size, n_blocks_x, block_size)
        blocks = blocks.transpose(1, 3, 0, 2, 4)
    else:
        blocks = particle.reshape(n_blocks_y, block_size, n_blocks_x, block_size)
        blocks = blocks.transpose(0, 2, 1, 3)
    
    return blocks.copy()


def process_block(block: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Process a single block through the FFT → inverse FFT pipeline.
    
    Parameters
    ----------
    block : np.ndarray
        Either 2D or 3D (2, block_size, block_size) array
        
    Returns
    -------
    result : dict
        Dictionary with block, effective_density, diffraction, intensity,
        amplitude, and phase.
    """
    
    if block.ndim == 3:
        effective_density = np.sum(block, axis=0)
    else:
        effective_density = block
    
    diffraction = compute_diffraction(effective_density)
    intensity = get_diffraction_intensity(diffraction, log_scale=True)
    amplitude, phase = reconstruct(diffraction)
    
    return {
        'block': block,
        'effective_density': effective_density,
        'diffraction': diffraction,
        'intensity': intensity,
        'amplitude': amplitude,
        'phase': phase
    }


def process_block_at_energy(
    block: np.ndarray,
    energy: float,
    scattering_factors: ScatteringFactors
) -> Dict[str, Any]:
    """
    Process a single block through the energy-dependent FFT pipeline.
    
    Parameters
    ----------
    block : np.ndarray
        Composition-resolved block of shape (2, block_size, block_size)
        
    energy : float
        X-ray energy in eV
        
    scattering_factors : ScatteringFactors
        ScatteringFactors object
        
    Returns
    -------
    result : dict
        Dictionary with block, energy, effective_density, diffraction, intensity,
        amplitude, phase, and scattering factor info.
    """
    
    if block.ndim != 3:
        raise ValueError("Block must be composition-resolved (3D) for energy-dependent processing")
    
    # Get scattering factors
    f_Ni = scattering_factors.get_scattering_factor('Ni', energy)
    f_Fe = scattering_factors.get_scattering_factor('Fe', energy)
    
    # Build complex effective density
    effective_density = block[SPECIES_NI] * f_Ni + block[SPECIES_FE] * f_Fe
    
    # Compute diffraction
    diffraction = np.fft.fftshift(np.fft.fft2(effective_density))
    intensity = get_diffraction_intensity(diffraction, log_scale=True)
    
    # Reconstruct
    amplitude, phase = reconstruct(diffraction)
    
    return {
        'block': block,
        'energy': energy,
        'effective_density': effective_density,
        'diffraction': diffraction,
        'intensity': intensity,
        'amplitude': amplitude,
        'phase': phase,
        'f_Ni': f_Ni,
        'f_Fe': f_Fe
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
# Note: Visualization functions have been moved to visualization.py

def print_particle_info(info: Dict[str, Any]) -> None:
    """Print detailed information about a particle."""
    
    print("=" * 70)
    print("CORE-SHELL PARTICLE INFORMATION (Composition-Resolved)")
    print("=" * 70)
    print(f"Grid size:        {info['grid_size']} × {info['grid_size']} pixels")
    print(f"Pixel size:       {info['pixel_size']:.1f} Å")
    print(f"Center:           ({info['center'][0]:.1f}, {info['center'][1]:.1f})")
    print("-" * 70)
    print("GEOMETRY (pixels):")
    print(f"  Outer radius:    {info['outer_radius']:.1f} pixels")
    print(f"  Core radius:     {info['core_radius']:.1f} pixels")
    print(f"  Shell thickness: {info['shell_thickness']:.1f} pixels")
    print("-" * 70)
    print("PHYSICAL DIMENSIONS:")
    print(f"  Grid extent:     {info['physical_extent_nm']:.1f} nm ({info['physical_extent_angstrom']:.0f} Å)")
    print(f"  Particle diam:   {info['physical_diameter_nm']:.1f} nm")
    print(f"  Outer radius:    {info['physical_outer_radius_angstrom']:.0f} Å")
    print(f"  Core radius:     {info['physical_core_radius_angstrom']:.0f} Å")
    print("-" * 70)
    print("RECIPROCAL SPACE:")
    print(f"  Q_max:           {info['q_max']:.4f} Å⁻¹ (at edge of diffraction pattern)")
    print("-" * 70)
    print("COMPOSITION:")
    core = info['core_composition']
    shell = info['shell_composition']
    print(f"  Core region:     Ni = {core['Ni']:.1%}, Fe = {core['Fe']:.1%}")
    print(f"  Shell region:    Ni = {shell['Ni']:.1%}, Fe = {shell['Fe']:.1%}")
    print("-" * 70)
    print("AREAS:")
    print(f"  Core area:       {info['core_area']} pixels ({info['core_fraction']:.1%} of particle)")
    print(f"  Shell area:      {info['shell_area']} pixels ({info['shell_fraction']:.1%} of particle)")
    print(f"  Total area:      {info['total_area']} pixels")
    print("=" * 70)


def get_total_density(particle: np.ndarray) -> np.ndarray:
    """Get total density from a composition-resolved particle."""
    
    if particle.ndim == 3:
        return np.sum(particle, axis=0)
    return particle


def verify_reconstruction(original: np.ndarray, amplitude: np.ndarray) -> Dict[str, float]:
    """Verify that reconstruction matches original."""
    
    original_2d = get_total_density(original)
    
    if original_2d.max() < 1e-10:
        return {
            'max_error': float(amplitude.max()),
            'mean_error': float(amplitude.mean()),
            'correlation': np.nan
        }
    
    orig_norm = original_2d / (original_2d.max() + 1e-10)
    recon_norm = amplitude / (amplitude.max() + 1e-10)
    
    diff = np.abs(orig_norm - recon_norm)
    
    orig_flat = orig_norm.flatten()
    recon_flat = recon_norm.flatten()
    
    if np.std(orig_flat) < 1e-10 or np.std(recon_flat) < 1e-10:
        correlation = np.nan
    else:
        correlation = float(np.corrcoef(orig_flat, recon_flat)[0, 1])
    
    metrics = {
        'max_error': float(diff.max()),
        'mean_error': float(diff.mean()),
        'correlation': correlation
    }
    
    return metrics
