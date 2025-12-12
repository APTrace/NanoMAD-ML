#!/usr/bin/env python3
"""
mad_loss.py

Loss function design for the MAD (Multi-wavelength Anomalous Diffraction) CNN.

This module implements a physics-informed loss function that combines:
1. Weighted MSE loss (intensity-aware)
2. Unit circle constraint (sin²+cos²=1)
3. Intensity reconstruction loss (physics self-consistency check)

The loss function is designed to train a CNN that predicts:
    Input:  [batch, 16, 16, 8] intensity patches + [batch, 8] f' + [batch, 8] f''
    Output: [batch, 16, 16, 4] → |F_T|, |F_A|, sin(Δφ), cos(Δφ)

Author: Claude (Anthropic) + Thomas
Date: December 2024
Based on: NanoMAD expert recommendations (NanoMAD_CNN_Design.md)
"""

import numpy as np
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn


# =============================================================================
# PHYSICS BACKGROUND
# =============================================================================
"""
THE MAD EQUATION
================

The Multi-wavelength Anomalous Diffraction equation describes how diffraction
intensity varies with X-ray energy near an absorption edge:

    I(Q, E) = |F_T|² + (f'² + f''²)|F_A/f₀|² + 2|F_T||F_A/f₀|·[f'·cos(Δφ) + f''·sin(Δφ)]
              ─────   ─────────────────────   ──────────────────────────────────────────
              Term 1        Term 2                            Term 3

Where:
    - |F_T|  : Total structure factor magnitude (Thomson-weighted, energy-independent)
    - |F_A|  : Anomalous structure factor magnitude (Thomson-weighted)  
    - Δφ     : Phase difference = arg(F_T) - arg(F_A)
    - f'(E)  : Real anomalous dispersion correction (varies with energy)
    - f''(E) : Imaginary anomalous correction (varies with energy)
    - f₀(Q)  : Thomson scattering factor (varies with scattering angle)

THE INVERSE PROBLEM
===================

NanoMAD solves this equation by iterative fitting: given I(E) at multiple 
energies, find (|F_T|, |F_A|, Δφ) that best reproduce the observed intensities.

Our CNN learns this inverse mapping directly:

    Forward (physics):  (|F_T|, |F_A|, Δφ, f', f'') → I(E)
    Inverse (CNN):      I(E), f', f'' → (|F_T|, |F_A|, sin(Δφ), cos(Δφ))

WHY A CUSTOM LOSS FUNCTION?
===========================

A simple MSE loss would work, but we can do better by incorporating physics:

1. INTENSITY WEIGHTING
   - Pixels with higher intensity have better signal-to-noise ratio
   - Errors in bright regions matter more than errors in dark regions
   - Mimics Poisson statistics: σ(I) ∝ √I, so weight by 1/σ ∝ 1/√I
   
2. UNIT CIRCLE CONSTRAINT
   - We predict sin(Δφ) and cos(Δφ) separately
   - They should satisfy sin² + cos² = 1
   - Without constraint, the network could predict (sin=0.5, cos=0.5) which is invalid
   - Soft constraint: penalize deviations from the unit circle
   
3. INTENSITY RECONSTRUCTION
   - The ultimate test: do our predictions correctly reproduce the input intensities?
   - Uses the MAD equation to reconstruct I from (|F_T|, |F_A|, sin(Δφ), cos(Δφ))
   - If reconstruction matches input, predictions are self-consistent
   - This is the most powerful regularizer because it ties predictions to physics

THE DEGENERACY PROBLEM
======================

The MAD equation has sign degeneracies:
    (F_T, Δφ) ↔ (-F_T, Δφ+π) give the same intensity
    (F_A, Δφ) ↔ (-F_A, Δφ+π) give the same intensity

NanoMAD handles this by enforcing F_T > 0 and F_A > 0 after fitting.

For the CNN:
    - Use Softplus activation for |F_T| and |F_A| → always positive
    - Use Tanh activation for sin(Δφ) and cos(Δφ) → range [-1, 1]

This removes the sign ambiguity at the architecture level.
"""


# =============================================================================
# LOSS FUNCTION IMPLEMENTATION
# =============================================================================

class MADLoss(nn.Module):
    """
    Physics-informed loss function for MAD parameter prediction.

    Combines three terms:

    1. Weighted MSE Loss:
       L_mse = Σ w(x,y) · ||pred - target||²
       where w(x,y) depends on weight_scheme parameter

    2. Unit Circle Constraint:
       L_unit = Σ (sin²(Δφ) + cos²(Δφ) - 1)²

    3. Intensity Reconstruction Loss (optional):
       L_recon = Σ w(x,y) · ||I_reconstructed - I_observed||² / ||I_observed||²

    Total loss:
       L = L_mse + λ_unit · L_unit + λ_recon · L_recon

    Parameters
    ----------
    lambda_unit : float
        Weight for unit circle constraint. Default 0.1.
        Higher values enforce sin² + cos² = 1 more strictly.

    lambda_recon : float
        Weight for intensity reconstruction loss. Default 0.1.
        Higher values enforce physics self-consistency more strictly.

    lambda_fa : float
        Weight for additional F_A-specific loss term. Default 0.0 (disabled).
        This adds an EXTRA loss term for the F_A channel with UNIFORM weighting
        (no intensity dependence), forcing the network to learn F_A in ALL regions.
        Recommended starting value: 1.0. Higher values emphasize F_A more.

        Why this helps: The standard intensity-weighted MSE loss causes the network
        to ignore low-intensity regions where F_A may be small but non-zero. The
        extra F_A term with uniform weighting ensures all F_A pixels contribute
        equally to the loss, preventing "holes" in F_A predictions.

    use_recon_loss : bool
        Whether to include intensity reconstruction loss.
        Requires f', f'', and optionally f₀ to be provided.
        Default True (recommended for physics consistency).

    use_log_mse : bool
        If True, compute MSE on log(magnitude) for |F_T| and |F_A|.
        This helps with the large dynamic range of structure factors.
        Default False.

    weight_scheme : str
        How to weight pixels based on intensity. Options:
        - 'sqrt': w = sqrt(I) (original, heavily favors bright pixels)
        - 'log': w = log(1+I) (compressed, more balanced - RECOMMENDED)
        - 'uniform': w = 1 (no intensity weighting)
        - 'sqrt_floor': w = max(sqrt(I), min_weight) (sqrt with minimum)
        Default 'sqrt' for backward compatibility.

        IMPORTANT: The original 'sqrt' scheme causes the CNN to ignore
        low-intensity (outer) regions. Use 'log' for balanced training.

    min_weight_fraction : float
        For 'sqrt_floor' scheme, the minimum weight as fraction of mean.
        Default 0.1 (10% of mean weight).

    Example
    -------
    >>> # Recommended settings for balanced training with F_A emphasis:
    >>> criterion = MADLoss(lambda_unit=0.1, weight_scheme='log', lambda_fa=1.0)
    >>> loss, loss_dict = criterion(
    ...     pred=model_output,      # [batch, 16, 16, 4]
    ...     target=labels,          # [batch, 16, 16, 4]
    ...     intensities=input_I,    # [batch, 16, 16, 8]
    ...     f_prime=fp,             # [batch, 8]
    ...     f_double_prime=fs,      # [batch, 8]
    ... )
    >>> print(loss_dict)
    {'mse': 0.0023, 'unit': 0.0012, 'recon': 0.0045, 'total': 0.0034}
    """

    # Available weighting schemes
    WEIGHT_SCHEMES = ['sqrt', 'log', 'uniform', 'sqrt_floor']

    def __init__(
        self,
        lambda_unit: float = 0.1,
        lambda_recon: float = 0.1,
        lambda_fa: float = 0.0,
        use_recon_loss: bool = True,
        use_log_mse: bool = False,
        log_scale_magnitudes: bool = False,
        weight_scheme: str = 'sqrt',
        min_weight_fraction: float = 0.1
    ):
        super().__init__()
        self.lambda_unit = lambda_unit
        self.lambda_recon = lambda_recon
        self.lambda_fa = lambda_fa
        self.use_recon_loss = use_recon_loss
        self.use_log_mse = use_log_mse
        self.log_scale_magnitudes = log_scale_magnitudes

        # Validate weight_scheme
        if weight_scheme not in self.WEIGHT_SCHEMES:
            raise ValueError(
                f"weight_scheme must be one of {self.WEIGHT_SCHEMES}, "
                f"got '{weight_scheme}'"
            )
        self.weight_scheme = weight_scheme
        self.min_weight_fraction = min_weight_fraction

        # =====================================================================
        # WEIGHT SCHEME EXPLANATION
        # =====================================================================
        # The weight_scheme parameter controls how pixels are weighted by intensity:
        #
        # 'sqrt' (original - PROBLEMATIC):
        #   w = sqrt(I), normalized to mean=1
        #   For I in [100, 1e6]: weights ratio ~100:1
        #   Problem: Low-intensity pixels nearly invisible to loss
        #   This causes the CNN to ignore outer regions of diffraction patterns
        #
        # 'log' (RECOMMENDED for balanced training):
        #   w = log(1+I), normalized to mean=1
        #   For I in [100, 1e6]: weights ratio ~3:1
        #   Much more balanced across intensity ranges
        #   Allows CNN to learn from both bright and dim regions
        #
        # 'uniform':
        #   w = 1 (all pixels equal)
        #   May hurt accuracy in high-SNR regions
        #   Use for testing or if you want completely equal treatment
        #
        # 'sqrt_floor':
        #   w = max(sqrt(I), min_fraction * mean(sqrt(I)))
        #   Compromise: sqrt weighting with guaranteed minimum
        #   Ensures even dark pixels contribute to loss
        # =====================================================================

        # =====================================================================
        # LOG-SCALED MAGNITUDES MODE
        # =====================================================================
        # When log_scale_magnitudes=True, the targets |F_T| and |F_A| have been
        # transformed with log1p(x) = log(1 + x) before training.
        #
        # This is done because raw magnitudes can be very large (0 to 80,000)
        # which makes MSE loss unstable. Log transform maps to ~[0, 11.3].
        #
        # IMPORTANT: When log_scale_magnitudes=True, this loss function will:
        # 1. Compute MSE in log space (which is what we want for training)
        # 2. Transform predictions BACK to linear space for intensity
        #    reconstruction loss (using expm1)
        #
        # At inference time, predictions will be in log space!
        # Use expm1(prediction) to get back to linear scale:
        #   F_T_linear = np.expm1(F_T_log)  # = exp(F_T_log) - 1
        # =====================================================================

    def _compute_weights(self, intensities: torch.Tensor) -> torch.Tensor:
        """
        Compute per-pixel weights based on intensity using the selected scheme.

        Parameters
        ----------
        intensities : torch.Tensor
            Input intensity patches, shape [batch, H, W, n_energies]

        Returns
        -------
        weights : torch.Tensor
            Per-pixel weights, shape [batch, H, W], normalized to mean=1
        """
        # Total intensity per pixel (sum over energy channels)
        total_I = intensities.sum(dim=-1)  # [batch, H, W]

        if self.weight_scheme == 'uniform':
            # All pixels equal weight
            weights = torch.ones_like(total_I)

        elif self.weight_scheme == 'sqrt':
            # Original scheme: sqrt(I)
            # WARNING: This heavily down-weights low-intensity pixels
            weights = torch.sqrt(total_I + 1e-6)

        elif self.weight_scheme == 'log':
            # Recommended: log(1+I) - much more balanced
            # For I in [100, 1e6]: ratio ~3:1 instead of ~100:1
            weights = torch.log1p(total_I)

        elif self.weight_scheme == 'sqrt_floor':
            # Sqrt with minimum floor
            weights = torch.sqrt(total_I + 1e-6)
            min_w = self.min_weight_fraction * weights.mean()
            weights = torch.clamp(weights, min=min_w)

        else:
            # Should not reach here due to __init__ validation
            raise ValueError(f"Unknown weight_scheme: {self.weight_scheme}")

        # Normalize to mean=1 (doesn't change loss scale)
        weights = weights / (weights.mean() + 1e-10)

        return weights

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        intensities: torch.Tensor,
        f_prime: torch.Tensor = None,
        f_double_prime: torch.Tensor = None,
        f0: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the total loss.

        Parameters
        ----------
        pred : torch.Tensor
            Model predictions, shape [batch, H, W, 4]
            Channels: |F_T|, |F_A|, sin(Δφ), cos(Δφ)

        target : torch.Tensor
            Ground truth labels, shape [batch, H, W, 4]
            Same channel order as pred.

        intensities : torch.Tensor
            Input intensity patches, shape [batch, H, W, n_energies]
            Used for weighting and reconstruction loss.

        f_prime : torch.Tensor, optional
            f'(E) at each energy, shape [batch, n_energies] or [n_energies]
            Required if use_recon_loss=True.

        f_double_prime : torch.Tensor, optional
            f''(E) at each energy, shape [batch, n_energies] or [n_energies]
            Required if use_recon_loss=True.

        f0 : torch.Tensor, optional
            Thomson factor f₀(Q) at each pixel, shape [batch, H, W]
            If not provided, assumes f₀=1 (valid for small Q-range patches).

        Returns
        -------
        total_loss : torch.Tensor
            Scalar loss value for backpropagation.

        loss_dict : dict
            Dictionary with individual loss components for logging.
        """

        # =====================================================================
        # 1. INTENSITY-WEIGHTED MSE LOSS
        # =====================================================================
        #
        # The weighting scheme is controlled by self.weight_scheme:
        # - 'sqrt': original, heavily favors bright pixels (PROBLEMATIC)
        # - 'log': more balanced, recommended for training
        # - 'uniform': no weighting
        # - 'sqrt_floor': sqrt with minimum floor
        #
        # See __init__ docstring and _compute_weights for details.
        #

        # Compute weights using the selected scheme
        weights = self._compute_weights(intensities)

        # Compute MSE per channel
        if self.use_log_mse:
            # For magnitudes, use log-space MSE (handles large dynamic range)
            # Only apply to channels 0 and 1 (|F_T| and |F_A|)
            pred_mag = pred[..., :2]
            target_mag = target[..., :2]
            mse_mag = (torch.log(pred_mag + 1) - torch.log(target_mag + 1)) ** 2
            
            # For sin/cos, use regular MSE
            mse_phase = (pred[..., 2:] - target[..., 2:]) ** 2
            
            mse = torch.cat([mse_mag, mse_phase], dim=-1)
        else:
            mse = (pred - target) ** 2  # [batch, H, W, 4]
        
        # Apply intensity weighting
        # Weights are [batch, H, W], need to broadcast to [batch, H, W, 4]
        weighted_mse = mse * weights.unsqueeze(-1)
        loss_mse = weighted_mse.mean()
        
        # =====================================================================
        # 2. UNIT CIRCLE CONSTRAINT
        # =====================================================================
        #
        # Physics motivation:
        # - We predict sin(Δφ) and cos(Δφ) as separate channels
        # - For any valid angle φ: sin²(φ) + cos²(φ) = 1
        # - Without constraint, network could predict invalid combinations
        #
        # Implementation:
        # - Compute sin² + cos² for predictions
        # - Penalize deviation from 1
        # - Use squared penalty for smoothness
        #
        # Note: Alternatively, you could normalize the outputs post-prediction:
        #   sin_norm = sin / sqrt(sin² + cos²)
        #   cos_norm = cos / sqrt(sin² + cos²)
        # But the soft constraint approach lets the network learn to satisfy
        # the constraint naturally, and provides gradient signal.
        #
        
        sin_pred = pred[..., 2]
        cos_pred = pred[..., 3]
        
        # How far from unit circle?
        radius_squared = sin_pred**2 + cos_pred**2
        unit_violation = (radius_squared - 1.0) ** 2
        
        loss_unit = unit_violation.mean()
        
        # =====================================================================
        # 3. INTENSITY RECONSTRUCTION LOSS
        # =====================================================================
        #
        # Physics motivation:
        # - This is the most powerful regularizer!
        # - The MAD equation tells us: given (|F_T|, |F_A|, Δφ, f', f''),
        #   we can compute what the intensity SHOULD be
        # - If our predictions are correct, the reconstructed intensity
        #   should match the input intensity
        # - This ties predictions directly to the physics
        #
        # Even if the network finds a different (F_T, F_A, Δφ) than NanoMAD,
        # if it reconstructs the intensity correctly, it's a valid solution!
        #
        # Implementation:
        # - Extract |F_T|, |F_A|, sin(Δφ), cos(Δφ) from predictions
        # - Use MAD equation to reconstruct intensity at each energy
        # - Compare to input intensity
        # - Weight by intensity (same reasoning as MSE weighting)
        #
        
        loss_recon = torch.tensor(0.0, device=pred.device)
        
        if self.use_recon_loss and f_prime is not None and f_double_prime is not None:
            loss_recon = self._intensity_reconstruction_loss(
                pred=pred,
                intensities=intensities,
                f_prime=f_prime,
                f_double_prime=f_double_prime,
                f0=f0,
                weights=weights
            )

        # =====================================================================
        # 4. F_A-SPECIFIC LOSS (UNIFORM WEIGHTING)
        # =====================================================================
        #
        # Physics motivation:
        # - The intensity-weighted MSE loss causes the network to ignore regions
        #   where F_A is naturally small (but non-zero)
        # - F_A (anomalous structure factor / Ni contribution) can have valid
        #   non-zero values even in low-intensity regions
        # - By adding an EXTRA loss term for F_A with UNIFORM weighting, we
        #   ensure all F_A pixels contribute equally to the loss
        #
        # Implementation:
        # - Extract F_A predictions (channel 1) and targets
        # - Compute MSE without intensity weighting (uniform)
        # - This supplements (doesn't replace) the weighted MSE loss
        #
        # Note: This is an ADDITIVE term, so F_A gets trained from both:
        # 1. The original intensity-weighted MSE (bright regions dominate)
        # 2. This uniform-weighted term (all regions contribute equally)
        #

        loss_fa = torch.tensor(0.0, device=pred.device)

        if self.lambda_fa > 0:
            # F_A is channel 1 (index 1)
            fa_pred = pred[..., 1]
            fa_target = target[..., 1]

            # MSE without intensity weighting (uniform)
            fa_mse = (fa_pred - fa_target) ** 2
            loss_fa = fa_mse.mean()

        # =====================================================================
        # COMBINE LOSSES
        # =====================================================================

        total_loss = (
            loss_mse
            + self.lambda_unit * loss_unit
            + self.lambda_recon * loss_recon
            + self.lambda_fa * loss_fa
        )

        # Build loss dictionary for logging
        loss_dict = {
            'mse': loss_mse.item(),
            'unit': loss_unit.item(),
            'recon': loss_recon.item() if isinstance(loss_recon, torch.Tensor) else 0.0,
            'fa': loss_fa.item() if isinstance(loss_fa, torch.Tensor) else 0.0,
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def _intensity_reconstruction_loss(
        self,
        pred: torch.Tensor,
        intensities: torch.Tensor,
        f_prime: torch.Tensor,
        f_double_prime: torch.Tensor,
        f0: torch.Tensor = None,
        weights: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute intensity reconstruction loss using the MAD equation.
        
        Reconstructs intensity from predictions and compares to observed.
        
        MAD equation:
            I(E) = |F_T|² + (f'² + f''²)|F_A/f₀|² 
                   + 2|F_T||F_A/f₀|·[f'·cos(Δφ) + f''·sin(Δφ)]
        
        NOTE: If self.log_scale_magnitudes=True, predictions are in log space
              and we convert back to linear space for reconstruction.
        """
        
        batch_size, H, W, n_energies = intensities.shape
        
        # Extract predictions
        F_T = pred[..., 0]  # [batch, H, W]
        F_A = pred[..., 1]  # [batch, H, W]
        sin_phi = pred[..., 2]  # [batch, H, W]
        cos_phi = pred[..., 3]  # [batch, H, W]
        
        # =====================================================================
        # CONVERT FROM LOG SPACE IF NEEDED
        # =====================================================================
        # If training with log-scaled magnitudes, predictions are log1p(F).
        # Convert back to linear scale for physics reconstruction:
        #   expm1(x) = exp(x) - 1, the inverse of log1p
        if self.log_scale_magnitudes:
            F_T = torch.expm1(F_T)
            F_A = torch.expm1(F_A)
        
        # Handle f₀ normalization
        # If f₀ not provided, assume f₀ = 1 (valid approximation for small patches)
        if f0 is None:
            F_A_norm = F_A
        else:
            # f0 shape: [batch, H, W]
            F_A_norm = F_A / (f0 + 1e-10)
        
        # Handle f'/f'' shape: could be [batch, n_energies] or [n_energies]
        if f_prime.dim() == 1:
            # Shape [n_energies] → broadcast to all batches
            fp = f_prime.view(1, 1, 1, -1)  # [1, 1, 1, n_energies]
            fs = f_double_prime.view(1, 1, 1, -1)
        else:
            # Shape [batch, n_energies] → add spatial dimensions
            fp = f_prime.view(batch_size, 1, 1, -1)  # [batch, 1, 1, n_energies]
            fs = f_double_prime.view(batch_size, 1, 1, -1)
        
        # Expand spatial tensors for broadcasting with energy dimension
        F_T_exp = F_T.unsqueeze(-1)  # [batch, H, W, 1]
        F_A_norm_exp = F_A_norm.unsqueeze(-1)  # [batch, H, W, 1]
        sin_phi_exp = sin_phi.unsqueeze(-1)  # [batch, H, W, 1]
        cos_phi_exp = cos_phi.unsqueeze(-1)  # [batch, H, W, 1]
        
        # MAD equation: reconstruct intensity at each energy
        # Term 1: |F_T|² (same at all energies)
        term1 = F_T_exp ** 2
        
        # Term 2: (f'² + f''²)|F_A/f₀|²
        term2 = (fp**2 + fs**2) * F_A_norm_exp**2
        
        # Term 3: 2|F_T||F_A/f₀|·[f'·cos(Δφ) + f''·sin(Δφ)]
        term3 = 2 * F_T_exp * F_A_norm_exp * (fp * cos_phi_exp + fs * sin_phi_exp)
        
        # Reconstructed intensity
        I_reconstructed = term1 + term2 + term3  # [batch, H, W, n_energies]
        
        # Compute relative squared error (normalized by observed intensity)
        # This makes the loss scale-invariant
        I_observed = intensities
        
        # Relative squared error: ((I_recon - I_obs) / (I_obs + ε))²
        # Weight by intensity to focus on reliable regions
        rel_error = (I_reconstructed - I_observed) / (I_observed + 1e-6)
        squared_error = rel_error ** 2
        
        # Apply spatial weighting if provided
        if weights is not None:
            # weights: [batch, H, W] → expand to [batch, H, W, n_energies]
            squared_error = squared_error * weights.unsqueeze(-1)
        
        # Mean over all dimensions
        loss = squared_error.mean()
        
        return loss


# =============================================================================
# HELPER FUNCTION: Reconstruct intensity (for validation/debugging)
# =============================================================================

def reconstruct_intensity_torch(
    F_T: torch.Tensor,
    F_A: torch.Tensor,
    sin_phi: torch.Tensor,
    cos_phi: torch.Tensor,
    f_prime: torch.Tensor,
    f_double_prime: torch.Tensor,
    f0: torch.Tensor = None
) -> torch.Tensor:
    """
    Reconstruct intensity from MAD parameters using the MAD equation.
    
    This is useful for:
    - Validation: check if predictions reproduce input intensities
    - Visualization: see what intensity the model thinks it's explaining
    - Debugging: identify where reconstruction fails
    
    Parameters
    ----------
    F_T : torch.Tensor
        Total structure factor magnitude, shape [batch, H, W]
    F_A : torch.Tensor
        Anomalous structure factor magnitude, shape [batch, H, W]
    sin_phi, cos_phi : torch.Tensor
        Sin and cos of phase difference, shape [batch, H, W]
    f_prime : torch.Tensor
        f'(E) at each energy, shape [n_energies] or [batch, n_energies]
    f_double_prime : torch.Tensor
        f''(E) at each energy, shape [n_energies] or [batch, n_energies]
    f0 : torch.Tensor, optional
        Thomson factor, shape [batch, H, W]. Default: 1.0
        
    Returns
    -------
    I_reconstructed : torch.Tensor
        Reconstructed intensity, shape [batch, H, W, n_energies]
    """
    
    # Handle f₀
    if f0 is None:
        F_A_norm = F_A
    else:
        F_A_norm = F_A / (f0 + 1e-10)
    
    # Handle f'/f'' shape
    if f_prime.dim() == 1:
        fp = f_prime.view(1, 1, 1, -1)
        fs = f_double_prime.view(1, 1, 1, -1)
    else:
        batch_size = f_prime.shape[0]
        fp = f_prime.view(batch_size, 1, 1, -1)
        fs = f_double_prime.view(batch_size, 1, 1, -1)
    
    # Expand spatial tensors
    F_T = F_T.unsqueeze(-1)
    F_A_norm = F_A_norm.unsqueeze(-1)
    sin_phi = sin_phi.unsqueeze(-1)
    cos_phi = cos_phi.unsqueeze(-1)
    
    # MAD equation
    term1 = F_T ** 2
    term2 = (fp**2 + fs**2) * F_A_norm**2
    term3 = 2 * F_T * F_A_norm * (fp * cos_phi + fs * sin_phi)
    
    I_reconstructed = term1 + term2 + term3
    
    return I_reconstructed


# =============================================================================
# NUMPY VERSION (for validation script compatibility)
# =============================================================================

def reconstruct_intensity_numpy(
    F_T: np.ndarray,
    F_A: np.ndarray,
    sin_phi: np.ndarray,
    cos_phi: np.ndarray,
    f_prime: np.ndarray,
    f_double_prime: np.ndarray,
    f0: np.ndarray = None
) -> np.ndarray:
    """
    NumPy version of intensity reconstruction for validation.
    
    Parameters
    ----------
    F_T, F_A : np.ndarray
        Structure factor magnitudes, shape [H, W]
    sin_phi, cos_phi : np.ndarray
        Phase components, shape [H, W]
    f_prime, f_double_prime : np.ndarray
        Anomalous corrections, shape [n_energies]
    f0 : np.ndarray, optional
        Thomson factor, shape [H, W]. Default: 1.0
        
    Returns
    -------
    I_reconstructed : np.ndarray
        Reconstructed intensity, shape [H, W, n_energies]
    """
    
    n_energies = len(f_prime)
    H, W = F_T.shape
    
    # Handle f₀
    if f0 is None:
        F_A_norm = F_A
    else:
        F_A_norm = F_A / (f0 + 1e-10)
    
    # Initialize output
    I_reconstructed = np.zeros((H, W, n_energies))
    
    # Compute for each energy
    for e in range(n_energies):
        fp = f_prime[e]
        fs = f_double_prime[e]
        
        term1 = F_T ** 2
        term2 = (fp**2 + fs**2) * F_A_norm**2
        term3 = 2 * F_T * F_A_norm * (fp * cos_phi + fs * sin_phi)
        
        I_reconstructed[:, :, e] = term1 + term2 + term3
    
    return I_reconstructed


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == '__main__':
    print("MAD Loss Function Module")
    print("=" * 60)
    print()
    print("This module provides a physics-informed loss function for")
    print("training a CNN to predict MAD parameters from multi-energy")
    print("diffraction intensities.")
    print()
    print("Components:")
    print("  1. Intensity-weighted MSE (higher signal = more reliable)")
    print("  2. Unit circle constraint (sin² + cos² = 1)")
    print("  3. Intensity reconstruction (physics self-consistency)")
    print()
    print("Example usage:")
    print("-" * 60)
    print("""
    import torch
    from mad_loss import MADLoss
    
    # Create loss function
    criterion = MADLoss(
        lambda_unit=0.1,      # Weight for unit circle constraint
        lambda_recon=0.1,     # Weight for intensity reconstruction
        use_recon_loss=True   # Enable physics self-consistency check
    )
    
    # In training loop:
    for batch in dataloader:
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(batch['intensity'], batch['f_prime'], batch['f_double_prime'])
        
        # Compute loss
        loss, loss_dict = criterion(
            pred=pred,
            target=batch['labels'],
            intensities=batch['intensity'],
            f_prime=batch['f_prime'],
            f_double_prime=batch['f_double_prime']
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Log individual components
        print(f"MSE: {loss_dict['mse']:.4f}, "
              f"Unit: {loss_dict['unit']:.4f}, "
              f"Recon: {loss_dict['recon']:.4f}")
    """)
    print("-" * 60)
    
    # Quick test with random data
    print("\nQuick test with random data:")
    
    batch_size, H, W, n_energies = 4, 16, 16, 8
    
    # Create random tensors
    pred = torch.rand(batch_size, H, W, 4)
    # Apply activations like the model would
    pred[..., :2] = torch.nn.functional.softplus(pred[..., :2])  # |F_T|, |F_A| positive
    pred[..., 2:] = torch.tanh(pred[..., 2:])  # sin/cos in [-1, 1]
    
    target = torch.rand(batch_size, H, W, 4)
    target[..., :2] = torch.nn.functional.softplus(target[..., :2])
    target[..., 2:] = torch.tanh(target[..., 2:])
    
    intensities = torch.rand(batch_size, H, W, n_energies) * 1000 + 1
    
    # Load f'/f'' dynamically from data files (same as training/inference)
    from core_shell import ScatteringFactors
    import numpy as np
    
    ENERGIES = [8313, 8318, 8323, 8328, 8333, 8338, 8343, 8348]  # eV, Ni K-edge
    
    try:
        sf = ScatteringFactors(data_dir='.')
        f_prime_np = np.array([sf.get_f_prime('Ni', E) for E in ENERGIES])
        f_double_prime_np = np.array([sf.get_f_double_prime('Ni', E) for E in ENERGIES])
        print(f"  Loaded f'/f'' from Nickel.f1f2 for energies {ENERGIES[0]}-{ENERGIES[-1]} eV")
    except FileNotFoundError:
        # Fallback if .f1f2 files not in current directory
        print("  Warning: Nickel.f1f2 not found, using approximate values")
        f_prime_np = np.array([-5.8, -6.2, -6.7, -7.3, -7.9, -7.5, -6.8, -6.2])
        f_double_prime_np = np.array([0.5, 0.5, 0.5, 0.5, 3.9, 3.9, 3.8, 3.7])
    
    f_prime = torch.tensor(f_prime_np, dtype=torch.float32)
    f_double_prime = torch.tensor(f_double_prime_np, dtype=torch.float32)
    
    print(f"  f'  values: {f_prime.numpy().round(2).tolist()}")
    print(f"  f'' values: {f_double_prime.numpy().round(2).tolist()}")
    
    # Create loss function and compute
    criterion = MADLoss(lambda_unit=0.1, lambda_recon=0.1)
    loss, loss_dict = criterion(
        pred=pred,
        target=target,
        intensities=intensities,
        f_prime=f_prime,
        f_double_prime=f_double_prime
    )
    
    print(f"  Total loss: {loss.item():.4f}")
    print(f"  MSE loss: {loss_dict['mse']:.4f}")
    print(f"  Unit circle loss: {loss_dict['unit']:.4f}")
    print(f"  Reconstruction loss: {loss_dict['recon']:.4f}")
    print()
    print("Module ready for use!")
