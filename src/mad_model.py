#!/usr/bin/env python3
"""
mad_model.py

CNN architecture for MAD (Multi-wavelength Anomalous Diffraction) parameter prediction.

This module implements a U-Net style network that takes multi-energy diffraction
intensity patches and predicts the physical parameters (|F_T|, |F_A|, sin(Δφ), cos(Δφ)).

Architecture Overview:
======================

The network has three main components:

1. SCATTERING FACTOR ENCODER
   - Takes f'(E) and f''(E) as input [batch, 16] (8 energies × 2)
   - Encodes to a feature vector [batch, 32]
   - This provides "global context" about the energy dependence

2. INTENSITY ENCODER (U-Net contracting path)
   - Takes intensity patches [batch, 8, 16, 16] (PyTorch convention: channels first)
   - Three encoding blocks: 8→32→64→128 channels
   - Spatial resolution: 16→8→4 via MaxPool
   - Skip connections saved for decoder

3. BOTTLENECK
   - Injects scattering factor features into the spatial representation
   - Broadcasts [batch, 32] → [batch, 32, 4, 4] and concatenates
   - Processes combined representation

4. INTENSITY DECODER (U-Net expanding path)
   - Three decoding blocks with skip connections
   - Spatial resolution: 4→8→16 via UpSample
   - Channels: 128→64→32

5. OUTPUT HEAD
   - 1×1 convolution to 4 channels
   - Softplus activation for |F_T|, |F_A| (ensures positive)
   - Tanh activation for sin(Δφ), cos(Δφ) (ensures [-1, 1])

Why U-Net?
==========
- Skip connections preserve spatial detail (important for diffraction patterns)
- Encoder captures context, decoder reconstructs at full resolution
- Well-suited for dense prediction tasks (output same size as input)

Why inject f'/f'' at bottleneck?
================================
- f'/f'' are global values (same for all pixels)
- Bottleneck is where the network has the most abstract representation
- Injection here doesn't interfere with spatial feature extraction
- The decoder uses this context to make pixel-wise predictions

Author: Claude (Anthropic) + Thomas
Date: December 2024
Based on: NanoMAD expert recommendations (NanoMAD_CNN_Design.md)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# =============================================================================
# BUILDING BLOCKS
# =============================================================================

class ConvBlock(nn.Module):
    """
    Double convolution block used in U-Net encoder and decoder.
    
    Structure:
        Conv2D → BatchNorm → LeakyReLU → Conv2D → BatchNorm → LeakyReLU
    
    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class EncoderBlock(nn.Module):
    """
    U-Net encoder block: ConvBlock followed by MaxPool for downsampling.
    
    Returns both the conv output (for skip connection) and the pooled output.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        skip : torch.Tensor
            Output before pooling (for skip connection)
        pooled : torch.Tensor
            Output after pooling (for next encoder level)
        """
        skip = self.conv(x)
        pooled = self.pool(skip)
        return skip, pooled


class DecoderBlock(nn.Module):
    """
    U-Net decoder block: Upsample → Concatenate skip → ConvBlock
    """
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        
        # Upsample doubles spatial dimensions
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # After concatenation, we have in_channels + skip_channels
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input from previous decoder level
        skip : torch.Tensor
            Skip connection from corresponding encoder level
        """
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)  # Concatenate along channel dimension
        return self.conv(x)


class ScatteringFactorEncoder(nn.Module):
    """
    MLP encoder for anomalous scattering factors f'(E) and f''(E).
    
    Takes the energy-dependent scattering factors and encodes them into
    a feature vector that provides global context for the decoder.
    
    Input: [batch, 16] (f' and f'' concatenated, 8 energies each)
    Output: [batch, 32] (encoded features)
    """
    
    def __init__(self, n_energies: int = 8, hidden_dim: int = 32, output_dim: int = 32):
        super().__init__()
        
        input_dim = n_energies * 2  # f' and f'' concatenated
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, f_prime: torch.Tensor, f_double_prime: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        f_prime : torch.Tensor
            Real anomalous correction, shape [batch, n_energies]
        f_double_prime : torch.Tensor
            Imaginary anomalous correction, shape [batch, n_energies]
            
        Returns
        -------
        features : torch.Tensor
            Encoded features, shape [batch, output_dim]
        """
        # Concatenate f' and f''
        x = torch.cat([f_prime, f_double_prime], dim=1)  # [batch, 16]
        return self.mlp(x)


# =============================================================================
# MAIN MODEL
# =============================================================================

class MADNet(nn.Module):
    """
    U-Net for MAD parameter prediction with scattering factor injection.
    
    This network predicts physical parameters from multi-energy diffraction data:
    
    Inputs:
        - intensity: [batch, n_energies, H, W] — diffraction intensities
        - f_prime: [batch, n_energies] — real anomalous correction
        - f_double_prime: [batch, n_energies] — imaginary anomalous correction
    
    Outputs:
        - [batch, 4, H, W] — predicted parameters:
            - Channel 0: |F_T| (total structure factor magnitude)
            - Channel 1: |F_A| (anomalous structure factor magnitude)
            - Channel 2: sin(Δφ) (sine of phase difference)
            - Channel 3: cos(Δφ) (cosine of phase difference)
    
    Parameters
    ----------
    n_energies : int
        Number of energy channels in input (default: 8)
    base_channels : int
        Number of channels in first encoder level (default: 32)
        Subsequent levels: base_channels → 2× → 4×
    sf_feature_dim : int
        Dimension of scattering factor encoded features (default: 32)
    """
    
    def __init__(
        self,
        n_energies: int = 8,
        base_channels: int = 32,
        sf_feature_dim: int = 32
    ):
        super().__init__()
        
        self.n_energies = n_energies
        self.base_channels = base_channels
        self.sf_feature_dim = sf_feature_dim
        
        # Channel progression: n_energies → 32 → 64 → 128
        c1 = base_channels       # 32
        c2 = base_channels * 2   # 64
        c3 = base_channels * 4   # 128
        
        # ---------------------------------------------------------------------
        # Scattering factor encoder
        # ---------------------------------------------------------------------
        self.sf_encoder = ScatteringFactorEncoder(
            n_energies=n_energies,
            hidden_dim=32,
            output_dim=sf_feature_dim
        )
        
        # ---------------------------------------------------------------------
        # Intensity encoder (contracting path)
        # ---------------------------------------------------------------------
        # Level 1: [B, 8, 16, 16] → [B, 32, 16, 16] → pool → [B, 32, 8, 8]
        self.enc1 = EncoderBlock(n_energies, c1)
        
        # Level 2: [B, 32, 8, 8] → [B, 64, 8, 8] → pool → [B, 64, 4, 4]
        self.enc2 = EncoderBlock(c1, c2)
        
        # Level 3 (bottleneck conv): [B, 64, 4, 4] → [B, 128, 4, 4]
        self.bottleneck_conv = ConvBlock(c2, c3)
        
        # ---------------------------------------------------------------------
        # Bottleneck (inject scattering factors)
        # ---------------------------------------------------------------------
        # After injection: [B, 128 + 32, 4, 4] = [B, 160, 4, 4]
        # Process back to [B, 128, 4, 4]
        self.bottleneck_fusion = ConvBlock(c3 + sf_feature_dim, c3)
        
        # ---------------------------------------------------------------------
        # Intensity decoder (expanding path)
        # ---------------------------------------------------------------------
        # Level 2: [B, 128, 4, 4] → upsample → [B, 128, 8, 8]
        #          concat skip2 → [B, 128+64, 8, 8] → [B, 64, 8, 8]
        self.dec2 = DecoderBlock(c3, c2, c2)
        
        # Level 1: [B, 64, 8, 8] → upsample → [B, 64, 16, 16]
        #          concat skip1 → [B, 64+32, 16, 16] → [B, 32, 16, 16]
        self.dec1 = DecoderBlock(c2, c1, c1)
        
        # ---------------------------------------------------------------------
        # Output head
        # ---------------------------------------------------------------------
        # 1×1 convolution to 4 output channels
        self.output_conv = nn.Conv2d(c1, 4, kernel_size=1)
    
    def forward(
        self,
        intensity: torch.Tensor,
        f_prime: torch.Tensor,
        f_double_prime: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        intensity : torch.Tensor
            Multi-energy diffraction intensities, shape [batch, n_energies, H, W]
        f_prime : torch.Tensor
            Real anomalous correction f'(E), shape [batch, n_energies]
        f_double_prime : torch.Tensor
            Imaginary anomalous correction f''(E), shape [batch, n_energies]
            
        Returns
        -------
        output : torch.Tensor
            Predicted parameters, shape [batch, 4, H, W]
            Channels: |F_T|, |F_A|, sin(Δφ), cos(Δφ)
        """
        batch_size = intensity.shape[0]
        
        # -----------------------------------------------------------------
        # 1. Encode scattering factors
        # -----------------------------------------------------------------
        sf_features = self.sf_encoder(f_prime, f_double_prime)  # [B, 32]
        
        # -----------------------------------------------------------------
        # 2. Intensity encoder
        # -----------------------------------------------------------------
        # Level 1
        skip1, x = self.enc1(intensity)  # skip1: [B, 32, 16, 16], x: [B, 32, 8, 8]
        
        # Level 2
        skip2, x = self.enc2(x)  # skip2: [B, 64, 8, 8], x: [B, 64, 4, 4]
        
        # Bottleneck conv
        x = self.bottleneck_conv(x)  # [B, 128, 4, 4]
        
        # -----------------------------------------------------------------
        # 3. Inject scattering factor features at bottleneck
        # -----------------------------------------------------------------
        # Expand sf_features from [B, 32] to [B, 32, 4, 4]
        # by broadcasting to match spatial dimensions
        H_bottleneck, W_bottleneck = x.shape[2], x.shape[3]
        sf_expanded = sf_features.unsqueeze(-1).unsqueeze(-1)  # [B, 32, 1, 1]
        sf_expanded = sf_expanded.expand(-1, -1, H_bottleneck, W_bottleneck)  # [B, 32, 4, 4]
        
        # Concatenate and process
        x = torch.cat([x, sf_expanded], dim=1)  # [B, 160, 4, 4]
        x = self.bottleneck_fusion(x)  # [B, 128, 4, 4]
        
        # -----------------------------------------------------------------
        # 4. Intensity decoder with skip connections
        # -----------------------------------------------------------------
        x = self.dec2(x, skip2)  # [B, 64, 8, 8]
        x = self.dec1(x, skip1)  # [B, 32, 16, 16]
        
        # -----------------------------------------------------------------
        # 5. Output head with activations
        # -----------------------------------------------------------------
        x = self.output_conv(x)  # [B, 4, 16, 16]
        
        # Apply channel-specific activations:
        # - Channels 0, 1 (|F_T|, |F_A|): Softplus → always positive
        # - Channels 2, 3 (sin, cos): Tanh → range [-1, 1]
        output = torch.empty_like(x)
        output[:, 0:2, :, :] = F.softplus(x[:, 0:2, :, :])  # |F_T|, |F_A|
        output[:, 2:4, :, :] = torch.tanh(x[:, 2:4, :, :])  # sin(Δφ), cos(Δφ)
        
        return output
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_model(
    n_energies: int = 8,
    base_channels: int = 32,
    device: str = 'cuda'
) -> MADNet:
    """
    Create and initialize the MADNet model.
    
    Parameters
    ----------
    n_energies : int
        Number of energy channels
    base_channels : int
        Base channel count (32 → ~200K params, 64 → ~800K params)
    device : str
        Device to place model on ('cuda' or 'cpu')
        
    Returns
    -------
    model : MADNet
        Initialized model on specified device
    """
    model = MADNet(n_energies=n_energies, base_channels=base_channels)
    model = model.to(device)
    
    print(f"MADNet created:")
    print(f"  Input: [{n_energies}, 16, 16] intensity + [{n_energies}] f' + [{n_energies}] f''")
    print(f"  Output: [4, 16, 16] (|F_T|, |F_A|, sin(Δφ), cos(Δφ))")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Device: {device}")
    
    return model


# =============================================================================
# INFERENCE HELPER
# =============================================================================

def predict_and_convert(
    model: MADNet,
    intensity: torch.Tensor,
    f_prime: torch.Tensor,
    f_double_prime: torch.Tensor,
    log_scale_magnitudes: bool = True
) -> dict:
    """
    Run inference and convert predictions to physical units.
    
    =========================================================================
    IMPORTANT: LOG-SCALE MAGNITUDE CONVERSION
    =========================================================================
    
    The model is trained with log-scaled magnitudes:
        Training target: log1p(|F|) = log(1 + |F|)
        
    This function converts predictions BACK to linear scale:
        Linear magnitude: expm1(pred) = exp(pred) - 1
    
    If you DON'T use this function, remember to convert manually:
        F_T_linear = np.expm1(F_T_log)
        F_A_linear = np.expm1(F_A_log)
    
    =========================================================================
    
    Parameters
    ----------
    model : MADNet
        Trained model
    intensity : torch.Tensor
        Input intensities, shape [batch, 8, 16, 16] (channels-first)
    f_prime : torch.Tensor
        f'(E), shape [batch, 8] or [8]
    f_double_prime : torch.Tensor
        f''(E), shape [batch, 8] or [8]
    log_scale_magnitudes : bool
        If True (default), convert magnitudes from log to linear scale.
        Set to False only if model was trained WITHOUT log scaling.
        
    Returns
    -------
    dict with:
        'F_T' : np.ndarray, shape [batch, 16, 16]
            Total structure factor magnitude (LINEAR scale)
        'F_A' : np.ndarray, shape [batch, 16, 16]
            Anomalous structure factor magnitude (LINEAR scale)
        'sin_delta_phi' : np.ndarray, shape [batch, 16, 16]
            sin(Δφ)
        'cos_delta_phi' : np.ndarray, shape [batch, 16, 16]
            cos(Δφ)
        'delta_phi' : np.ndarray, shape [batch, 16, 16]
            Phase difference Δφ in radians, computed as arctan2(sin, cos)
    """
    import numpy as np
    
    model.eval()
    device = next(model.parameters()).device
    
    # Move to device
    intensity = intensity.to(device)
    if f_prime.dim() == 1:
        f_prime = f_prime.unsqueeze(0).expand(intensity.shape[0], -1)
    if f_double_prime.dim() == 1:
        f_double_prime = f_double_prime.unsqueeze(0).expand(intensity.shape[0], -1)
    f_prime = f_prime.to(device)
    f_double_prime = f_double_prime.to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(intensity, f_prime, f_double_prime)
    
    # Convert to numpy (channels-last)
    output = output.permute(0, 2, 3, 1).cpu().numpy()  # [batch, H, W, 4]
    
    # Extract channels
    F_T = output[..., 0]
    F_A = output[..., 1]
    sin_phi = output[..., 2]
    cos_phi = output[..., 3]
    
    # =========================================================================
    # CONVERT MAGNITUDES FROM LOG TO LINEAR SCALE
    # =========================================================================
    if log_scale_magnitudes:
        F_T = np.expm1(F_T)  # exp(x) - 1, inverse of log1p
        F_A = np.expm1(F_A)
    
    # Compute phase difference
    delta_phi = np.arctan2(sin_phi, cos_phi)
    
    return {
        'F_T': F_T,
        'F_A': F_A,
        'sin_delta_phi': sin_phi,
        'cos_delta_phi': cos_phi,
        'delta_phi': delta_phi,
    }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("MADNet Architecture Test")
    print("=" * 70)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Create model
    model = create_model(n_energies=8, base_channels=32, device=device)
    
    # Create dummy input
    batch_size = 4
    n_energies = 8
    H, W = 16, 16
    
    intensity = torch.randn(batch_size, n_energies, H, W).to(device)
    f_prime = torch.randn(batch_size, n_energies).to(device)
    f_double_prime = torch.randn(batch_size, n_energies).to(device)
    
    print(f"\nInput shapes:")
    print(f"  intensity: {intensity.shape}")
    print(f"  f_prime: {f_prime.shape}")
    print(f"  f_double_prime: {f_double_prime.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(intensity, f_prime, f_double_prime)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"  Expected: [4, 4, 16, 16]")
    
    # Check output ranges
    print(f"\nOutput statistics:")
    print(f"  |F_T| (ch 0): min={output[:, 0].min():.3f}, max={output[:, 0].max():.3f} (should be > 0)")
    print(f"  |F_A| (ch 1): min={output[:, 1].min():.3f}, max={output[:, 1].max():.3f} (should be > 0)")
    print(f"  sin(Δφ) (ch 2): min={output[:, 2].min():.3f}, max={output[:, 2].max():.3f} (should be in [-1, 1])")
    print(f"  cos(Δφ) (ch 3): min={output[:, 3].min():.3f}, max={output[:, 3].max():.3f} (should be in [-1, 1])")
    
    # Verify activations work correctly
    assert (output[:, 0:2] >= 0).all(), "|F_T| and |F_A| should be non-negative"
    assert (output[:, 2:4] >= -1).all() and (output[:, 2:4] <= 1).all(), "sin/cos should be in [-1, 1]"
    
    print("\n✓ All tests passed!")
    print(f"\nModel ready for training with {model.count_parameters():,} parameters")
