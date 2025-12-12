# NanoMAD-ML Development Guide

This document provides context for developers (human or AI) working on this codebase.

**Last Updated**: December 2024

---

## Project Summary

**NanoMAD ML** is a machine learning replacement for traditional iterative MAD (Multi-wavelength Anomalous Diffraction) fitting in BCDI (Bragg Coherent Diffraction Imaging) analysis of core-shell nanoparticles.

**The Problem**: Traditional NanoMAD fitting is slow (hours per 3D dataset) because it fits parameters pixel-by-pixel.

**The Solution**: A CNN that directly predicts MAD parameters from multi-energy diffraction patches in seconds.

---

## The Physics (Essential Background)

### Core-Shell Nanoparticles

We study bimetallic nanoparticles with:
- **Core**: Fe-rich (e.g., Ni₃Fe = 75% Ni, 25% Fe)
- **Shell**: Pure Ni

The goal is to separate the contributions of each element using X-ray diffraction.

### The MAD Equation

```
I(Q,E) = |F_T|² + (f'² + f''²)|F_A/f₀|² + 2|F_T||F_A|/f₀·[f'cos(Δφ) + f''sin(Δφ)]
```

Where:
- `|F_T|` = Total structure factor magnitude (all atoms)
- `|F_A|` = Anomalous structure factor magnitude (Ni only, varies with energy)
- `Δφ = φ_T - φ_A` = Phase difference
- `f'(E), f''(E)` = Anomalous scattering corrections (change dramatically near Ni K-edge ~8333 eV)
- `f₀(Q)` = Thomson scattering factor (Q-dependent, energy-independent)

### What the CNN Predicts

From 8-energy intensity patches, the CNN outputs:
1. `|F_T|` - Total structure factor magnitude
2. `|F_A|` - Anomalous structure factor magnitude
3. `sin(Δφ)` - Sine of phase difference
4. `cos(Δφ)` - Cosine of phase difference

From these, we derive `|F_N|` (non-anomalous = Fe contribution):
```python
F_N² = F_T² + F_A² - 2·F_T·F_A·cos(Δφ)
```

---

## Repository Structure

```
NanoMAD-ML/
├── src/                    # Source code
│   ├── core_shell.py       # Physics engine (~4800 lines)
│   ├── mad_model.py        # U-Net CNN architecture
│   ├── mad_loss.py         # Physics-informed loss function
│   ├── train.py            # Training pipeline
│   ├── inference.py        # 2D/3D inference with post-processing
│   ├── evaluate.py         # Evaluation metrics
│   ├── generate_data.py    # Synthetic data generation
│   ├── augmentation.py     # Physics-preserving augmentations
│   ├── visualization.py    # Plotting utilities
│   └── validate.py         # MAD equation validation
│
├── data/                   # Reference data
│   ├── Nickel.f1f2         # Ni anomalous scattering factors
│   ├── Iron.f1f2           # Fe anomalous scattering factors
│   └── thomson_factors.py  # Thomson (f₀) coefficients
│
├── docs/                   # Documentation
├── notebooks/              # Jupyter notebooks
├── models/                 # Pre-trained checkpoints
└── examples/               # Example shell scripts
```

---

## Quick Reference: Key Functions

### Particle Creation

```python
from src.core_shell import create_particle_with_shape, DEFAULT_PIXEL_SIZE

particle, metadata = create_particle_with_shape(
    grid_size=128,
    shape_type='hexagon',  # circle, hexagon, polygon, ellipse, polygon_centrosymmetric
    composition_mode='sharp',  # sharp, radial_gradient, linear_gradient, janus, uniform
    core_fraction=0.5,
    core_composition={'Ni': 0.75, 'Fe': 0.25},
    shell_composition={'Ni': 1.0, 'Fe': 0.0},
    verbose=False
)
# particle: shape (2, 128, 128) - [Ni, Fe] density maps
# metadata: dict with 'outer_mask', 'core_mask', etc.
```

### Ground Truth Labels

```python
from src.core_shell import compute_ground_truth_labels

labels = compute_ground_truth_labels(
    particle=particle_with_displacement,
    pixel_size=DEFAULT_PIXEL_SIZE,
    output_size=128
)
# Returns dict: F_T_mag, F_A_mag, sin_delta_phi, cos_delta_phi, delta_phi, F_N_mag
```

### Diffraction Computation

```python
from src.core_shell import compute_diffraction_oversampled_cropped, ScatteringFactors

sf = ScatteringFactors(data_dir='data')
diffraction_dict = compute_diffraction_oversampled_cropped(
    particle=particle_with_displacement,
    energies=[8313, 8318, 8323, 8328, 8333, 8338, 8343, 8348],
    pixel_size=DEFAULT_PIXEL_SIZE,
    scattering_factors=sf,
    output_size=128
)
# Returns dict: {energy: complex_diffraction_array}
```

---

## Data Format Conventions

### Particle Array
Shape `(2, H, W)`:
- `[0, :, :]` = Ni density
- `[1, :, :]` = Fe density

### Intensity Patches
Shape `(N, 16, 16, 8)` - last axis is energy channels

### Labels
Shape `(N, 16, 16, 4)`:
- Channel 0: log1p(|F_T|)
- Channel 1: log1p(|F_A|)
- Channel 2: sin(Δφ)
- Channel 3: cos(Δφ)

### Standard Energies
8 energies around Ni K-edge (8333 eV):
```python
ENERGIES = [8313, 8318, 8323, 8328, 8333, 8338, 8343, 8348]  # eV
```

---

## Training Commands

### Recommended

```bash
python src/train.py \
    --data-dir /path/to/synthetic_data \
    --weight-scheme log \
    --log-transform-intensity \
    --lambda-fa 1.0 \
    --epochs 200 \
    --output-dir training_output
```

### Inference (Must Match Training!)

```bash
python src/inference.py \
    -c models/checkpoint_v5.pt \
    --intensity /path/to/intensity.npy \
    --energies /path/to/energies.npy \
    --log-transform-intensity \
    -o output_dir/
```

---

## Common Issues & Solutions

### "ModuleNotFoundError: No module named 'torch'"

The demo notebook (`notebooks/01_demonstration.ipynb`) is designed to work without PyTorch. It only imports from `core_shell.py` which uses numpy. PyTorch is only needed for actual training/inference.

### "TypeError: create_particle_with_shape() got unexpected keyword argument"

The API uses `core_composition` and `shell_composition` dicts, NOT `ni_fraction_core`:

```python
# CORRECT:
create_particle_with_shape(
    core_composition={'Ni': 0.75, 'Fe': 0.25},
    shell_composition={'Ni': 1.0, 'Fe': 0.0},
    ...
)

# WRONG (old API):
create_particle_with_shape(ni_fraction_core=0.75, ...)
```

### High R-factor in Evaluation

The `evaluate.py` script uses a simplified MAD equation that doesn't match the full physics in `core_shell.py`. This causes a **scale mismatch** leading to high R-factors (~5000). This is expected - focus on **correlation** (>0.9 is good) instead.

---

## File Interdependencies

```
core_shell.py ─────────────────┐
    │                          │
    │ imports from             │ imports from
    ▼                          ▼
data/thomson_factors.py    data/Nickel.f1f2, Iron.f1f2

mad_model.py ──────────────────┐
    │                          │
    │ imported by              │ imported by
    ▼                          ▼
train.py                    inference.py

mad_loss.py ──────────────────┐
    │                          │
    │ imported by              │ may import
    ▼                          ▼
train.py                    core_shell.py
```

---

## Current Status (December 2024)

### Working Features
- Physics engine (core_shell.py) - comprehensive particle simulation
- CNN architecture and training pipeline
- 2D and 3D inference with Hann blending
- Overlapping patch training for smoother results
- Configurable loss weighting schemes
- F_A-specific loss term (`--lambda-fa`)
- Post-processing pipeline (unit circle, Z-smoothing, contrast enhancement)
- Evaluation script with haze diagnostic

### Known Limitations
- CNN is 2D; 3D is processed slice-by-slice (no z-axis correlation)
- Haze artifact (can use `--enhance-contrast` post-processing)
- No uncertainty quantification yet

---

## Contact

This project is part of Thomas Sarrazin's PhD research at ESRF (ID01 beamline).
AI assistance provided by Claude (Anthropic).
