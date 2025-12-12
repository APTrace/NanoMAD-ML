# NanoMAD-ML

**CNN-based replacement for traditional NanoMAD MAD fitting in BCDI analysis of core-shell nanoparticles.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This pipeline uses a U-Net CNN to extract MAD (Multi-wavelength Anomalous Diffraction) parameters from multi-energy X-ray diffraction data of core-shell nanoparticles.

| Traditional NanoMAD | NanoMAD ML |
|---------------------|------------|
| Iterative fitting per pixel | Single forward pass |
| Hours per 3D dataset | Seconds |
| No spatial context | Uses neighboring pixels |
| Sensitive to initialization | No initialization needed |

**Input**: 16x16x8 intensity patches (Q-space at 8 energies) + f'/f'' scattering factors
**Output**: 16x16x4 physical parameters (|F_T|, |F_A|, sin(delta_phi), cos(delta_phi))

From these outputs, we derive |F_N| (non-anomalous structure factor), which isolates the **Fe core contribution** from the Ni shell.

---

## Installation

```bash
git clone https://github.com/yourusername/NanoMAD-ML.git
cd NanoMAD-ML
pip install -r requirements.txt
```

---

## Quick Start

```bash
# 1. Generate training data
python src/generate_data.py -n 500 -o /path/to/data --sf-dir data --overlap --stride 8

# 2. Train the model (with F_A fix)
python src/train.py --data-dir /path/to/data --output-dir training_output \
    --weight-scheme log --log-transform-intensity --lambda-fa 1.0 --epochs 200

# 3. Run 3D inference
python src/inference.py -c models/checkpoint_v5.pt \
    --intensity data_3d.npy --energies energies.npy \
    --log-transform-intensity -o output/

# 4. Run 2D inference (single particle)
python src/inference.py --mode 2d -c models/checkpoint_v5.pt \
    --test-file particle.npz --log-transform-intensity -o output/
```

**New to this project?** Start with `notebooks/01_demonstration.ipynb` - it runs without PyTorch and explains all the physics.

---

## Repository Structure

```
NanoMAD-ML/
├── src/                         # Source code
│   ├── core_shell.py            # Physics engine - particles, diffraction
│   ├── mad_model.py             # U-Net CNN architecture
│   ├── mad_loss.py              # Physics-informed loss function
│   ├── train.py                 # Training pipeline
│   ├── inference.py             # 2D/3D inference with post-processing
│   ├── evaluate.py              # Evaluation metrics
│   ├── generate_data.py         # Synthetic data generation
│   ├── augmentation.py          # Physics-preserving augmentations
│   ├── visualization.py         # Plotting utilities
│   └── validate.py              # MAD equation validation
│
├── data/                        # Reference data
│   ├── Nickel.f1f2              # Ni anomalous scattering factors
│   ├── Iron.f1f2                # Fe anomalous scattering factors
│   └── thomson_factors.py       # Thomson scattering factors
│
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md          # CNN design rationale
│   ├── PHYSICS.md               # Physics background
│   ├── INPUT_FORMAT.md          # Data format specification
│   ├── TECHNICAL_NOTES.md       # Artifact analysis & roadmap
│   └── DEVELOPMENT.md           # Developer guide
│
├── notebooks/                   # Jupyter notebooks
│   ├── 01_demonstration.ipynb   # START HERE - physics explanation
│   ├── 02_workflow.ipynb        # Full pipeline demo
│   ├── 03_evaluation.ipynb      # Interactive evaluation
│   ├── 04_data_generation.ipynb # Training data creation
│   ├── 05_tests.ipynb           # Validation tests
│   ├── 06_real_data_example.ipynb
│   └── 07_phase_retrieval.ipynb
│
├── models/                      # Pre-trained models
│   └── checkpoint_v5.pt         # Pre-trained checkpoint
│
└── examples/                    # Example shell scripts
    ├── train_example.sh
    └── inference_example.sh
```

---

## The Physics

### The MAD Equation

```
I(Q,E) = |F_T|^2 + (f'^2 + f''^2)|F_A/f0|^2 + 2|F_T||F_A|/f0 [f' cos(dphi) + f'' sin(dphi)]
```

Where:
- `F_T` = Total structure factor (all atoms)
- `F_A` = Anomalous structure factor (Ni at K-edge)
- `dphi = phi_T - phi_A` = Phase difference
- `f'(E), f''(E)` = Anomalous corrections (energy-dependent)

### Derived Quantity: F_N (Fe Core)

```python
F_N^2 = F_T^2 + F_A^2 - 2*F_T*F_A*cos(dphi)
```

This isolates the non-anomalous (Fe) contribution from the total scattering.

---

## Usage Examples

### Python API

```python
from src.core_shell import create_particle_with_shape, ScatteringFactors

# Create a particle
particle, metadata = create_particle_with_shape(
    grid_size=128,
    shape_type='hexagon',
    composition_mode='sharp',
    core_fraction=0.5,
    core_composition={'Ni': 0.75, 'Fe': 0.25},
    shell_composition={'Ni': 1.0, 'Fe': 0.0},
)

# Compute scattering factors
sf = ScatteringFactors(data_dir='data')
```

### Training

```bash
python src/train.py \
    --data-dir /path/to/synthetic_data \
    --weight-scheme log \
    --log-transform-intensity \
    --lambda-fa 1.0 \
    --epochs 200 \
    --output-dir training_output
```

### Inference

```bash
# 3D mode (default)
python src/inference.py -c models/checkpoint_v5.pt \
    --intensity data_3d.npy --energies energies.npy \
    --log-transform-intensity -o output/

# With contrast enhancement (experimental)
python src/inference.py -c models/checkpoint_v5.pt \
    --intensity data_3d.npy --energies energies.npy \
    --log-transform-intensity --enhance-contrast -o output/
```

### Evaluation

```bash
python src/evaluate.py \
    --predictions output/ \
    --intensity test_intensity.npy \
    --energies test_energies.npy \
    -o evaluation_results/
```

---

## Post-Processing (Default ON)

The inference script automatically applies:

1. **Unit circle enforcement**: sin^2 + cos^2 = 1
2. **Z-axis smoothing** (sigma=1.0): Reduces slice discontinuities
3. **Stride=4 blending**: 75% patch overlap for smooth XY transitions

Disable with: `--no-unit-circle --no-smooth-z --stride 8`

---

## Key Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Correlation | > 0.9 | Spatial pattern matching |
| F_A hole fraction | < 2% | Signal continuity |
| Haze score | < 0.3 | Fringe contrast |

---

## Dependencies

```
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.4.0
torch>=1.9.0
```

---

## License

MIT License - see [LICENSE](LICENSE)

---

## Authors

- **Thomas Sarrazin** - PhD researcher, ESRF ID01
- **Claude (Anthropic)** - AI pair programmer

December 2024
