# NanoMAD CNN Input Specification

## ⚠️ CRITICAL: Log-Scaled Magnitudes

**The model outputs |F_T| and |F_A| in LOG SCALE!**

At inference, you MUST convert back to linear scale:
```python
F_T_linear = np.expm1(F_T_predicted)  # exp(x) - 1
F_A_linear = np.expm1(F_A_predicted)
```

Or use the helper function:
```python
from mad_model import predict_and_convert
results = predict_and_convert(model, intensity, f_prime, f_double_prime)
# results['F_T'] and results['F_A'] are already in linear scale
```

---

## Model Input Requirements

### Intensity Data

| Property | Value |
|----------|-------|
| Shape | `(N, 16, 16, 8)` |
| N | Number of patches (any) |
| Spatial | 16 × 16 pixels per patch |
| Energies | 8 channels |
| Dtype | float32 |
| Values | Non-negative (intensity = |amplitude|²) |

### Energies Used in Training

```python
ENERGIES = [8313, 8318, 8323, 8328, 8333, 8338, 8343, 8348]  # eV
```

- 8 energies around Ni K-edge (8333 eV)
- 5 eV spacing
- 35 eV total span

### Scattering Factors

| Property | Shape | Description |
|----------|-------|-------------|
| f'(E) | `(8,)` | Real anomalous correction for Ni at each energy |
| f''(E) | `(8,)` | Imaginary anomalous correction for Ni at each energy |

**These must correspond to your actual measurement energies.**

---

## Model Output

| Channel | Meaning | Activation | Range | Scale |
|---------|---------|------------|-------|-------|
| 0 | \|F_T\| | Softplus | ≥ 0 | **LOG** (use expm1 to convert) |
| 1 | \|F_A\| | Softplus | ≥ 0 | **LOG** (use expm1 to convert) |
| 2 | sin(Δφ) | Tanh | [-1, 1] | Linear |
| 3 | cos(Δφ) | Tanh | [-1, 1] | Linear |

Output shape: `(N, 16, 16, 4)` (or `(N, 4, 16, 16)` in PyTorch channels-first)

**Recover physical values:**
```python
F_T = np.expm1(output[..., 0])     # Linear magnitude
F_A = np.expm1(output[..., 1])     # Linear magnitude  
delta_phi = np.arctan2(output[..., 2], output[..., 3])  # Phase in radians
```

---

## Other Training Parameters

| Parameter | Value |
|-----------|-------|
| Q_Bragg | 3.09 Å⁻¹ (Ni 111) |
| FFT grid | 256 × 256 |
| Output grid (before patching) | 128 × 128 |
| Patch size | 16 × 16 |
| Patches per 128×128 image | 64 (8 × 8 grid) |

---

## PyTorch Convention Note

The model internally uses **channels-first** format `(N, 8, 16, 16)`.

If you provide `(N, 16, 16, 8)`, transpose before feeding to model:
```python
intensity_chw = intensity_hwc.permute(0, 3, 1, 2)  # (N, 8, 16, 16)
```

---

## Files Needed for Inference

- `mad_model.py` — Model architecture
- `checkpoint_best.pt` — Trained weights (output from training)
