# NanoMAD-ML Technical Notes

This document contains detailed technical analysis, artifact mitigation strategies, and development roadmap for the NanoMAD-ML pipeline.

**Last Updated**: December 2024 (v0.1.0)

---

## Table of Contents

1. [Blocky Artifacts Analysis](#1-blocky-artifacts-analysis)
2. [Pipeline Improvements Roadmap](#2-pipeline-improvements-roadmap)
3. [F_A Signal Gaps Fix](#3-fa-signal-gaps-fix)
4. [Post-Processing Pipeline](#4-post-processing-pipeline)
5. [Evaluation Metrics](#5-evaluation-metrics)
6. [Known Limitations](#6-known-limitations)

---

## 1. Blocky Artifacts Analysis

### Problem Summary

The CNN inference produces blocky artifacts that manifest differently depending on viewing orientation:

| Slice Orientation | Artifact Pattern | Root Cause |
|-------------------|------------------|------------|
| **XY (z=const)** | 16x16 grid pattern | Patch boundaries during inference |
| **XZ (y=const)** | Horizontal banding/stripes | Slice-by-slice processing (no z-correlation) |
| **YZ (x=const)** | Vertical banding/stripes | Slice-by-slice processing (no z-correlation) |

### Two Distinct Artifact Sources

```
ARTIFACT SOURCES
================
1. XY PLANE ARTIFACTS (within each 2D slice)
   - Cause: 16x16 patch boundaries
   - Visible as: Grid pattern in XY view
   - Solution: Smaller stride (4), Hann window blending

2. Z-AXIS ARTIFACTS (across slices)
   - Cause: No correlation between adjacent Z slices
   - Visible as: Stripes in XZ and YZ views
   - Solution: Z-smoothing post-processing, 2.5D/3D architecture (future)
```

### Historical Issues (Fixed in v2)

| Issue | Solution | Status |
|-------|----------|--------|
| Loss ignores low-intensity pixels | `--weight-scheme log` | FIXED |
| Huge input dynamic range | `--log-transform-intensity` | FIXED |
| Outer regions not learned | Above fixes enabled learning | FIXED |

### Solutions Implemented

**XY Artifacts:**
- Overlapping patches with Hann window blending (default)
- Stride=4 (75% overlap) for smooth transitions

**Z-Axis Artifacts:**
- Gaussian smoothing along Z-axis (`--smooth-z 1.0`, default ON)
- Smooths sin/cos separately to avoid phase wrapping issues

---

## 2. Pipeline Improvements Roadmap

### Completed (v0.1)

| Feature | Status | Notes |
|---------|--------|-------|
| Log weighting loss | DONE | `--weight-scheme log` |
| Log-transform inputs | DONE | `--log-transform-intensity` |
| F_A-specific loss | DONE | `--lambda-fa 1.0` |
| Unit circle enforcement | DONE | Default ON |
| Z-axis smoothing | DONE | Default ON, sigma=1.0 |
| Local contrast enhancement | DONE | `--enhance-contrast` (experimental) |
| Evaluation script | DONE | `evaluate.py` with haze diagnostic |
| F_A hole detection | DONE | Added to evaluation |

### Recommended Training Command (v5)

```bash
python src/train.py \
    --data-dir /path/to/synthetic_data \
    --weight-scheme log \
    --log-transform-intensity \
    --lambda-fa 1.0 \
    --epochs 200 \
    --output-dir training_output
```

### Future Improvements (No Retraining)

1. **Test-Time Augmentation (TTA)**: Average 4 rotations for smoother output
2. **Tune Z-smoothing sigma**: Find optimal value for data quality vs artifact reduction
3. **Benchmark against traditional NanoMAD**: Direct comparison on same data

### Future Improvements (Requires Retraining)

1. **Sparsity penalty**: L1 regularization to encourage near-zero predictions
2. **SSIM loss addition**: May improve spatial coherence
3. **Larger patches (32x32)**: Fewer boundaries = fewer artifacts
4. **2.5D approach**: Input adjacent slices for Z-context
5. **3D native architecture**: Major rewrite, eliminates all boundary artifacts

---

## 3. F_A Signal Gaps Fix

### Problem Identified

|F_A| (anomalous structure factor / Ni contribution) showed holes/gaps in the diffraction pattern where signal should exist. These gaps propagate to F_N since:

```
F_N² = F_T² + F_A² - 2·F_T·F_A·cos(Δφ)
```

### Root Cause

1. Loss function intensity weighting causes network to ignore low-intensity regions
2. No dedicated F_A loss term - F_T dominates in bright regions
3. Network conflates "naturally small F_A" with "predict zero"

### Solution: Extra F_A Loss Term (A1 Approach)

Added `--lambda-fa` parameter that adds uniform-weighted F_A loss:

```python
# Extra F_A term (uniform weighting - all pixels contribute equally)
fa_mse = mse[..., 1].mean()  # Channel 1 = |F_A|
total_loss = loss_mse + lambda_fa * fa_mse + lambda_unit * loss_unit
```

### Results (lambda_fa=1.0)

| Metric | Before (v3/v4) | After (v5 FA fix) | Change |
|--------|----------------|-------------------|--------|
| F_A hole fraction | ~5-10% (est.) | **0.61%** | EXCELLENT |
| Correlation | 0.903 | 0.903 | Maintained |
| Haze score | 0.421 | 0.470 | +12% (trade-off) |
| Local fringe contrast | 0.579 | 0.530 | -8% |

**Key findings:**
- F_A holes effectively eliminated - 0.61% hole fraction (target <2%)
- Correlation preserved at 0.90
- Haze increased slightly (can be post-processed with `--enhance-contrast`)

---

## 4. Post-Processing Pipeline

### Default Post-Processing (run_inference_3d.py)

| Step | Default | Flag to Disable |
|------|---------|-----------------|
| Unit circle projection | ON | `--no-unit-circle` |
| Z-axis Gaussian smoothing (σ=1.0) | ON | `--no-smooth-z` |
| Stride=4 blending (75% overlap) | ON | `--stride 8` |

### Local Contrast Enhancement (Experimental)

To reduce haze artifact (shallow fringe valleys):

```bash
python src/inference.py -c models/checkpoint_v5.pt \
    --intensity data_3d.npy --energies energies.npy \
    --log-transform-intensity \
    --enhance-contrast --contrast-gamma 2.0 \
    -o output_enhanced/
```

**How it works:**
- Computes local max in 7x7x7 sliding window
- Applies gamma correction: `enhanced = (value/local_max)^gamma * local_max`
- Gamma > 1 pushes valleys deeper while preserving peaks

**Results (gamma=2.0):**
- Haze score: 0.42 → 0.30 (improved)
- Deep valley fraction: 12% → 34% (improved)
- Correlation: 0.90 → 0.84 (trade-off)

---

## 5. Evaluation Metrics

### Key Metrics (evaluate.py)

| Metric | Target | Description |
|--------|--------|-------------|
| Correlation | > 0.9 | Spatial pattern matching |
| Unit circle violation | < 0.01 | Phase prediction consistency |
| Haze score | < 0.3 | Good fringe contrast |
| F_A hole fraction | < 2% | F_A signal continuity |

### Haze Diagnostic

The "haze" artifact causes fringe valleys to not go deep enough relative to local peaks.

**Measurement method:**
- Uses 7x7x7 sliding window to compute local max around each voxel
- Valley depth = 1 - (value / local_max)
- Deep valleys (depth > 0.8) = good, shallow (depth < 0.5) = haze

**Interpretation:**
- Local fringe contrast > 0.8: Excellent
- 0.6-0.8: Good
- 0.4-0.6: Moderate haze
- < 0.4: Significant haze

---

## 6. Known Limitations

1. **2D Architecture**: CNN is 2D; 3D processed slice-by-slice (no z-axis correlation)
2. **Haze Artifact**: Fringe valleys don't go as deep as ground truth (mitigated with `--enhance-contrast`)
3. **No Uncertainty Quantification**: Single point estimates, no error bars
4. **Self-Consistency R-factor**: High (~5000) due to simplified MAD equation in evaluation; use correlation instead

### High R-factor Explanation

The `evaluate.py` script uses a simplified MAD equation that doesn't match the full physics in `core_shell.py`. This causes a **scale mismatch** leading to high R-factors. This is expected - focus on **correlation** (>0.9 is good) instead.

---

## References

- CLAUDE.md (now docs/DEVELOPMENT.md) - Project context for AI assistants
- README.md - User documentation
- docs/ARCHITECTURE.md - CNN design rationale
- docs/PHYSICS.md - Physics background

---

*Last updated: December 2024*
