# NanoMAD ML Insight: Technical Guidance for the ML Approach

**Purpose:** Provide detailed answers to the questions raised in `NanoMAD_ML_Approach.md`, based on analysis of the original NanoMAD source code, and offer additional insights to guide the ML implementation.

---

## Executive Summary

Your ML approach is fundamentally sound. The key insight is that NanoMAD's fitting is **local** (each Q-point is fitted independently), which means a CNN can learn this mapping. However, there are several physics-informed considerations that will significantly impact your success, particularly around:

1. **Phase representation** (use sin/cos, not raw angle)
2. **The f₀ normalization** (critical and Q-dependent)
3. **Degeneracies** (sign ambiguities that NanoMAD handles post-fit)
4. **Loss weighting** (intensity-weighted losses will help)

---

## Part 1: Answers to Physics Questions

### Q1: Is your ground truth correct?

**Yes, with one important clarification.**

Your definitions are correct:
- F_A = F_Ni (anomalous at Ni edge) ✓
- F_N = F_Fe (non-anomalous at Ni edge) ✓
- Δφ = arg(F_T) - arg(F_A) ✓

**However**, there's a subtlety about what F_T represents in NanoMAD. Looking at the MAD equation in the code ([NanoMAD.cpp line 384](NanoMAD.cpp#L384)):

```cpp
mIcalc(i) = mFT*mFT + fa*fa*(fs*fs+fp*fp) + 2*mFT*fa*(fp*c+fs*s);
```

The F_T in this equation is the total structure factor **excluding the energy-dependent anomalous terms** (f' and f''). In other words:

```
F_total_measured = F_T + (f'/f₀ + i·f''/f₀)·F_A
```

So F_T already includes the Thomson scattering from the anomalous atoms (via f₀), but not their resonant contribution. This is consistent with your definition where:

```
F_T = F_Ni + F_Fe  (with Ni's f₀ contribution, not f'+if'')
```

**Recommendation:** Your simulation is likely correct as long as you compute F_Ni using only f₀(Q) for the scattering factor (not f₀ + f' + if''). The energy-dependent parts are handled separately in the MAD equation.

---

### Q2: Phase wrapping — should you predict sin(Δφ) and cos(Δφ)?

**Strongly recommended: YES, predict sin(Δφ) and cos(Δφ) separately.**

This is exactly what NanoMAD does internally! Look at the `Icalc()` function:

```cpp
const REAL c=cos(mDeltaPhi);  // cos(Δφ)
const REAL s=sin(mDeltaPhi);  // sin(Δφ)
// ...
mIcalc(i)=... + 2*mFT*fa*(fp*c+fs*s);
```

The intensity depends on cos(Δφ) and sin(Δφ), not on Δφ directly. This means:

1. **The physics is naturally in sin/cos space** — the neural network sees intensity, which depends on sin/cos
2. **No discontinuity at ±π** — predicting sin/cos avoids the branch cut problem
3. **Easy to recover Δφ** — use `atan2(sin, cos)` after prediction

**Implementation:**

```python
# Output layer: 4 channels instead of 3
# [|F_T|, |F_A|, sin(Δφ), cos(Δφ)]

# Post-processing to get Δφ:
delta_phi = np.arctan2(pred_sin, pred_cos)
```

**Additional constraint:** Consider adding a soft constraint that sin²(Δφ) + cos²(Δφ) ≈ 1. You could either:
- Normalize the outputs: `sin_norm = sin / sqrt(sin² + cos²)`
- Add a regularization term to the loss: `λ·(sin² + cos² - 1)²`

---

### Q3: Normalization — what are typical ranges? Should you normalize by f₀?

**Critical insight from the code:** NanoMAD normalizes F_A by f₀(Q) internally.

From [NanoMAD.cpp line 383](NanoMAD.cpp#L383):
```cpp
const REAL fa = mFA/mF0;  // F_A normalized by Thomson factor
```

This normalization is important because:
- f₀(Q) varies with scattering angle (falls off at high Q)
- Without normalization, F_A values would have a strong Q-dependence that's just geometric, not physical

**Recommendation:**

**Option A (Simpler):** Have the CNN predict |F_T|, |F_A|, sin(Δφ), cos(Δφ) in raw units, then you handle the f₀ normalization in post-processing. This keeps the network outputs interpretable.

**Option B (Physics-informed):** Provide f₀(Q) as an additional input channel to the CNN, so it can learn the normalization. This might help generalization across different Q-ranges.

**Typical ranges:**
- Looking at line 162 in the code: `mFT=sqrt(tmp)` where `tmp` is the average intensity
- And line 163: `mFA=mFT/10.` as initial guess

This suggests |F_A| is typically ~10% of |F_T|, though this varies by system. For your Ni-Fe system:
- If you have 75% Ni in the core, F_A/F_T could be substantial
- Normalize by the maximum intensity in your dataset, or use log-scale for the structure factors

---

### Q4: Edge cases — what happens when |F_A| ≈ 0 or |F_T| ≈ |F_A|?

**The code handles these explicitly!**

From [NanoMAD.cpp lines 866-875](NanoMAD.cpp#L866-L875):
```cpp
if(MaxAbs((*pos)->Iobs())==0)
{
   // all-null intensity => put a dummy value
   cout<<"WARNING: Iobs=0 in all points ?"<<endl;
   vFtFaDphi0[*pos].push_back(ftfadphi0(0,0,0,0));
   continue;
}
```

And from [NanoMAD.cpp lines 1014-1024](NanoMAD.cpp#L1014-L1024):
```cpp
// If FT + FA * (f'+if")/f0 < 0, might be a false minimum
if(((*pos)->FT() < abs((*pos)->FA()*fpmin/(*pos)->F0()*cos((*pos)->DeltaPhi())))
   ||((*pos)->FT() < (*pos)->FA()))
{
   // try by starting with a much smaller FA
   (*pos)->FA() = (*pos)->FA()/4;
   // ... retry with damping
}
```

**Edge case behaviors:**

1. **|F_A| ≈ 0 (no anomalous scatterer):**
   - Δφ becomes poorly defined (any angle gives similar fit)
   - The interference term (Term 3) vanishes
   - NanoMAD can still fit |F_T|, but Δφ has large uncertainty

2. **|F_T| ≈ |F_A| (little non-anomalous contribution):**
   - This means |F_N| ≈ 0
   - Physically meaningful! (pure anomalous scatterer at this Q-point)
   - But can lead to numerical instability in computing F_N

3. **|F_T| < |F_A| (physically impossible in the MAD framework):**
   - This indicates a fitting failure
   - NanoMAD explicitly checks for this and retries

**Recommendations for ML:**

1. **Mask low-intensity regions:** Don't train on (or weight heavily) Q-points where total intensity is very low — the signal-to-noise is poor and Δφ is meaningless.

2. **Add physical constraints to the loss:**
   ```python
   # Soft constraint: |F_T| >= |F_A| (triangle inequality)
   constraint_loss = relu(|F_A| - |F_T|)
   ```

3. **Uncertainty estimation:** Consider having the network also predict uncertainties (σ_FT, σ_FA, σ_Δφ). This naturally handles edge cases where predictions should have low confidence.

---

## Part 2: Answers to ML Architecture Questions

### Q5: Local vs global — can each Q-point be predicted from local information?

**Yes, NanoMAD treats each Q-point completely independently!**

From [NanoMAD.cpp lines 958-1041](NanoMAD.cpp#L958-L1041), the LSQ fitting loop:
```cpp
for(vector<HKLMAD*>::iterator pos=vHKLMAD.begin(); pos!=vHKLMAD.end(); ++pos)
{
   LSQNumObj myLSQObj("Refining FT,FA & Delta Phi");
   myLSQObj.SetRefinedObj(**pos);  // Fit this single Q-point
   // ...
}
```

Each HKLMAD object (one per Q-point) is fitted independently. There are **no spatial correlations** enforced in NanoMAD between neighboring Q-points.

**However**, for your ML approach, using patches (16×16) instead of individual pixels is a good idea because:

1. **Noise reduction:** Neighboring Q-points should have similar (F_T, F_A, Δφ) values (structural continuity)
2. **Context helps:** The CNN can learn that structure factors vary smoothly in Q-space
3. **Regularization:** The patch-based approach implicitly regularizes against noisy pixel-wise predictions

**This is potentially an advantage of ML over NanoMAD** — NanoMAD doesn't exploit spatial correlations, but your CNN can.

---

### Q6: Loss function — should you weight by intensity?

**Yes, absolutely!**

NanoMAD does exactly this. From [NanoMAD.cpp lines 164-166](NanoMAD.cpp#L164-L166):
```cpp
mWeight.resize(mIobsSigma.numElements());
for(int i=0;i<mIobsSigma.numElements();++i)
   mWeight(i) = 1./(mIobsSigma(i)*mIobsSigma(i));  // 1/σ²
```

The weight is 1/σ², which for Poisson statistics means higher intensity → smaller relative error → more weight.

**Recommended loss function:**

```python
def mad_loss(pred, target, intensity):
    """
    Intensity-weighted loss for MAD parameter prediction.

    pred: [batch, H, W, 4] — |F_T|, |F_A|, sin(Δφ), cos(Δφ)
    target: [batch, H, W, 4] — ground truth
    intensity: [batch, H, W] — total intensity (sum over energies)
    """
    # Base MSE loss
    mse = (pred - target)**2

    # Weight by sqrt(intensity) — higher signal = more reliable
    # (or use intensity directly for stronger weighting)
    weights = torch.sqrt(intensity + 1e-6)  # +eps for stability
    weights = weights / weights.mean()  # normalize

    # Expand weights to match prediction shape
    weights = weights.unsqueeze(-1)  # [batch, H, W, 1]

    weighted_mse = (mse * weights).mean()

    # Add sin²+cos² constraint
    sin_pred, cos_pred = pred[..., 2], pred[..., 3]
    norm_constraint = ((sin_pred**2 + cos_pred**2) - 1)**2

    return weighted_mse + 0.1 * norm_constraint.mean()
```

---

### Q7: Degeneracies — are there known ambiguities?

**Yes! NanoMAD handles two important sign degeneracies.**

From [NanoMAD.cpp lines 186-199](NanoMAD.cpp#L186-L199) and [405-419](NanoMAD.cpp#L405-L419):

```cpp
void HKLMAD::GlobalOptRandomMove(...)
{
   // ...
   if(mFT<0)
   {
      mFT=abs(mFT);
      mDeltaPhi+=M_PI;  // Flip phase by π
   }
   if(mFA<0)
   {
      mFA=abs(mFA);
      mDeltaPhi+=M_PI;  // Flip phase by π
   }
}
```

**The degeneracies:**

1. **(F_T, Δφ) ↔ (-F_T, Δφ+π):** Flipping the sign of F_T and adding π to the phase gives the same intensity. NanoMAD enforces F_T > 0.

2. **(F_A, Δφ) ↔ (-F_A, Δφ+π):** Same for F_A. NanoMAD enforces F_A > 0.

**For ML training:**

- **Enforce positive magnitudes:** Your ground truth should always have |F_T| ≥ 0 and |F_A| ≥ 0
- **Use ReLU or softplus for magnitude outputs:** This ensures predictions are always positive
- **Be careful with phases:** If your ground truth has F_T or F_A with negative real parts, the "phase" relative to some absolute reference could differ from NanoMAD's convention

**Recommendation:**
```python
# Output layer
F_T_mag = softplus(raw_output[..., 0])  # Always positive
F_A_mag = softplus(raw_output[..., 1])  # Always positive
sin_dphi = tanh(raw_output[..., 2])     # Range [-1, 1]
cos_dphi = tanh(raw_output[..., 3])     # Range [-1, 1]
```

---

## Part 3: Answers to Validation Questions

### Q8: Comparison metric — how to compare ML to NanoMAD?

**Multi-level validation:**

1. **Parameter-level (direct comparison):**
   ```python
   # For each Q-point:
   error_FT = |F_T_ml - F_T_nanomad| / F_T_nanomad
   error_FA = |F_A_ml - F_A_nanomad| / F_A_nanomad
   error_dphi = |wrap(Δφ_ml - Δφ_nanomad)|  # wrap to [-π, π]
   ```

2. **Intensity reconstruction (physics check):**
   Use your ML predictions to reconstruct the intensity via the MAD equation:
   ```python
   I_reconstructed(E) = |F_T|² + (f'²+f''²)|F_A/f₀|² + 2|F_T||F_A|/f₀·[f'cos(Δφ)+f''sin(Δφ)]

   R_factor = Σ|I_obs - I_reconstructed| / Σ|I_obs|
   ```
   This checks that your predictions are self-consistent with the physics.

3. **Derived quantity check (F_N):**
   ```python
   F_N_ml = compute_FN(F_T_ml, F_A_ml, dphi_ml)
   F_N_nanomad = compute_FN(F_T_nanomad, F_A_nanomad, dphi_nanomad)
   # Compare these
   ```

4. **Downstream task (reconstruction quality):**
   The ultimate test — does phase retrieval on F_A and F_N give sensible real-space images?

---

### Q9: Does ML-derived F_A reconstruct to a sensible image?

This is the critical test. Key things to check:

1. **Support consistency:** Does the reconstructed |ρ_Ni(r)| match the expected particle shape?

2. **Core-shell structure:** For your Ni₃Fe core / Ni shell system, the reconstructed Ni density should show the full particle (core + shell).

3. **Phase consistency:** The strain field (phase of the reconstruction) should be continuous and physically reasonable.

4. **Composition ratio:** When you compute F_A/F_N or the real-space ratio ρ_Ni/ρ_Fe, does it give ~75% Ni in the core as expected?

**Potential pitfall:** If the ML makes systematic errors in Δφ, the phase retrieval might converge but give wrong strain fields. Always check against known simulation ground truth first.

---

### Q10: Experimental datasets for testing

For eventual experimental validation:

1. **Start with high-flux data** where NanoMAD already works well — use this to verify ML matches NanoMAD on "easy" cases.

2. **Then test on challenging data** where NanoMAD struggles (lower flux, higher noise) — this is where ML could potentially outperform.

3. **Known samples:** Test on samples with known composition (e.g., pure Ni particle as control — F_Fe should be ~0).

---

## Part 4: Additional Insights from the NanoMAD Code

### Insight 1: The Grid Search Starting Points

NanoMAD doesn't just do gradient descent — it first does a coarse grid search ([lines 860-954](NanoMAD.cpp#L860-L954)):

```cpp
const REAL stepFt=max_ft/25.0;   // 25 steps
const REAL stepFa=max_fa/25.0;   // 25 steps
const REAL stepPhi=2*M_PI/24.0;  // 24 steps (~15°)
```

This is 25 × 25 × 24 = 15,000 evaluations per Q-point! The grid search finds good starting points before LSQ refinement.

**Implication for ML:** Your CNN essentially learns to skip this expensive grid search. The network should learn to make good "initial guesses" that are already close to the optimum.

---

### Insight 2: The Thomson Factor is Q-Dependent

From [NanoMAD.cpp lines 146-151](NanoMAD.cpp#L146-L151):

```cpp
REAL x=h; REAL y=k; REAL z=l;
cell.MillerToOrthonormalCoords(x,y,z);
const REAL sithsl=sqrt(x*x+y*y+z*z)/2.;  // sin(θ)/λ
mF0=at.GetScatteringFactor(sithsl);       // f₀ at this Q
```

f₀ is computed for each Q-point using sin(θ)/λ. This means:
- High-Q regions have lower f₀ (scattering falls off)
- The normalization `fa = mFA/mF0` is different at each Q-point

**For ML:** If you want the network to generalize across Q-space, providing f₀(Q) as an input (or the Q-magnitude) could help.

---

### Insight 3: NanoMAD's Uncertainty Estimation

NanoMAD computes uncertainties using the variance-covariance matrix from LSQ fitting ([lines 292-369](NanoMAD.cpp#L292-L369)). The uncertainties on F_N and Δφ_N-A are propagated from the fitted parameters using standard error propagation.

**For ML:** Consider training a network to predict uncertainties alongside the parameters. This could be done with:
- A separate output head for uncertainties
- A probabilistic network (e.g., mixture density network)
- Monte Carlo dropout for uncertainty estimation

---

## Part 5: Recommended Architecture

Based on my analysis, here's a suggested architecture:

```
Input: [batch, 16, 16, 8]  — intensities at 8 energies
       [batch, 16, 16, 1]  — f₀(Q) for each pixel (optional)
       [batch, 8]          — f'(E), f''(E) values (shared across patch)

Encoder:
  Conv2D(8→32, 3×3) → BatchNorm → ReLU
  Conv2D(32→64, 3×3) → BatchNorm → ReLU
  Conv2D(64→128, 3×3) → BatchNorm → ReLU

  # Inject f', f'' as a global context
  # (these are the same for all Q-points, just different per energy)
  # Could concatenate encoded f'/f'' features to bottleneck

Decoder:
  Conv2D(128→64, 3×3) → BatchNorm → ReLU
  Conv2D(64→32, 3×3) → BatchNorm → ReLU
  Conv2D(32→4, 1×1)  — output: [|F_T|, |F_A|, sin(Δφ), cos(Δφ)]

Output activations:
  |F_T|: softplus (ensures positive)
  |F_A|: softplus (ensures positive)
  sin(Δφ): tanh (range [-1, 1])
  cos(Δφ): tanh (range [-1, 1])

Output: [batch, 16, 16, 4]
```

**Key design choices:**
1. **Fully convolutional:** Preserves spatial structure, can handle different patch sizes
2. **sin/cos output:** Avoids phase wrapping issues
3. **Softplus for magnitudes:** Ensures physical constraints (positive values)
4. **f'/f'' injection:** The energy-dependent factors are global knowledge, not spatial

---

## Summary: Key Takeaways

| Topic | Recommendation |
|-------|----------------|
| Ground truth | Your definitions are correct; ensure F_Ni uses only f₀, not f'+if'' |
| Phase representation | **Use sin(Δφ) and cos(Δφ)** — avoids discontinuities |
| Normalization | Normalize by max intensity; optionally include f₀(Q) as input |
| Edge cases | Mask low-intensity regions; add soft constraint |F_T| ≥ |F_A| |
| Locality | Each Q-point is independent in NanoMAD; patches help with regularization |
| Loss weighting | **Weight by intensity** — higher signal = more reliable |
| Degeneracies | Enforce positive magnitudes (softplus); watch sign conventions |
| Validation | Multi-level: parameter comparison, intensity reconstruction, real-space reconstruction |

**The ML approach has a real advantage:** It can exploit spatial correlations between neighboring Q-points (via the patch-based architecture), which NanoMAD ignores. This could be the key to handling noisy data better.

---

*This document provides technical guidance based on analysis of the NanoMAD source code. Good luck with the ML implementation — the physics is sound and the approach is promising!*
