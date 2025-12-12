# CNN Design for MAD Parameter Prediction

**Goal:** Design a CNN that takes multi-energy diffraction intensity patches and outputs the MAD parameters (|F_T|, |F_A|, sin(Δφ), cos(Δφ)), bypassing NanoMAD's iterative fitting entirely.

---

## Design Philosophy

### What the CNN Must Learn

The CNN needs to learn the **inverse mapping** of the MAD equation:

```
Forward (physics):  (|F_T|, |F_A|, Δφ, f', f'') → I(E)
Inverse (CNN):      I(E), f', f'' → (|F_T|, |F_A|, sin(Δφ), cos(Δφ))
```

The key insight: **the CNN doesn't need to know the MAD equation explicitly**. Given enough training data with (input, label) pairs, it can learn this inverse mapping purely from examples.

### Input Design: What to Include?

#### f' and f'' — INCLUDE

**Decision: Include f' and f'' as inputs.**

The anomalous dispersion corrections f'(E) and f''(E) should be provided to the network. Here's why:

**Arguments FOR including f'/f'':**

1. **Energy range flexibility** — You may want to experiment with different energy ranges (±5 eV, ±10 eV, ±50 eV around E₀). Without f'/f'', you'd need to retrain for each choice.

2. **Robustness to experimental variation** — Exact energies may vary slightly between experiments due to beamline calibration. Providing f'/f'' tells the network "here's what this energy channel actually means."

3. **Future generalization** — Eventually you may want to apply this to other bimetallic systems (not just NiFe) or other absorption edges. Including f'/f'' makes a single trained model potentially usable across different elements.

4. **Physics-informed** — The MAD equation explicitly depends on f' and f''. Giving the network this information is giving it the "key" to decode the energy dependence.

**Arguments AGAINST (why you might skip it):**

1. **Simpler architecture** — Fewer inputs, fewer things to go wrong.

2. **Fixed experimental setup** — If you always use the exact same 8 energies, the network can learn the mapping implicitly.

**Verdict:** The flexibility benefits outweigh the added complexity. Include f'/f''.

---

#### f₀ (Thomson factor) — SKIP (for now)

**Decision: Do NOT include f₀ as input (initially).**

The Thomson scattering factor f₀ is Q-dependent (varies across pixels in your patch) but energy-independent (same at all 8 energies for a given pixel).

**Why f₀ is less critical than f'/f'':**

1. **f₀ varies smoothly with Q** — It's essentially a slow-varying envelope that falls off at high Q. Within a 16×16 patch, f₀ doesn't change dramatically.

2. **The CNN can learn Q-dependence implicitly** — If your training data includes patches from different Q-regions, the network will learn that high-Q patches have systematically lower intensities.

3. **f₀ is deterministic** — Unlike f'/f'' which you choose experimentally, f₀ is fixed by physics for a given element and Q. The network can learn this relationship from the data.

4. **In the MAD equation, f₀ appears only as normalization:**
   ```cpp
   const REAL fa = mFA/mF0;  // F_A normalized by f₀
   ```
   The network can learn to predict |F_A| directly without needing explicit f₀.

**When to reconsider:**

If you observe systematic errors that correlate with Q-position (e.g., works well near Bragg peak, fails in tails), consider adding |Q| as an input channel:
```python
Q_magnitude = sqrt(Qx² + Qy² + Qz²)  # [16, 16] — one value per pixel
```
This is simpler than f₀ and provides the same information.

---

### Summary: Input/Output Design

```
INPUTS:
├── intensity: [batch, 16, 16, N_energies]  — REQUIRED
│       Multi-energy diffraction intensities
│
├── f': [batch, N_energies]                  — INCLUDE
│       Real anomalous correction at each energy
│
├── f'': [batch, N_energies]                 — INCLUDE
│       Imaginary anomalous correction at each energy
│
└── f₀ or |Q|: [batch, 16, 16]              — SKIP (add later if needed)
        Thomson factor or Q-magnitude per pixel

OUTPUTS:
└── [batch, 16, 16, 4]
        ├── |F_T|      — Total structure factor magnitude (Softplus activation)
        ├── |F_A|      — Anomalous structure factor magnitude (Softplus activation)
        ├── sin(Δφ)    — Sine of phase difference (Tanh activation)
        └── cos(Δφ)    — Cosine of phase difference (Tanh activation)
```

**Note on variable N_energies:** If you want to support different numbers of energy channels, you'll need either:
- Pad to a fixed maximum (e.g., always 12 channels, zero-pad unused ones)
- Use an architecture that handles variable-length sequences (attention-based)

For simplicity, start with fixed N_energies=8.

---

## Architecture Design

### Option 1: U-Net with f'/f'' Injection (Recommended)

A U-Net captures both local details and broader context through skip connections. The f'/f'' values are encoded and injected at the bottleneck.

```
INPUTS:
  intensity: [batch, 16, 16, 8]  — 8 energy channels
  fp:        [batch, 8]          — f' at each energy
  fs:        [batch, 8]          — f'' at each energy

┌─────────────────────────────────────────────────────────────────┐
│  f'/f'' ENCODER (global context, shared across all pixels)      │
│                                                                 │
│  Concat(fp, fs) → [batch, 16]                                   │
│  Linear(16→32) → ReLU → Linear(32→32) → ReLU                    │
│  → fpfs_features [batch, 32]                                    │
└─────────────────────────────────────────────────────────────────┘

INTENSITY ENCODER (contracting path):
┌─────────────────────────────────────────────┐
│ Conv2D(8→32, 3×3, padding='same')           │
│ BatchNorm → LeakyReLU(0.1)                  │
│ Conv2D(32→32, 3×3, padding='same')          │
│ BatchNorm → LeakyReLU(0.1)                  │
│ → skip1 [16, 16, 32]                        │
│ MaxPool2D(2×2) → [8, 8, 32]                 │
├─────────────────────────────────────────────┤
│ Conv2D(32→64, 3×3, padding='same')          │
│ BatchNorm → LeakyReLU(0.1)                  │
│ Conv2D(64→64, 3×3, padding='same')          │
│ BatchNorm → LeakyReLU(0.1)                  │
│ → skip2 [8, 8, 64]                          │
│ MaxPool2D(2×2) → [4, 4, 64]                 │
├─────────────────────────────────────────────┤
│ Conv2D(64→128, 3×3, padding='same')         │
│ BatchNorm → LeakyReLU(0.1)                  │
│ Conv2D(128→128, 3×3, padding='same')        │
│ BatchNorm → LeakyReLU(0.1)                  │
│ → [4, 4, 128]                               │
└─────────────────────────────────────────────┘

BOTTLENECK (inject f'/f'' features):
┌─────────────────────────────────────────────┐
│ fpfs_features [batch, 32]                   │
│   → expand to [batch, 32, 4, 4]             │
│   → concat with encoder output              │
│   → [batch, 160, 4, 4]                      │
│                                             │
│ Conv2D(160→128, 3×3, padding='same')        │
│ BatchNorm → LeakyReLU(0.1)                  │
└─────────────────────────────────────────────┘

DECODER (expanding path):
┌─────────────────────────────────────────────┐
│ UpSample2D(2×2) → [8, 8, 128]               │
│ Concat(skip2) → [8, 8, 192]                 │
│ Conv2D(192→64, 3×3, padding='same')         │
│ BatchNorm → LeakyReLU(0.1)                  │
│ Conv2D(64→64, 3×3, padding='same')          │
│ BatchNorm → LeakyReLU(0.1)                  │
├─────────────────────────────────────────────┤
│ UpSample2D(2×2) → [16, 16, 64]              │
│ Concat(skip1) → [16, 16, 96]                │
│ Conv2D(96→32, 3×3, padding='same')          │
│ BatchNorm → LeakyReLU(0.1)                  │
│ Conv2D(32→32, 3×3, padding='same')          │
│ BatchNorm → LeakyReLU(0.1)                  │
└─────────────────────────────────────────────┘

OUTPUT HEAD:
┌─────────────────────────────────────────────┐
│ Conv2D(32→4, 1×1)  — raw output             │
│                                             │
│ Split into 4 channels:                      │
│   ch0: Softplus → |F_T|  (positive)         │
│   ch1: Softplus → |F_A|  (positive)         │
│   ch2: Tanh → sin(Δφ)    (range [-1,1])     │
│   ch3: Tanh → cos(Δφ)    (range [-1,1])     │
└─────────────────────────────────────────────┘

Output: [batch, 16, 16, 4]
```

**Parameter count:** ~200K parameters (very lightweight)

**Why inject f'/f'' at the bottleneck?**
- The bottleneck is where the network has the most "abstract" representation
- f'/f'' provides global context (same for all pixels) — this is the right place to inject it
- Doesn't interfere with spatial feature extraction in encoder
- The decoder then uses this context to make pixel-wise predictions

**Why U-Net?**
- Skip connections preserve spatial detail (important for high-frequency structure in diffraction)
- Encoder captures context (neighboring Q-points)
- Decoder reconstructs pixel-wise predictions
- Well-established, easy to train

---

### PyTorch Implementation of Option 1

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two conv layers with BatchNorm and LeakyReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class MAD_UNet(nn.Module):
    """
    U-Net for MAD parameter prediction with f'/f'' injection.

    Inputs:
        intensity: [batch, H, W, N_energies] — multi-energy diffraction
        fp: [batch, N_energies] — f' at each energy
        fs: [batch, N_energies] — f'' at each energy

    Output:
        [batch, H, W, 4] — |F_T|, |F_A|, sin(Δφ), cos(Δφ)
    """
    def __init__(self, n_energies=8):
        super().__init__()

        # f'/f'' encoder (global context)
        self.fpfs_encoder = nn.Sequential(
            nn.Linear(n_energies * 2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        # Intensity encoder
        self.enc1 = ConvBlock(n_energies, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)

        # Bottleneck (with f'/f'' injection)
        self.bottleneck = ConvBlock(128 + 32, 128)  # +32 from fpfs

        # Decoder
        self.dec2 = ConvBlock(128 + 64, 64)  # +64 from skip2
        self.dec1 = ConvBlock(64 + 32, 32)   # +32 from skip1

        # Output head
        self.output = nn.Conv2d(32, 4, 1)

        # Activations for output channels
        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()

    def forward(self, intensity, fp, fs):
        # intensity: [batch, H, W, C] → [batch, C, H, W]
        x = intensity.permute(0, 3, 1, 2)

        # Encode f'/f''
        fpfs = torch.cat([fp, fs], dim=-1)  # [batch, 16]
        fpfs_feat = self.fpfs_encoder(fpfs)  # [batch, 32]

        # Encoder
        e1 = self.enc1(x)                    # [batch, 32, H, W]
        e2 = self.enc2(F.max_pool2d(e1, 2))  # [batch, 64, H/2, W/2]
        e3 = self.enc3(F.max_pool2d(e2, 2))  # [batch, 128, H/4, W/4]

        # Inject f'/f'' at bottleneck
        B, C, H4, W4 = e3.shape
        fpfs_spatial = fpfs_feat[:, :, None, None].expand(-1, -1, H4, W4)
        b = self.bottleneck(torch.cat([e3, fpfs_spatial], dim=1))

        # Decoder with skip connections
        d2 = self.dec2(torch.cat([F.interpolate(b, scale_factor=2, mode='nearest'), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2, mode='nearest'), e1], dim=1))

        # Output
        raw = self.output(d1)  # [batch, 4, H, W]

        # Apply activations per channel
        FT = self.softplus(raw[:, 0:1])   # |F_T| ≥ 0
        FA = self.softplus(raw[:, 1:2])   # |F_A| ≥ 0
        sin_dphi = self.tanh(raw[:, 2:3]) # sin(Δφ) ∈ [-1, 1]
        cos_dphi = self.tanh(raw[:, 3:4]) # cos(Δφ) ∈ [-1, 1]

        out = torch.cat([FT, FA, sin_dphi, cos_dphi], dim=1)

        # [batch, 4, H, W] → [batch, H, W, 4]
        return out.permute(0, 2, 3, 1)


# Usage example:
# model = MAD_UNet(n_energies=8)
# pred = model(intensity, fp, fs)  # [batch, 16, 16, 4]
```

---

### Option 2: Pure Convolutional (No Pooling)

If you want to stay fully local and avoid any information mixing between distant Q-points:

```
Input: [batch, 16, 16, 8]

Conv2D(8→32, 3×3, padding='same') → BatchNorm → LeakyReLU
Conv2D(32→64, 3×3, padding='same') → BatchNorm → LeakyReLU
Conv2D(64→64, 3×3, padding='same') → BatchNorm → LeakyReLU
Conv2D(64→64, 3×3, padding='same') → BatchNorm → LeakyReLU
Conv2D(64→32, 3×3, padding='same') → BatchNorm → LeakyReLU
Conv2D(32→4, 1×1) → output activations

Output: [batch, 16, 16, 4]
```

**Receptive field:** With 5 layers of 3×3 convolutions, each output pixel "sees" an 11×11 region of the input. This provides local context without global mixing.

**Pros:** Simple, fast, respects locality
**Cons:** May not capture larger-scale correlations

---

### Option 3: ResNet-Style with Residual Blocks

For deeper networks with better gradient flow:

```
Input: [batch, 16, 16, 8]

# Initial projection
Conv2D(8→64, 3×3, padding='same') → BatchNorm → LeakyReLU

# Residual blocks (no downsampling to preserve resolution)
ResBlock(64→64) × 4
  Each block:
    Conv2D(64→64, 3×3) → BatchNorm → LeakyReLU
    Conv2D(64→64, 3×3) → BatchNorm
    + skip connection → LeakyReLU

# Output
Conv2D(64→4, 1×1) → output activations

Output: [batch, 16, 16, 4]
```

---

## Loss Function Design

### Base Loss: Weighted MSE

```python
def weighted_mse_loss(pred, target, weights):
    """
    pred: [batch, H, W, 4] — |F_T|, |F_A|, sin(Δφ), cos(Δφ)
    target: [batch, H, W, 4] — ground truth
    weights: [batch, H, W] — per-pixel weights
    """
    mse = (pred - target) ** 2  # [batch, H, W, 4]

    # Weight each pixel
    weights = weights.unsqueeze(-1)  # [batch, H, W, 1]
    weighted_mse = mse * weights

    return weighted_mse.mean()
```

### What Should the Weights Be?

**Option A: Intensity-based (recommended)**
```python
# Sum intensity across energies
total_intensity = input_intensities.sum(dim=-1)  # [batch, H, W]

# Sqrt scaling (Poisson-like: higher signal = more reliable)
weights = torch.sqrt(total_intensity + 1e-6)

# Normalize so weights average to 1
weights = weights / weights.mean()
```

**Option B: Structure factor magnitude**
```python
# Weight more where there's actual signal
weights = target_FT + 1e-6  # Weight by |F_T|
weights = weights / weights.mean()
```

**Option C: Binary mask**
```python
# Only train on pixels above noise floor
threshold = 0.01 * total_intensity.max()
weights = (total_intensity > threshold).float()
```

I recommend **Option A** — it's simple, physically motivated, and doesn't require ground truth values.

---

### Physics Constraints in the Loss

#### Constraint 1: sin² + cos² = 1

```python
def unit_circle_loss(pred):
    """Soft constraint that sin²(Δφ) + cos²(Δφ) ≈ 1"""
    sin_pred = pred[..., 2]
    cos_pred = pred[..., 3]
    norm_sq = sin_pred**2 + cos_pred**2
    return ((norm_sq - 1) ** 2).mean()
```

#### Constraint 2: |F_T| ≥ |F_A| (triangle inequality)

This comes from F_T = F_A + F_N, where F_N has non-negative magnitude.

Actually, wait — this isn't strictly true! If F_A and F_N point in opposite directions, |F_T| could be less than |F_A|. Let me reconsider...

The correct constraint is: **|F_T| ≥ ||F_A| - |F_N||** (triangle inequality).

But since we don't predict |F_N| directly, we can't enforce this easily. Instead, let the network learn it from data. If your ground truth satisfies physical constraints, the network should learn them implicitly.

**Skip this constraint** — it's not strictly valid and could hurt training.

#### Constraint 3: Intensity Reconstruction Loss (Physics-Informed)

This is powerful: use the MAD equation to check if predictions are self-consistent.

```python
def intensity_reconstruction_loss(pred, input_intensities, fp, fs, f0):
    """
    Check: do the predicted parameters reconstruct the input intensities?

    pred: [batch, H, W, 4] — |F_T|, |F_A|, sin(Δφ), cos(Δφ)
    input_intensities: [batch, H, W, 8] — observed I(E)
    fp: [8] — f' at each energy
    fs: [8] — f'' at each energy
    f0: [batch, H, W] — Thomson factor (or scalar if uniform)
    """
    FT = pred[..., 0]  # [batch, H, W]
    FA = pred[..., 1]
    sin_dphi = pred[..., 2]
    cos_dphi = pred[..., 3]

    fa = FA / f0  # Normalized

    # Reconstruct intensity at each energy
    I_recon = []
    for i in range(8):
        I_i = (FT**2
               + fa**2 * (fp[i]**2 + fs[i]**2)
               + 2 * FT * fa * (fp[i] * cos_dphi + fs[i] * sin_dphi))
        I_recon.append(I_i)

    I_recon = torch.stack(I_recon, dim=-1)  # [batch, H, W, 8]

    # Compare to input
    recon_error = ((I_recon - input_intensities) ** 2).mean()

    return recon_error
```

**Note:** This requires f', f'', and f₀. If you don't want to provide these to the network, you can still use them in the loss function during training — it's just a consistency check.

**This is a very powerful regularizer** because it ties the predictions back to the actual physics. Even if the network finds a different (F_T, F_A, Δφ) than NanoMAD would, if it reconstructs the intensity correctly, it's a valid solution.

---

### Complete Loss Function

```python
class MADLoss(nn.Module):
    def __init__(self, lambda_unit=0.1, lambda_recon=0.1, use_recon_loss=True):
        super().__init__()
        self.lambda_unit = lambda_unit
        self.lambda_recon = lambda_recon
        self.use_recon_loss = use_recon_loss

    def forward(self, pred, target, input_intensities, fp=None, fs=None, f0=None):
        """
        pred: [batch, H, W, 4]
        target: [batch, H, W, 4]
        input_intensities: [batch, H, W, 8]
        fp, fs: [8] anomalous corrections (optional, for recon loss)
        f0: [batch, H, W] Thomson factor (optional)
        """

        # 1. Intensity-weighted MSE on parameters
        total_I = input_intensities.sum(dim=-1)
        weights = torch.sqrt(total_I + 1e-6)
        weights = weights / weights.mean()

        mse = (pred - target) ** 2
        weighted_mse = (mse * weights.unsqueeze(-1)).mean()

        # 2. Unit circle constraint for sin/cos
        sin_pred = pred[..., 2]
        cos_pred = pred[..., 3]
        unit_loss = ((sin_pred**2 + cos_pred**2 - 1) ** 2).mean()

        # 3. Intensity reconstruction loss (optional)
        recon_loss = 0.0
        if self.use_recon_loss and fp is not None:
            recon_loss = self.intensity_reconstruction_loss(
                pred, input_intensities, fp, fs, f0
            )

        # Total loss
        total_loss = (weighted_mse
                      + self.lambda_unit * unit_loss
                      + self.lambda_recon * recon_loss)

        return total_loss, {
            'mse': weighted_mse.item(),
            'unit': unit_loss.item(),
            'recon': recon_loss.item() if isinstance(recon_loss, torch.Tensor) else 0.0
        }
```

---

## Training Strategy

### Data Augmentation

```python
def augment_patch(intensity_patch, labels):
    """
    Augment a single training sample.

    intensity_patch: [16, 16, 8]
    labels: [16, 16, 4] — |F_T|, |F_A|, sin(Δφ), cos(Δφ)
    """

    # 1. Random 90° rotations (preserves physics)
    k = np.random.randint(0, 4)
    intensity_patch = np.rot90(intensity_patch, k, axes=(0, 1))
    labels = np.rot90(labels, k, axes=(0, 1))

    # 2. Random flips
    if np.random.random() > 0.5:
        intensity_patch = np.flip(intensity_patch, axis=0)
        labels = np.flip(labels, axis=0)
    if np.random.random() > 0.5:
        intensity_patch = np.flip(intensity_patch, axis=1)
        labels = np.flip(labels, axis=1)

    # 3. Intensity scaling (simulate different exposure times)
    scale = np.random.uniform(0.5, 2.0)
    intensity_patch = intensity_patch * scale
    # Note: |F_T| and |F_A| should scale as sqrt(intensity)
    labels[..., 0] *= np.sqrt(scale)  # |F_T|
    labels[..., 1] *= np.sqrt(scale)  # |F_A|
    # sin(Δφ) and cos(Δφ) are unchanged

    # 4. Add noise (Poisson-like)
    if np.random.random() > 0.5:
        noise_level = np.random.uniform(0.01, 0.1)
        noise = np.random.normal(0, noise_level * np.sqrt(intensity_patch + 1))
        intensity_patch = np.maximum(intensity_patch + noise, 0)

    return intensity_patch, labels
```

### Training Schedule

```python
# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Learning rate schedule: cosine annealing with warm restarts
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=50, T_mult=2
)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()

        pred = model(batch['intensity'])
        loss, loss_dict = criterion(
            pred,
            batch['labels'],
            batch['intensity']
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    scheduler.step()
```

### Curriculum Learning (Optional but Helpful)

Start with easy examples, gradually increase difficulty:

```python
# Phase 1 (epochs 0-50): Train on noise-free data
# Phase 2 (epochs 50-100): Add moderate noise
# Phase 3 (epochs 100+): Full noise levels

def get_noise_level(epoch):
    if epoch < 50:
        return 0.0
    elif epoch < 100:
        return 0.05 * (epoch - 50) / 50  # Ramp from 0 to 0.05
    else:
        return 0.05 + 0.05 * min((epoch - 100) / 100, 1.0)  # Ramp to 0.1
```

---

## Summary: My Recommended Design

### Architecture
**U-Net style** with:
- Input: [batch, 16, 16, 8] — raw intensities only
- Encoder: 3 levels, (32→64→128 channels)
- Decoder: Mirror of encoder with skip connections
- Output: [batch, 16, 16, 4] with Softplus for magnitudes, Tanh for sin/cos

### Loss Function
```python
L = L_weighted_MSE + 0.1 * L_unit_circle + 0.1 * L_intensity_reconstruction
```

Where:
- **L_weighted_MSE**: MSE weighted by sqrt(total_intensity)
- **L_unit_circle**: (sin² + cos² - 1)² constraint
- **L_intensity_reconstruction**: Check predictions against MAD equation (optional, requires f'/f'')

### Training
- AdamW optimizer, lr=1e-3
- Cosine annealing schedule
- Data augmentation: rotations, flips, intensity scaling, noise injection
- Optional curriculum: start noise-free, gradually add noise

### Key Design Decisions

| Decision | Recommendation | Reason |
|----------|----------------|--------|
| f'/f'' as input? | **No** (to start) | Network can learn energy dependence implicitly |
| Architecture | U-Net | Balances local detail and context |
| Phase output | sin(Δφ), cos(Δφ) | Avoids discontinuity at ±π |
| Magnitude activation | Softplus | Ensures positive values |
| Loss weighting | sqrt(intensity) | Higher signal = more reliable |
| Physics constraint | Intensity reconstruction | Most powerful regularizer |

---

## What Makes This Different from NanoMAD

| Aspect | NanoMAD | CNN Approach |
|--------|---------|--------------|
| Per-pixel independence | Yes (each Q-point fitted alone) | No (CNN sees spatial context) |
| Exploits neighbor correlation | No | Yes (via convolutions) |
| Computational cost | O(N × iterations) | O(1) forward pass |
| Noise handling | None (fits each point) | Learned denoising via spatial context |
| Requires f'/f'' | Yes (explicitly) | No (learned implicitly) |

**The main advantage of the CNN approach for your noise problem:** NanoMAD treats each pixel independently, so noise in one pixel has no way to be "smoothed out" by neighbors. The CNN naturally uses spatial context, which should help with noise robustness.

---

*Good luck with the implementation! Start simple (Option 1 U-Net, basic loss), verify it works on noise-free data, then add complexity as needed.*
