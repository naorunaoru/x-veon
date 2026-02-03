# Experiment Tracking

This document tracks all training experiments, their status, and key findings.

## Status Legend

- ✓ **SUCCESS** - Working, ready for production use
- ⚠️ **WIP** - Work in progress, incomplete
- ✗ **FAILED** - Completed but failed/broken, do not use
- 📊 **ANALYSIS** - Completed, analyzing results
- 🔍 **PLANNED** - Planned but not started

---

## Experiments by Date

### v4 - Linear Space Training ✓ SUCCESS

**Date:** 2026-02-02 to 2026-02-03
**Script:** [scripts/train/train_v4.py](../scripts/train/train_v4.py)
**Checkpoint:** [checkpoints/v4_production/best.pt](../checkpoints/v4_production/best.pt)
**Status:** ✓ **Production Model**

**Approach:**
- Dataset: v4 (DHT-demosaiced RAFs, 4× downsampled, linear space)
- Input: Actual CFA from RAF (raw sensor data)
- Target: DHT demosaic of same RAF (acts as "teacher")
- Loss: L1 + 0.1×gradient + 0.05×chroma
- Training: ~100+ epochs
- Augmentation: Horizontal flip, vertical flip (CFA-safe)

**Results:**
- **PSNR:** 48.21 dB (validation)
- **Quality:** Good general quality, slightly soft on fine textures
- **HDR:** Successfully preserves highlights (values > 1.0)

**Key Insight:**
The model learns to produce output similar to DHT but can potentially exceed it since it's learning the statistical relationship, not copying DHT's algorithm.

**Logs:** [logs/train_v4.log](../logs/train_v4.log)

---

### v4_ft - JPEG Fine-tune ✗ FAILED

**Date:** 2026-02-03
**Scripts:**
- [_obsolete/train_v4_finetune.py](../_obsolete/train_v4_finetune.py)
- [_obsolete/train_v4_finetune_fast.py](../_obsolete/train_v4_finetune_fast.py)
**Checkpoint:** [checkpoints/experiments/checkpoints_v4_ft/best.pt](../checkpoints/experiments/checkpoints_v4_ft/best.pt)
**Status:** ✗ **DO NOT USE**

**Approach:**
- Started from v4 checkpoint
- Dataset: JPEGs converted to linear + synthetic CFA
- Used flip augmentation
- Goal: Improve texture quality

**Problem:**
Output shows periodic **6×6 grid artifacts** matching CFA pattern size.

**Hypothesis:**
Initially suspected flip augmentation breaks CFA alignment, but this may be incorrect. Flipping RGB before mosaicing should be valid since we use the same fixed CFA pattern.

**Possible Real Causes:**
1. Different data distribution (synthetic CFA from JPEGs vs real RAFs)
2. Training hyperparameters not tuned for fine-tuning
3. Learning rate too high for fine-tuning
4. Something in the synthetic CFA generation pipeline

**Next Steps:**
- 🔍 Investigate: Save intermediate outputs to visualize artifact formation
- 🔍 Try: Fine-tune without augmentation as control
- 🔍 Try: Use real RAFs instead of synthetic CFA from JPEGs

**Logs:** [logs/train_v4_ft.log](../logs/train_v4_ft.log)

---

### Torture Fine-tune ✗ FAILED

**Date:** 2026-02-03
**Script:** [scripts/train/experiments/train_torture.py](../scripts/train/experiments/train_torture.py)
**Dataset Script:** [scripts/utils/torture_v2.py](../scripts/utils/torture_v2.py)
**Checkpoint:** [checkpoints/experiments/checkpoints_torture/best.pt](../checkpoints/experiments/checkpoints_torture/best.pt)
**Status:** ✗ **DO NOT USE**

**Approach:**
- Started from v4 checkpoint
- Dataset: Torture test v2 (5000 synthetic patterns)
  - 13 pattern types: stripes, checkers, circles, edges, Julia fractals, Perlin noise
  - Smart color selection in HSV space
- Low learning rate (1e-5) for fine-tuning

**Problem:**
Complete **catastrophic forgetting**. Model outputs pixel mush on real images while maintaining performance on synthetic patterns.

**Root Cause:**
Synthetic torture patterns have fundamentally different statistics than real photos:
- No natural textures or lighting
- Extreme high-frequency content
- Uniform color distributions
- No camera noise characteristics

The model "forgot" how to process natural images while optimizing for synthetic edge cases.

**Lesson Learned:**
Pure synthetic fine-tuning doesn't work. Would need:
- Mixed dataset (90% real + 10% synthetic)
- Strong regularization to prevent forgetting
- Or: Use torture tests only for evaluation, not training

**Torture Dataset Details:**
- Location: `/Volumes/External/torture_dataset/`
- Size: 5000 samples, ~1.5GB
- Pattern types:
  - 0-10: Geometric (stripes, checkers, circles, edges, etc.)
  - 11: Julia set fractals (50% zoomed for detail)
  - 12: Perlin fBm noise

---

### v4_ssim - SSIM Loss Experiment ⚠️ WIP

**Date:** 2026-02-03
**Scripts:**
- [scripts/train/experiments/train_v4_ssim.py](../scripts/train/experiments/train_v4_ssim.py)
- [scripts/train/experiments/train_v4_ssim_fixed.py](../scripts/train/experiments/train_v4_ssim_fixed.py)
**Checkpoint:** [checkpoints/experiments/checkpoints_v4_ssim/](../checkpoints/experiments/checkpoints_v4_ssim/)
**Status:** ⚠️ **Not Started** (empty checkpoint directory)

**Hypothesis:**
v4 model is slightly soft on fine textures. SSIM loss might help preserve texture detail while maintaining structural quality.

**Approach:**
- Start from v4 checkpoint or train from scratch
- Loss: Combination of L1, SSIM, and gradient
  - Suggested: 0.4×L1 + 0.5×SSIM + 0.1×gradient
- Dataset: Same v4 linear dataset

**Difference Between Scripts:**
- `train_v4_ssim.py` - Initial version
- `train_v4_ssim_fixed.py` - Fixed version (presumably bug fixes)
- **TODO:** Document what was fixed

**Questions to Answer:**
1. Does SSIM loss improve texture sharpness?
2. Does it introduce artifacts?
3. What's the optimal weight balance?
4. Does it maintain HDR capability?

**Status:** Waiting to be run. Empty checkpoint directory.

**Logs:** [logs/training_ssim.log](../logs/training_ssim.log) (minimal output)

---

### Full-Resolution SSIM Training ⚠️ WIP

**Date:** 2026-02-03
**Script:** [scripts/train/experiments/train_fullres_ssim.py](../scripts/train/experiments/train_fullres_ssim.py)
**Status:** ⚠️ **Experimental**

**Approach:**
- Use full-resolution JPEGs (not downsampled RAFs)
- Convert sRGB to linear, simulate CFA
- SSIM + gradient loss for texture preservation
- **Note:** No flip augmentation (to avoid CFA artifacts)

**Rationale:**
- Full resolution might help model learn finer texture details
- SSIM loss explicitly optimizes for perceptual quality
- Avoid artifacts from flip augmentation

**Concerns:**
1. Training on synthetic CFA (from JPEGs) vs real CFA - domain gap?
2. JPEG artifacts in training data
3. Computational cost of full-resolution training

**Status:** Code written, not extensively tested.

---

## Historical Experiments (Archived)

### v3 - sRGB Space Training

**Date:** Prior to 2026-02-02
**Script:** [_obsolete/train_v3.py](../_obsolete/train_v3.py)
**Checkpoint:** [checkpoints/archive/v3_20260202_39.1dB_ep79.pt](../checkpoints/archive/v3_20260202_39.1dB_ep79.pt)
**Status:** Superseded by v4

**Results:** 39.1 dB PSNR

**Problem:**
Training in sRGB space with gamma correction. This clips highlights and loses HDR capability.

**Why v4 is Better:**
- Linear space preserves HDR (values > 1.0)
- No gamma-related artifacts
- More physically accurate

---

## Open Research Questions

### 1. Understanding v4_ft Artifacts

**Question:** What actually caused the 6×6 grid pattern in v4_ft?

**Hypotheses:**
- ❓ Flip augmentation breaking CFA alignment (less likely)
- ❓ Different data distribution (synthetic vs real CFA)
- ❓ Training hyperparameters
- ❓ Learning rate too high
- ❓ Something in synthetic CFA generation

**Proposed Investigation:**
1. Save intermediate outputs during training
2. Train without augmentation as control
3. Compare synthetic CFA vs real CFA characteristics
4. Try different learning rates (1e-5, 1e-6)

---

### 2. Texture Preservation

**Question:** How to improve texture sharpness without losing stability?

**Approaches to Try:**
- ✓ SSIM loss (experiment ready, not run)
- 🔍 Full-resolution training (code ready)
- 🔍 Different augmentation strategies
- 🔍 Adversarial loss component
- 🔍 Curriculum learning (coarse → fine)

**Metrics:**
- PSNR (current: 48.2 dB)
- SSIM
- Visual texture scores
- Laplacian variance (sharpness)

---

### 3. Fine-Tuning Without Forgetting

**Question:** What's the right way to improve v4 without catastrophic forgetting?

**Approaches to Try:**
- Mixed dataset (90% real RAFs + 10% synthetic/difficult cases)
- Elastic weight consolidation (EWC)
- Lower learning rate with longer training
- Freeze early layers, fine-tune decoder only
- Progressive fine-tuning (gradually increase synthetic ratio)

---

### 4. Inference Optimization

**Question:** Can we speed up inference without quality loss?

**Ideas:**
- Smaller model (knowledge distillation)
- Quantization (INT8, INT16)
- TensorRT or CoreML optimization
- Adaptive patch sizing (large patches for smooth areas)

---

## Experiment Naming Convention

Going forward, use this naming scheme:

```
v{major}.{minor}_{variant}_{status}

Examples:
- v5.0_ssim_wip         (SSIM loss, in progress)
- v5.1_fullres_failed   (Full-res, failed)
- v6.0_mixed_success    (Mixed dataset, succeeded)
```

**Checkpoint directories:**
```
checkpoints/experiments/{experiment_name}/
```

**Log files:**
```
logs/train_{experiment_name}.log
```

---

## Next Experiments to Run

### Priority 1: Run v4_ssim

**Goal:** Determine if SSIM loss improves texture without artifacts

**Steps:**
1. Decide: `train_v4_ssim.py` vs `train_v4_ssim_fixed.py`
2. Start from v4 checkpoint
3. Low learning rate (1e-5)
4. Short run (20-30 epochs) for quick evaluation
5. Compare outputs visually + PSNR/SSIM metrics

**Expected Time:** ~4-6 hours on MPS

---

### Priority 2: Investigate v4_ft Artifacts

**Goal:** Understand root cause of 6×6 grid pattern

**Steps:**
1. Visualize CFA patterns from synthetic vs real data
2. Train without augmentation
3. Train with very low LR (1e-6)
4. Save outputs every N batches to see when artifacts appear

**Expected Time:** ~2-3 iterations × 2 hours each

---

### Priority 3: Mixed Dataset Fine-tune

**Goal:** Improve on difficult cases without catastrophic forgetting

**Approach:**
- 80% real RAFs (original v4 dataset)
- 15% high-sharpness RAFs (from top200_sharp.json)
- 5% torture patterns (carefully selected, not all 13 types)

**Expected Time:** ~8-12 hours

---

## Metrics Dashboard

| Experiment | PSNR (dB) | SSIM | Texture Score | HDR Support | Artifacts | Status |
|------------|-----------|------|---------------|-------------|-----------|--------|
| v3         | 39.1      | ?    | ?             | No          | None      | Archived |
| **v4**     | **48.2**  | ?    | Medium        | Yes         | None      | ✓ Production |
| v4_ft      | ?         | ?    | ?             | ?           | 6×6 grid  | ✗ Broken |
| torture_ft | ?         | ?    | ?             | ?           | Pixel mush | ✗ Broken |
| v4_ssim    | -         | -    | -             | -           | -         | Not run |

---

**Last Updated:** 2026-02-03
