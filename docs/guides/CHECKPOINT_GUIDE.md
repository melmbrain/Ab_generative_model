# Checkpoint System Guide

**Status**: ✅ Fully implemented and tested
**Date**: 2025-10-31

---

## 🎯 Overview

The training system now automatically saves checkpoints **every epoch** to protect against data loss during long training runs (which can take several hours).

---

## 💾 Checkpoint Files Created

When you train with `--name my_experiment`, the system creates:

### 1. **Per-Epoch Checkpoints** (every epoch)
```
checkpoints/my_experiment_epoch1.pt
checkpoints/my_experiment_epoch2.pt
checkpoints/my_experiment_epoch3.pt
...
checkpoints/my_experiment_epoch20.pt
```
- Saved after **every epoch**
- Contains full model state, optimizer, scheduler
- Can resume from any epoch

### 2. **Latest Checkpoint** (updated every epoch)
```
checkpoints/my_experiment_latest.pt
```
- Always contains the **most recent** epoch
- Convenient for resuming after crashes
- Gets overwritten each epoch

### 3. **Best Model Checkpoint** (when validation improves)
```
checkpoints/my_experiment_best.pt
```
- Saved only when validation loss **improves**
- This is the model you'll want to use for production
- Contains the best performing epoch

---

## 📊 What's Saved in Each Checkpoint

Each checkpoint file contains:

```python
{
    'epoch': 5,                          # Epoch number
    'model_state_dict': {...},           # Model weights
    'optimizer_state_dict': {...},       # Optimizer state (Adam)
    'scheduler_state_dict': {...},       # LR scheduler state
    'best_val_loss': 2.1234,             # Best validation loss so far
    'val_loss': 2.3456,                  # Current validation loss
    'config': {                          # Model configuration
        'vocab_size': 25,
        'd_model': 256
    }
}
```

This allows you to resume training **exactly** where you left off, including:
- Model weights
- Optimizer momentum
- Learning rate schedule position
- Best validation loss tracking

---

## 🔄 How to Resume Training

If training crashes or you stop it early, resume with:

### Resume from Latest Checkpoint
```bash
python3 train.py \
  --config small \
  --batch-size 32 \
  --epochs 20 \
  --device cuda \
  --name my_experiment \
  --resume-from checkpoints/my_experiment_latest.pt
```

### Resume from Specific Epoch
```bash
python3 train.py \
  --config small \
  --batch-size 32 \
  --epochs 20 \
  --device cuda \
  --name my_experiment \
  --resume-from checkpoints/my_experiment_epoch10.pt
```

**What happens when you resume:**
1. ✅ Loads model weights from checkpoint
2. ✅ Restores optimizer state (momentum, etc.)
3. ✅ Restores LR scheduler position
4. ✅ Continues from next epoch (e.g., epoch 11 if resuming from epoch 10)
5. ✅ Preserves best validation loss tracking

---

## 📝 Training Output

### Normal Training (no resume)
```
Epoch 1
--------------------------------------------------
  Step 0: loss=3.4715

Epoch 1 Summary:
  Train Loss: 3.4715
  Val Loss:   3.0185
  Validity:   100.0%
  Diversity:  10.0%
✅ New best validation loss: 3.0185
💾 Checkpoint saved: checkpoints/my_experiment_epoch1.pt
⭐ Best model saved: checkpoints/my_experiment_best.pt
```

### With Resume
```
📁 Resuming from checkpoint: checkpoints/my_experiment_epoch5.pt
Loading checkpoint: checkpoints/my_experiment_epoch5.pt
✅ Resumed from epoch 5, val_loss=2.3456
   Will continue from epoch 6

5. Starting training...
======================================================================
Starting Training: my_experiment
======================================================================
Device: cuda
Model parameters: 5,623,449
Resuming from epoch: 5
======================================================================

Epoch 6
--------------------------------------------------
...
```

---

## 🎯 Typical Workflow

### Full Training Run (20 epochs)
```bash
# Start training
python3 train.py \
  --config small \
  --batch-size 32 \
  --epochs 20 \
  --device cuda \
  --eval-interval 2 \
  --name production_small_v1

# After completion, checkpoints exist:
# - checkpoints/production_small_v1_epoch1.pt
# - checkpoints/production_small_v1_epoch2.pt
# - ...
# - checkpoints/production_small_v1_epoch20.pt
# - checkpoints/production_small_v1_latest.pt
# - checkpoints/production_small_v1_best.pt  <-- USE THIS ONE!
```

### If Training Crashes at Epoch 12
```bash
# Check which checkpoint to resume from
ls -lh checkpoints/production_small_v1_*.pt

# Resume from latest
python3 train.py \
  --config small \
  --batch-size 32 \
  --epochs 20 \
  --device cuda \
  --eval-interval 2 \
  --name production_small_v1 \
  --resume-from checkpoints/production_small_v1_latest.pt

# Will continue from epoch 13 → 20
```

---

## 💡 Use Cases

### 1. **Training Interrupted**
Your computer crashes at epoch 15/20:
```bash
# Resume from where you left off
--resume-from checkpoints/my_experiment_latest.pt
```
Lost work: **0 epochs** ✅

### 2. **Experiment with More Epochs**
You trained for 20 epochs, want to try 30:
```bash
# Resume from epoch 20, train to 30
python3 train.py --epochs 30 --resume-from checkpoints/my_experiment_epoch20.pt
```

### 3. **Learning Rate Adjustment**
Model still improving, want to fine-tune with lower LR:
```bash
# Resume from best model, continue with lower LR
python3 train.py --lr 1e-5 --resume-from checkpoints/my_experiment_best.pt
```

### 4. **GPU Availability**
Started on CPU, GPU became available:
```bash
# Resume on GPU from CPU checkpoint
python3 train.py --device cuda --resume-from checkpoints/my_experiment_latest.pt
```

---

## 📏 Disk Space Usage

### Checkpoint Sizes

| Config | Parameters | Checkpoint Size | 20 Epochs Total |
|--------|------------|-----------------|-----------------|
| tiny   | 0.95M      | ~12 MB          | ~240 MB         |
| small  | 5.6M       | ~66 MB          | ~1.3 GB         |
| medium | 44M        | ~528 MB         | ~10.6 GB        |
| large  | 100M+      | ~1.2 GB         | ~24 GB          |

**Plus**: `_latest.pt` and `_best.pt` (2 extra copies)

### Managing Disk Space

If disk space is limited, you can:

1. **Delete old epoch checkpoints** (keep best + latest):
```bash
# Keep only best, latest, and every 5th epoch
rm checkpoints/my_experiment_epoch[1-4].pt
rm checkpoints/my_experiment_epoch[6-9].pt
# etc.
```

2. **Train with fewer checkpoints** (modify code to save every N epochs)

3. **Use smaller model** (tiny/small instead of medium/large)

---

## 🔍 Inspecting Checkpoints

### Check Checkpoint Contents
```python
import torch

# Load checkpoint
checkpoint = torch.load('checkpoints/my_experiment_epoch10.pt')

# Check what's inside
print(f"Epoch: {checkpoint['epoch']}")
print(f"Val Loss: {checkpoint['val_loss']:.4f}")
print(f"Best Val Loss: {checkpoint['best_val_loss']:.4f}")
print(f"Config: {checkpoint['config']}")

# Load just the model weights (for inference)
model.load_state_dict(checkpoint['model_state_dict'])
```

### Find Best Epoch
```bash
# List checkpoints by modification time
ls -lht checkpoints/my_experiment_*.pt | head

# The "best.pt" file always contains the best model
```

---

## ⚠️ Important Notes

### 1. **Checkpoint Compatibility**
- Checkpoints are tied to model architecture
- Cannot load `small` checkpoint into `medium` model
- Cannot load different vocab_size checkpoint

### 2. **Resuming Requirements**
When resuming, you **must** use the **same**:
- `--config` (model architecture)
- `--batch-size` can change
- `--device` can change (CPU → GPU or vice versa)
- `--lr` will be ignored (uses scheduler from checkpoint)
- `--name` should match (but not required)

### 3. **epoch parameter**
When using `--resume-from`, the `--epochs` parameter means **total epochs**, not additional epochs:
```bash
# If you resume from epoch 10:
--epochs 20  # Will train epochs 11-20 (10 more)
--epochs 15  # Will train epochs 11-15 (5 more)
--epochs 10  # Will not train (already at epoch 10)
```

### 4. **Overwriting Checkpoints**
If you resume with the same `--name`, existing checkpoints will be overwritten:
```bash
# This will overwrite epoch11.pt, epoch12.pt, etc.
python3 train.py --name my_experiment --resume-from checkpoints/my_experiment_epoch10.pt
```

---

## ✅ Testing

Checkpoint system has been tested and verified:

```bash
# Test 1: Save checkpoints every epoch ✅
python3 train.py --config tiny --epochs 3 --name test

# Verify: epoch1.pt, epoch2.pt, epoch3.pt, latest.pt, best.pt created ✅

# Test 2: Resume from checkpoint ✅
python3 train.py --epochs 5 --name test --resume-from checkpoints/test_epoch2.pt

# Verify: Resumed from epoch 2, continued to epoch 5 ✅
```

---

## 🎯 Best Practices

### 1. **For Long Training Runs** (hours)
Always use meaningful experiment names:
```bash
--name production_small_2025_10_31
```

### 2. **Monitor Disk Space**
Check available space before training:
```bash
df -h checkpoints/
```

### 3. **Backup Important Checkpoints**
After training completes, backup the best model:
```bash
cp checkpoints/my_experiment_best.pt backups/
```

### 4. **Clean Up After Success**
Once you have your best model, delete intermediate checkpoints:
```bash
# Keep only best and latest
rm checkpoints/my_experiment_epoch*.pt
```

### 5. **Use Latest for Crashes**
If training crashes, always resume from `_latest.pt`:
```bash
--resume-from checkpoints/my_experiment_latest.pt
```

---

## 📚 Examples

### Example 1: Full Training
```bash
# Start training (will run for ~3 hours)
python3 train.py \
  --config small \
  --batch-size 32 \
  --epochs 20 \
  --device cuda \
  --eval-interval 2 \
  --name antibody_gen_v1

# Result after completion:
# checkpoints/antibody_gen_v1_best.pt      <-- USE THIS
# checkpoints/antibody_gen_v1_epoch*.pt    (1-20)
# checkpoints/antibody_gen_v1_latest.pt
```

### Example 2: Resume After Crash
```bash
# Training crashed at epoch 14
# Resume from latest:
python3 train.py \
  --config small \
  --batch-size 32 \
  --epochs 20 \
  --device cuda \
  --eval-interval 2 \
  --name antibody_gen_v1 \
  --resume-from checkpoints/antibody_gen_v1_latest.pt

# Will continue: epochs 15-20
```

### Example 3: Extended Training
```bash
# Trained for 20 epochs, want to try 30
python3 train.py \
  --config small \
  --batch-size 32 \
  --epochs 30 \
  --device cuda \
  --eval-interval 2 \
  --name antibody_gen_v1 \
  --resume-from checkpoints/antibody_gen_v1_epoch20.pt

# Will continue: epochs 21-30
```

---

## 🚀 Summary

✅ **Automatic checkpoint saving** every epoch
✅ **Three types** of checkpoints: per-epoch, latest, best
✅ **Full state saved**: model, optimizer, scheduler, metrics
✅ **Easy resuming** with `--resume-from` flag
✅ **Protection** against data loss during long runs
✅ **Tested and verified** working correctly

**You're protected!** Even if training crashes at epoch 19/20, you can resume from epoch 19 and only lose <20 minutes of work instead of 6 hours.

---

Last Updated: 2025-10-31
