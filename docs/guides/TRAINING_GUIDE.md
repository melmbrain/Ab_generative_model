# Training Guide

## Quick Start

### 1. Test Run (2-3 minutes)
Train on a small subset to verify everything works:
```bash
python3 train.py \
    --config tiny \
    --epochs 2 \
    --max-samples 100 \
    --batch-size 4 \
    --name test_run
```

### 2. Small Training Run (30-60 minutes)
Train on 10k samples with tiny model:
```bash
python3 train.py \
    --config tiny \
    --epochs 10 \
    --max-samples 10000 \
    --batch-size 16 \
    --name tiny_10k
```

### 3. Full Training Run (Hours to Days)
Train on full dataset:
```bash
# Small model (recommended for CPU)
python3 train.py \
    --config small \
    --epochs 20 \
    --batch-size 16 \
    --name small_full

# Medium model (requires GPU)
python3 train.py \
    --config medium \
    --epochs 20 \
    --batch-size 32 \
    --device cuda \
    --name medium_full
```

---

## Command Line Arguments

### Model Configuration
- `--config`: Model size (`tiny`, `small`, `medium`, `large`)
  - `tiny`: 0.95M params - Fast, good for testing
  - `small`: 5.6M params - Good balance for CPU
  - `medium`: ~40M params - Better quality, needs GPU
  - `large`: ~100M params - Best quality, needs powerful GPU

### Data
- `--train-data`: Path to training JSON (default: `data/generative/train.json`)
- `--val-data`: Path to validation JSON (default: `data/generative/val.json`)
- `--batch-size`: Batch size (default: 16)
- `--max-samples`: Limit training samples for testing (optional)

### Training
- `--epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 1e-4)
- `--device`: Device to use (`cpu` or `cuda`, auto-detected)
- `--eval-interval`: Evaluate every N epochs (default: 1)
- `--early-stopping`: Stop if no improvement for N epochs (default: 5)

### Experiment
- `--name`: Experiment name (default: auto-generated timestamp)

---

## Output Files

### Checkpoints
Saved to `checkpoints/` directory:
```
checkpoints/
├── my_experiment_epoch1.pt
├── my_experiment_epoch2.pt
└── my_experiment_epoch5.pt    # Best model
```

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Validation loss
- Configuration

### Logs
Saved to `logs/` directory in JSONL format:
```
logs/
├── my_experiment.jsonl        # Training logs
└── my_experiment_checkpoints.jsonl  # Checkpoint metadata
```

---

## Monitoring Training

### View Progress
```bash
python3 monitor_training.py logs/my_experiment.jsonl
```

### List All Experiments
```bash
python3 monitor_training.py list
```

### Compare Experiments
```bash
python3 monitor_training.py compare logs/exp1.jsonl logs/exp2.jsonl
```

---

## Training Process

### What Happens During Training

**Each Epoch**:
1. **Training Phase**
   - Forward pass: Model predicts antibody sequences
   - Loss computation: Cross-entropy on predicted tokens
   - Backward pass: Compute gradients
   - Optimizer step: Update model weights
   - Gradient clipping: Prevent exploding gradients

2. **Validation Phase**
   - Forward pass on validation set
   - Compute validation loss
   - No gradient updates

3. **Generation Evaluation** (every eval_interval epochs)
   - Generate 100 antibody sequences
   - Compute quality metrics:
     - Validity: % valid amino acid sequences
     - Diversity: % unique sequences
     - Length statistics
   - Log sample sequences

4. **Checkpoint Saving**
   - Save model if validation loss improved
   - Store checkpoint metadata

5. **Learning Rate Adjustment**
   - Reduce LR if validation loss plateaus
   - Patience: 2 epochs

6. **Early Stopping Check**
   - Stop if no improvement for N epochs

---

## Expected Results

### Initial Training (Epoch 1-2)
- Loss: ~3.0-4.0 (cross-entropy)
- Validity: 100% (all valid amino acids)
- Diversity: Low (40-50%, lots of repetition)
- Sequences: Repetitive patterns (e.g., "GGGGGGGG...")

### Mid Training (Epoch 5-10)
- Loss: ~2.0-2.5
- Validity: 100%
- Diversity: Improving (60-70%)
- Sequences: More varied, some structure

### Well-Trained (Epoch 15-20)
- Loss: ~1.5-2.0
- Validity: 98-100%
- Diversity: High (70-80%)
- Sequences: Realistic antibody-like sequences

### Signs of Good Training
✅ Validation loss decreasing
✅ High validity (>95%)
✅ Increasing diversity (>70%)
✅ Sequences look like antibodies (varied amino acids)
✅ Train/val loss gap is reasonable (<0.5)

### Signs of Problems
❌ Validation loss increasing (overfitting)
❌ Low diversity (<50%, mode collapse)
❌ Repetitive sequences (GGGG..., AAAA...)
❌ Large train/val gap (>1.0, overfitting)
❌ Loss stuck/not decreasing (learning rate too high/low)

---

## Training Strategy

### 1. Start Small
```bash
# Quick test (5 min)
python3 train.py --config tiny --epochs 2 --max-samples 100

# Verify everything works before full training
```

### 2. Iterate on Subset
```bash
# Train on 10k samples (1 hour)
python3 train.py --config small --epochs 10 --max-samples 10000

# Tune hyperparameters here
```

### 3. Full Training
```bash
# Full dataset with tuned hyperparameters
python3 train.py --config medium --epochs 20 --batch-size 32 --device cuda
```

### 4. Monitor and Adjust
- Watch logs with `monitor_training.py`
- Check sample generations
- Adjust learning rate if needed
- Use early stopping to prevent overfitting

---

## Hyperparameter Tuning

### Learning Rate
- Default: `1e-4` (good starting point)
- Too high: Loss explodes or oscillates
- Too low: Training very slow
- Adjust: `--lr 5e-5` or `--lr 2e-4`

### Batch Size
- Larger = faster, more memory, less noise
- Smaller = slower, less memory, more noise
- CPU: 8-16
- GPU: 32-64 (depends on VRAM)

### Model Size
- Start with `small` on CPU
- Use `medium` or `large` on GPU
- Bigger = better quality but slower

### Epochs
- Start with 10-20 epochs
- Use early stopping (patience=5)
- Best model may be in middle of training

---

## Computational Requirements

### CPU Training
- **Tiny model**: ~1 hour for 10k samples
- **Small model**: ~3-5 hours for 10k samples
- **Full dataset (127k)**: Days to weeks

### GPU Training (16GB VRAM)
- **Small model**: ~30 min for 10k samples
- **Medium model**: ~1-2 hours for 10k samples
- **Full dataset (127k)**: 4-8 hours

### Recommendations
- **Testing**: CPU with tiny model (100-1000 samples)
- **Development**: CPU/GPU with small model (10k samples)
- **Production**: GPU with medium/large model (full dataset)

---

## Resuming Training

### Load Checkpoint
```python
import torch
from generators.transformer_seq2seq import create_model

# Create model
model = create_model('small', vocab_size=25)

# Load checkpoint
checkpoint = torch.load('checkpoints/my_experiment_epoch5.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Continue training or use for generation
```

---

## Example Training Session

```bash
# 1. Start training
python3 train.py \
    --config small \
    --epochs 20 \
    --batch-size 16 \
    --name antibody_v1 \
    --early-stopping 5

# 2. Monitor in another terminal
python3 monitor_training.py logs/antibody_v1.jsonl

# 3. View all experiments
python3 monitor_training.py list

# 4. After training, check best checkpoint
# Look for: checkpoints/antibody_v1_epoch*.pt
# Use the epoch with lowest validation loss
```

---

## Troubleshooting

### Out of Memory
- Reduce `--batch-size`
- Use smaller model (`--config tiny`)
- Limit samples (`--max-samples 1000`)

### Training Too Slow
- Increase `--batch-size` (if memory allows)
- Use GPU (`--device cuda`)
- Reduce `--eval-interval 5` (evaluate less often)

### Loss Not Decreasing
- Check learning rate (try `--lr 5e-5` or `--lr 2e-4`)
- Verify data is loading correctly
- Check for NaN in loss (gradient explosion)

### Low Diversity
- Train longer (more epochs)
- Increase model size
- Check training data diversity

### Overfitting (train loss << val loss)
- Reduce model size
- Add more training data
- Use early stopping
- Check data augmentation

---

## Next Steps After Training

1. **Evaluate Best Model**
   - Generate 1000 sequences
   - Compute comprehensive metrics
   - Compare to baseline

2. **Test on Specific Antigens**
   - Generate antibodies for target antigens
   - Vary target pKd (5.0, 7.0, 9.0)
   - Analyze generated sequences

3. **Integration with Discriminator**
   - Score generated antibodies
   - Filter by predicted affinity
   - Rank candidates

4. **Production Deployment**
   - Create inference API
   - Optimize generation speed
   - Add validation checks

---

## Status

✅ **Training Script Complete**
- Full training loop implemented
- Validation and evaluation working
- Checkpoint saving functional
- Logging integrated
- Tested on subset

🚀 **Ready for Training!**

---

**Last Updated**: 2025-10-31
