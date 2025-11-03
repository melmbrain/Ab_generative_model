# Metrics and Monitoring Guide

## Overview

Comprehensive metrics system for tracking antibody generation model training.

---

## Metrics Module

**File**: `generators/metrics.py`

### 1. SequenceMetrics

Evaluates quality of generated antibody sequences.

**Features**:
- **Sequence Validity**: Checks if sequences contain only valid amino acids
- **Diversity**: Measures uniqueness of generated sequences
- **Length Statistics**: Mean, min, max sequence lengths
- **Amino Acid Distribution**: Frequency analysis of amino acids

**Usage**:
```python
from generators.tokenizer import AminoAcidTokenizer
from generators.metrics import SequenceMetrics

tokenizer = AminoAcidTokenizer()
metrics = SequenceMetrics(tokenizer)

# Evaluate sequences
sequences = ["ACDEFGH", "IKLMNPQ", "RSTVWY"]
results = metrics.evaluate_batch(sequences)

print(f"Validity: {results['validity']:.1f}%")
print(f"Diversity: {results['diversity']['unique_ratio']:.1f}%")
```

### 2. TrainingLogger

Logs and tracks training progress.

**Features**:
- Logs training/validation loss per epoch
- Tracks sequence quality metrics
- Saves logs to JSONL format
- Generates training plots (with matplotlib)
- Identifies best epoch by metric

**Usage**:
```python
from generators.metrics import TrainingLogger

logger = TrainingLogger(log_dir='logs', experiment_name='transformer_v1')

# During training
for epoch in range(num_epochs):
    logger.log_epoch_start(epoch)

    # Training loop
    for step, batch in enumerate(train_loader):
        loss = train_step(batch)
        logger.log_train_step(step, loss, batch_size)

    # Validation
    val_loss = validate()
    val_metrics = evaluate_generation()

    logger.log_epoch_end(epoch, train_loss, val_loss, val_metrics)

# Save checkpoint
logger.save_checkpoint_info(epoch, model_path, metrics)

# Print summary
logger.print_summary()
```

---

## Monitoring Tool

**File**: `monitor_training.py`

Real-time training progress viewer.

### Commands

**1. List all experiments**:
```bash
python monitor_training.py list
```

**2. View experiment progress**:
```bash
python monitor_training.py logs/transformer_v1.jsonl
```

**3. Compare experiments**:
```bash
python monitor_training.py compare logs/exp1.jsonl logs/exp2.jsonl
```

### Example Output

```
======================================================================
Training Progress
======================================================================

Total Epochs: 10

Latest Epoch (10):
  Train Loss: 1.2345
  Val Loss:   1.3456
  Validity:   95.0%
  Diversity:  80.0%

Best Validation Loss: 1.2000 (Epoch 7)

Recent History (last 5 epochs):
  Epoch | Train Loss | Val Loss  | Time
  ------|------------|-----------|-------
      6 |     1.2800 |    1.3200 | 120.5s
      7 |     1.2400 |    1.2000 | 118.3s
      8 |     1.2600 |    1.3100 | 121.2s
      9 |     1.2500 |    1.2900 | 119.8s
     10 |     1.2345 |    1.3456 | 122.1s

Total Training Time: 3.35 hours
======================================================================
```

---

## Tracked Metrics

### Training Metrics

| Metric | Description | Goal |
|--------|-------------|------|
| Train Loss | Cross-entropy loss on training set | Minimize |
| Val Loss | Cross-entropy loss on validation set | Minimize |
| Epoch Time | Time per epoch (seconds) | Monitor |

### Generation Quality Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Validity | % of sequences with valid amino acids only | > 95% |
| Diversity | % of unique sequences generated | > 70% |
| Avg Length | Mean sequence length | ~120 AA (heavy), ~200 AA (light) |
| AA Distribution | Amino acid frequency | Similar to training data |

### Advanced Metrics (Future)

- **Affinity Correlation**: Correlation between target and predicted pKd
- **Structure Quality**: Secondary structure prediction scores
- **CDR Similarity**: Similarity to known CDR sequences
- **Developability**: Aggregation, immunogenicity, stability scores

---

## Log File Format

Logs are saved in JSONL (JSON Lines) format:

```json
{"epoch": 1, "train_loss": 3.2, "val_loss": 3.4, "epoch_time": 120.5, "timestamp": 1698765432.1, "val_metrics": {"validity": 95.0, "diversity": {"unique_ratio": 80.0}}}
{"epoch": 2, "train_loss": 2.8, "val_loss": 3.0, "epoch_time": 118.2, "timestamp": 1698765552.3, "val_metrics": {"validity": 96.5, "diversity": {"unique_ratio": 82.0}}}
```

Each line is a valid JSON object representing one epoch.

---

## Visualization

The logger can generate matplotlib plots showing:
- Training/validation loss curves
- Validity over time
- Diversity over time
- Epoch duration trends

**Generate plots**:
```python
logger.plot_metrics(save_path='logs/training_curves.png')
```

---

## Best Practices

### 1. Monitor Validation Loss
- Use early stopping if val loss stops improving
- Best epoch may not be the last epoch

### 2. Check Sequence Quality
- Validity should be high (> 95%)
- Diversity should stay high (> 70%)
- Low diversity = mode collapse

### 3. Generation Samples
- Inspect generated sequences each epoch
- Look for patterns, repetitions, or artifacts
- Compare to real antibody sequences

### 4. Checkpoint Strategy
- Save model at best validation loss
- Keep last N checkpoints
- Save checkpoint info with metrics

---

## Integration with Training

The metrics system integrates seamlessly with training scripts:

```python
# Setup
logger = TrainingLogger(log_dir='logs', experiment_name='my_exp')
metrics_eval = SequenceMetrics(tokenizer)

# Training loop
for epoch in range(num_epochs):
    logger.log_epoch_start(epoch)

    # Train
    train_loss = train_epoch(model, train_loader)

    # Validate
    val_loss = validate_epoch(model, val_loader)

    # Generate samples and evaluate
    generated_seqs = generate_samples(model, val_loader, n=100)
    val_metrics = metrics_eval.evaluate_batch(generated_seqs)

    # Log
    logger.log_epoch_end(epoch, train_loss, val_loss, val_metrics)

    # Save checkpoint if best
    if val_loss < best_val_loss:
        save_checkpoint(model, f'checkpoints/model_epoch{epoch}.pt')
        logger.save_checkpoint_info(epoch, checkpoint_path, val_metrics)

logger.print_summary()
```

---

## Files Created

```
generators/
├── metrics.py           # Metrics and logging module

logs/                    # Training logs directory
├── *.jsonl             # Training logs (JSONL format)
└── *.png               # Training plots (optional)

monitor_training.py      # Monitoring tool
METRICS_GUIDE.md        # This guide
```

---

## Status

✅ **Implemented**:
- Sequence validation
- Diversity metrics
- Length statistics
- AA distribution
- Training logger
- Log file system
- Monitoring tool
- Progress display

📋 **Ready for Training**:
All metrics infrastructure is in place and tested. Ready to track training progress!

---

**Last Updated**: 2025-10-31
