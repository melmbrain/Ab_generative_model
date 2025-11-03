# Computational Optimization Strategy

**Goal**: Train generative model on 158k samples efficiently
**Challenge**: Limited computational resources, long training times
**Solution**: Multi-tier optimization strategy

---

## 🎯 Overview: Progressive Training Approach

Instead of training a large model on full dataset immediately, use a **staged approach**:

```
Stage 1: Small model + Small data (1k samples)     → 10 min training
Stage 2: Medium model + Medium data (10k samples)  → 1-2 hours training
Stage 3: Full model + Full data (158k samples)     → 10-20 hours training
```

**Benefits**:
- ✅ Quick iteration and debugging (Stage 1)
- ✅ Validate approach before committing compute (Stage 2)
- ✅ Only scale up if results are promising (Stage 3)

---

## 📊 Optimization Strategy Matrix

| Optimization | Speed Gain | Complexity | Priority |
|--------------|------------|------------|----------|
| **1. Smaller Model** | 10-50x | Low | 🔥 High |
| **2. Progressive Data** | 5-20x | Low | 🔥 High |
| **3. Cached Embeddings** | 2-5x | Medium | 🔥 High |
| **4. Efficient Architecture** | 2-4x | Medium | ⭐ Medium |
| **5. Mixed Precision** | 2-3x | Low | ⭐ Medium |
| **6. Gradient Accumulation** | 1.5-2x | Low | ⭐ Medium |
| **7. Optimized Data Loading** | 1.2-1.5x | Low | ⭐ Medium |
| **8. Transfer Learning** | 5-10x | High | 💡 Future |
| **9. Distillation** | 3-5x | High | 💡 Future |
| **10. Cloud/Distributed** | 10-100x | High | 💡 If needed |

---

## 🚀 Strategy 1: Smaller Model First (10-50x speedup)

### Problem
- Full Transformer: ~50-100M parameters
- Training time: 20+ hours on single GPU
- Debugging is slow

### Solution: Start Small
**Tiny Model** (Stage 1):
```python
config_tiny = {
    'd_model': 128,           # vs 512 (16x smaller)
    'nhead': 4,               # vs 8
    'num_encoder_layers': 2,  # vs 6
    'num_decoder_layers': 2,  # vs 6
    'dim_feedforward': 512,   # vs 2048
    'dropout': 0.1
}
# Parameters: ~2-5M (vs 50-100M)
# Training time: 10 minutes (vs 20 hours)
# Memory: ~1GB (vs 8GB+)
```

**Small Model** (Stage 2):
```python
config_small = {
    'd_model': 256,           # vs 512
    'nhead': 4,               # vs 8
    'num_encoder_layers': 3,  # vs 6
    'num_decoder_layers': 3,  # vs 6
    'dim_feedforward': 1024,  # vs 2048
    'dropout': 0.1
}
# Parameters: ~10-20M
# Training time: 1-2 hours
# Memory: ~2-4GB
```

**Full Model** (Stage 3 - only if needed):
```python
config_full = {
    'd_model': 512,
    'nhead': 8,
    'num_encoder_layers': 6,
    'num_decoder_layers': 6,
    'dim_feedforward': 2048,
    'dropout': 0.1
}
```

### Implementation
```python
# Easy switching between configs
MODEL_CONFIGS = {
    'tiny': config_tiny,
    'small': config_small,
    'full': config_full
}

# Start with tiny
model = Seq2SeqGenerator(config=MODEL_CONFIGS['tiny'])
```

---

## 📦 Strategy 2: Progressive Data Loading (5-20x speedup)

### Problem
- Training on 158k samples takes forever
- Can't validate approach quickly

### Solution: Staged Training

**Stage 1: Proof of Concept (1k samples)**
```python
# Use first 1k samples
train_subset = train_data[:1000]
val_subset = val_data[:100]

# Train for 5-10 epochs
# Time: 10 minutes
# Goal: Verify model trains, loss decreases
```

**Stage 2: Validation (10k samples)**
```python
# Use 10k samples
train_subset = train_data[:10000]
val_subset = val_data[:1000]

# Train for 20 epochs
# Time: 1-2 hours
# Goal: Check quality of generated sequences
```

**Stage 3: Full Training (158k samples)**
```python
# Only proceed if Stage 2 shows promise
train_data_full = train_data  # All 127k
val_data_full = val_data       # All 16k

# Train for 50 epochs
# Time: 10-20 hours
# Goal: Production model
```

### Data Selection Strategy
Don't use random samples - use **high-quality samples**:

```python
# Filter for high-affinity binders (better signal)
high_quality = [s for s in train_data if s['pKd'] > 7.0]

# Use these for initial training
train_subset = high_quality[:1000]
```

---

## 💾 Strategy 3: Precompute Embeddings (2-5x speedup)

### Problem
- Tokenization + embedding happens every epoch
- Wastes compute repeating same work

### Solution: Cache Embeddings

**Approach 1: Simple Caching**
```python
# Preprocess all data once
preprocessed_data = []
for sample in train_data:
    preprocessed = {
        'antigen_tokens': tokenizer.encode(sample['antigen_sequence']),
        'antibody_tokens': tokenizer.encode_pair(
            sample['antibody_heavy'],
            sample['antibody_light']
        ),
        'pKd': sample['pKd']
    }
    preprocessed_data.append(preprocessed)

# Save to disk
with open('data/generative/train_preprocessed.pkl', 'wb') as f:
    pickle.dump(preprocessed_data, f)

# Load once, use for all epochs
# Speedup: 2-3x
```

**Approach 2: ESM-2 Embeddings (if using ESM-2)**
```python
# If we use ESM-2 embeddings instead of learned embeddings
# Precompute ALL ESM-2 embeddings once

from transformers import EsmModel, EsmTokenizer

esm_model = EsmModel.from_pretrained("facebook/esm2_t12_35M_UR50D")

# Precompute for all sequences
embeddings_cache = {}
for sample in train_data:
    # Only compute once per unique sequence
    if sample['antigen_sequence'] not in embeddings_cache:
        emb = get_esm_embedding(sample['antigen_sequence'])
        embeddings_cache[sample['antigen_sequence']] = emb

# Save cache
# Speedup: 3-5x (ESM-2 is expensive)
```

---

## 🏗️ Strategy 4: Efficient Architecture (2-4x speedup)

### Option A: LSTM Instead of Transformer (Simpler)

**Why?**
- LSTMs are 3-5x faster than Transformers
- Still effective for sequences
- Good baseline

```python
class LSTMSeq2Seq:
    """
    Faster alternative to Transformer
    """
    def __init__(self):
        self.encoder = nn.LSTM(
            input_size=256,
            hidden_size=512,
            num_layers=2,
            bidirectional=True
        )

        self.decoder = nn.LSTM(
            input_size=256,
            hidden_size=512,
            num_layers=2
        )

# Speedup: 3-4x vs Transformer
# Performance: Slightly worse but acceptable
```

**Recommendation**: Start with LSTM, upgrade to Transformer if needed

### Option B: Lightweight Transformer

```python
# Use efficient attention implementations
class EfficientTransformer:
    def __init__(self):
        # Linear attention (O(N) instead of O(N^2))
        self.attention = LinearAttention()

        # Shared weights between layers
        encoder_layer = TransformerEncoderLayer(...)
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_layers=4,
            enable_nested_tensor=True  # PyTorch optimization
        )

# Speedup: 2-3x
```

---

## ⚡ Strategy 5: Training Optimizations

### A. Mixed Precision Training (2-3x speedup)

```python
# Use FP16 instead of FP32
# Requires: PyTorch with CUDA

from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    # Forward pass in FP16
    with autocast():
        output = model(batch)
        loss = criterion(output, target)

    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# Speedup: 2-3x on GPU
# Memory: 40-50% reduction
```

### B. Gradient Accumulation (Larger Effective Batch)

```python
# Simulate batch_size=128 with batch_size=32
accumulation_steps = 4  # 32 * 4 = 128

optimizer.zero_grad()
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Benefits:
# - Larger effective batch size
# - Better gradient estimates
# - Faster convergence
```

### C. Optimized Batch Size

```python
# Find optimal batch size for your hardware
# Larger = faster, but limited by memory

# Test different batch sizes:
batch_sizes = [8, 16, 32, 64, 128]

for bs in batch_sizes:
    try:
        model.train()
        batch = get_batch(size=bs)
        loss = model(batch)
        loss.backward()
        print(f"Batch size {bs}: OK")
    except RuntimeError as e:
        print(f"Batch size {bs}: Out of memory")
        break

# Use largest that fits
```

---

## 💿 Strategy 6: Data Loading Optimization

### A. Preload Data to Memory

```python
# Instead of loading from disk each time
# Load entire dataset to RAM once

class FastDataLoader:
    def __init__(self, data_path):
        # Load all data to memory
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        # Preprocess everything
        self.preprocessed = [
            self.preprocess(sample) for sample in self.data
        ]

    def __getitem__(self, idx):
        return self.preprocessed[idx]  # Already preprocessed!

# Speedup: 1.5-2x
```

### B. Use NumPy Arrays Instead of Lists

```python
# Convert tokenized data to NumPy arrays
import numpy as np

# Faster indexing and batching
antigen_tokens = np.array(antigen_tokens, dtype=np.int32)
antibody_tokens = np.array(antibody_tokens, dtype=np.int32)

# 20-30% faster batch creation
```

---

## 🎓 Strategy 7: Transfer Learning (5-10x speedup)

### Option: Use Pre-trained Protein Language Model

Instead of training from scratch, **fine-tune** existing model:

```python
# Option 1: Fine-tune ESM-2
from transformers import EsmForMaskedLM

model = EsmForMaskedLM.from_pretrained("facebook/esm2_t12_35M_UR50D")

# Add decoder for generation
decoder = TransformerDecoder(...)

# Fine-tune on your data (much faster than training from scratch)
# Speedup: 5-10x
# Performance: Often better (pre-training helps)
```

**Pros**:
- Much faster convergence
- Better performance with less data
- Leverages 500M+ proteins pre-training

**Cons**:
- More complex setup
- Larger model (need GPU)

---

## 📈 Strategy 8: Early Stopping & Checkpointing

### A. Early Stopping

```python
# Don't waste compute on overfitting
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(max_epochs):
    train_loss = train_epoch(model, train_loader)
    val_loss = validate(model, val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        save_checkpoint(model, 'best.pth')
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

# Saves hours of unnecessary training
```

### B. Checkpoint Resume

```python
# Save checkpoints every epoch
# If training crashes, resume from last checkpoint

checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')

# Resume training
checkpoint = torch.load('checkpoint_epoch_10.pth')
model.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

---

## ☁️ Strategy 9: Cloud/Distributed Training (10-100x)

### When Local Compute is Insufficient

**Option 1: Google Colab (Free GPU)**
```python
# Free Tesla T4 GPU (16GB)
# Good for Stage 2 (10k samples)
# Time limit: 12 hours

# Upload data to Google Drive
# Run training in Colab notebook
# Download trained model
```

**Option 2: AWS/GCP (Paid)**
```python
# AWS p3.2xlarge: V100 GPU (16GB) - $3/hour
# Train Stage 3 in 4-6 hours = $12-18

# AWS p3.8xlarge: 4x V100 GPUs - $12/hour
# Train Stage 3 in 1-2 hours = $12-24
```

**Option 3: Distributed Training**
```python
# If you have multiple GPUs
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Train on 4 GPUs
# Speedup: 3-4x (not linear due to communication)
```

---

## 🎯 Recommended Implementation Strategy

### Phase 1: Quick Validation (Today - 1 hour)

```python
# Configuration
model_config = 'tiny'        # 2-5M params
data_size = 1000             # 1k samples
batch_size = 32
epochs = 10

# Expected time: 10-15 minutes
# Goal: Verify everything works
```

**Checklist**:
- [ ] Model trains without errors
- [ ] Loss decreases
- [ ] Generates valid sequences
- [ ] Can overfit on small data (good sign!)

### Phase 2: Quality Check (Tomorrow - 2 hours)

```python
# Configuration
model_config = 'small'       # 10-20M params
data_size = 10000            # 10k samples (high-quality only)
batch_size = 64
epochs = 20

# Expected time: 1-2 hours
# Goal: Check quality of outputs
```

**Checklist**:
- [ ] Generated sequences look reasonable
- [ ] Validate with discriminator
- [ ] Diversity check (unique sequences)
- [ ] Affinity correlation check

### Phase 3: Full Training (If Phase 2 succeeds - 6-12 hours)

```python
# Configuration
model_config = 'full'        # 50-100M params
data_size = 158000           # Full dataset
batch_size = 64              # Or max that fits
epochs = 50
mixed_precision = True       # Use FP16
gradient_accumulation = 2

# Expected time: 10-20 hours (or 4-6 on cloud GPU)
# Goal: Production model
```

**Checklist**:
- [ ] Monitor training curves
- [ ] Save checkpoints every epoch
- [ ] Early stopping if overfitting
- [ ] Evaluate on test set

---

## 📊 Expected Timelines

### Local Training (CPU/Consumer GPU)

| Stage | Model | Data | Time | Cost |
|-------|-------|------|------|------|
| 1 - POC | Tiny | 1k | 10 min | Free |
| 2 - Validate | Small | 10k | 1-2 hours | Free |
| 3 - Full | Full | 158k | 20-40 hours | Free |

### Cloud Training (AWS p3.2xlarge - V100 GPU)

| Stage | Model | Data | Time | Cost |
|-------|-------|------|------|------|
| 1 - POC | Tiny | 1k | 2 min | $0.10 |
| 2 - Validate | Small | 10k | 20 min | $1 |
| 3 - Full | Full | 158k | 4-6 hours | $12-18 |

**Recommendation**:
- Stages 1 & 2: Run locally (free)
- Stage 3: Consider cloud if >12 hours locally

---

## 🛠️ Implementation Recommendations

### What to Build First

1. **Tiny Model + 1k Samples** ✅
   - LSTM-based (simpler than Transformer)
   - Proves concept quickly
   - Easy to debug

2. **Validation Tools** ✅
   - Generate samples
   - Score with discriminator
   - Measure diversity

3. **Progressive Training Script** ✅
   - Supports tiny/small/full configs
   - Automatic checkpointing
   - Early stopping

4. **Monitoring** ✅
   - Track loss curves
   - Sample generation each epoch
   - Validation metrics

### What NOT to Build Yet

- ❌ Complex distributed training
- ❌ Custom CUDA kernels
- ❌ Advanced architecture (attention variants)
- ❌ Ensemble models

**Principle**: Start simple, scale only if needed

---

## 📝 Code Structure for Optimization

```python
# configs.py
MODEL_CONFIGS = {
    'tiny': {
        'd_model': 128,
        'nhead': 4,
        'num_layers': 2,
        'dim_feedforward': 512
    },
    'small': {...},
    'full': {...}
}

DATA_CONFIGS = {
    'tiny': {'n_samples': 1000, 'epochs': 10},
    'small': {'n_samples': 10000, 'epochs': 20},
    'full': {'n_samples': 158135, 'epochs': 50}
}

# train.py
def train(model_size='tiny', data_size='tiny'):
    model_config = MODEL_CONFIGS[model_size]
    data_config = DATA_CONFIGS[data_size]

    model = create_model(model_config)
    data = load_data(data_config['n_samples'])

    train_model(model, data, epochs=data_config['epochs'])

# Usage:
train(model_size='tiny', data_size='tiny')    # 10 min
train(model_size='small', data_size='small')  # 1-2 hours
train(model_size='full', data_size='full')    # 10-20 hours
```

---

## ✅ Action Plan

### Immediate (Next 2 hours)

1. **Implement LSTM Seq2Seq** (simpler than Transformer)
   - Encoder-decoder architecture
   - 3-5x faster than Transformer
   - Good baseline

2. **Create training script** with:
   - Model size configs (tiny/small/full)
   - Data size configs (1k/10k/158k)
   - Checkpointing
   - Early stopping

3. **Test on 1k samples**
   - Verify training works
   - Check output quality
   - Measure speed

### Short-term (Tomorrow)

4. **Train on 10k samples**
   - Use small model
   - Validate with discriminator
   - Decide if Transformer is needed

5. **Optimize based on results**
   - If LSTM works: scale to full data
   - If LSTM fails: implement Transformer

### Medium-term (This Week)

6. **Full training** (if Stage 2 succeeds)
   - Consider cloud GPU if >12 hours locally
   - Monitor carefully
   - Evaluate on test set

---

## 🎯 Success Criteria for Each Stage

### Stage 1 (1k samples, 10 min)
- ✅ Training completes without errors
- ✅ Loss decreases to near-zero (overfitting is GOOD here)
- ✅ Generates valid sequences (correct amino acids)
- ✅ Can memorize small dataset

### Stage 2 (10k samples, 1-2 hours)
- ✅ Validation loss < training loss (not overfitting)
- ✅ 50%+ generated sequences are valid
- ✅ 30%+ score pKd > 7.0 with discriminator
- ✅ Sequences show diversity

### Stage 3 (158k samples, 10-20 hours)
- ✅ Affinity correlation ρ > 0.4
- ✅ 70%+ valid sequences
- ✅ 50%+ score pKd > 7.0
- ✅ Beats random baseline significantly

---

## 🚀 Ready to Implement

**Recommended Next Steps**:

1. Create **LSTM-based model** (faster than Transformer)
2. Create **progressive training script** (tiny/small/full)
3. Test on **1k samples** (10 minutes)
4. Iterate based on results

**Time estimate**: 2-3 hours to have a working system

**Want me to proceed with implementation?**

---

**Last Updated**: 2025-10-31
**Status**: Strategy complete, ready for implementation
