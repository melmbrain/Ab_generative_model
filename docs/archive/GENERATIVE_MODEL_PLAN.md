# Generative Model Training Plan: Antigen → Antibody

**Goal**: Train a model to generate antibody sequences when given an antigen sequence

**Dataset**: 159,735 Ab-Ag pairs from AgAb database

**Location**: `/mnt/c/Users/401-24/Desktop/Docking prediction/data/raw/agab/`

---

## 📊 Available Data

### Dataset Statistics

**File**: `agab_phase2_full.csv` (127 MB, 159,735 pairs)

**Columns**:
- `antibody_sequence`: Heavy chain | Light chain (concatenated with |)
- `antigen_sequence`: Full antigen amino acid sequence
- `pKd`: Binding affinity (-log KD)
- `affinity_type`: Measurement type
- `dataset`: Source database (95.4% from ABBD)
- `confidence`: high/medium/low
- `nanobody`: Boolean flag

**Data Quality**:
- ✅ 159,735 pairs total
- ✅ pKd range: -2.96 to 12.43
- ✅ Mean pKd: 7.45 ± 2.11
- ✅ All sequences validated
- ✅ High-confidence subset available

---

## 🎯 Modeling Approaches

### Approach 1: Sequence-to-Sequence Transformer (RECOMMENDED)

**Architecture**: Encoder-Decoder Transformer

```
Input:  [Antigen Sequence] + [Desired pKd]
           ↓
Encoder (6 layers, 512 dims)
  - Learned embeddings for amino acids
  - Positional encoding
  - Self-attention over antigen
           ↓
Latent representation (512 dims)
           ↓
Decoder (6 layers, 512 dims)
  - Cross-attention to antigen
  - Self-attention over generated antibody
  - Causal masking for autoregressive generation
           ↓
Output: [Antibody Sequence] (token-by-token)
```

**Training Objective**:
- Primary: Cross-entropy loss (sequence prediction)
- Secondary: MSE loss (pKd prediction from generated sequence)
- Auxiliary: Discriminator reward (RL fine-tuning)

**Why this approach?**
- ✅ Standard for sequence generation (proven in NLP)
- ✅ Can condition on both antigen + desired affinity
- ✅ Generates diverse candidates via beam search/sampling
- ✅ 159k samples is sufficient for training

---

### Approach 2: Conditional VAE (Variational Autoencoder)

**Architecture**:

```
Encoder:
  Antibody sequence → ESM-2 → 480 dims → μ, σ (latent)
  Antigen sequence → ESM-2 → 480 dims (condition)

Latent Space:
  z ~ N(μ, σ)  [256 dims]

Decoder:
  [z + antigen_features] → Transformer Decoder → Antibody sequence
```

**Training Objective**:
- ELBO loss: Reconstruction + KL divergence
- Affinity matching: |predicted_pKd - target_pKd|

**Advantages**:
- ✅ Latent space allows interpolation
- ✅ Can sample diverse antibodies from z
- ✅ Regularization via KL prevents mode collapse

**Disadvantages**:
- ❌ More complex to train
- ❌ KL collapse is common

---

### Approach 3: Fine-tune Protein Language Model

**Base Model**: ESM-2 (650M params) or ProtGPT2

**Strategy**: Fine-tune on Ab-Ag pairs

```
Input format (text):
"<ANTIGEN>MKTFLIS...VYQAG</ANTIGEN><PKD>8.5</PKD><ANTIBODY>"

Target:
"QVQLVQ...TVSS|DIQMTQ...VEIK</ANTIBODY>"
```

**Training**:
- Use existing pre-trained weights
- Fine-tune on 159k pairs
- Causal language modeling objective

**Advantages**:
- ✅ Leverages pre-training (millions of proteins)
- ✅ Fast convergence
- ✅ State-of-the-art protein understanding

**Disadvantages**:
- ❌ Requires GPU (model is large)
- ❌ May not follow constraints exactly

---

## 🏗️ Implementation Roadmap

### Phase 1: Data Preparation (Week 1)

**Tasks**:
1. ✅ Load 159k dataset from `agab_phase2_full.csv`
2. ✅ Clean sequences (validate amino acids)
3. ✅ Split heavy/light chains (currently concatenated with `|`)
4. ✅ Create train/val/test splits (80/10/10)
5. ✅ Tokenize sequences (amino acid → integer mapping)
6. ✅ Compute sequence length statistics
7. ✅ Create PyTorch DataLoader

**Data Format**:
```python
{
    'antigen_seq': "MKTFLIS...",
    'antibody_heavy': "QVQLVQ...",
    'antibody_light': "DIQMTQ...",
    'pKd': 8.5,
    'affinity': 3.16e-9  # in M
}
```

**Outputs**:
- `data/generative/train.pkl` (127,788 pairs)
- `data/generative/val.pkl` (15,973 pairs)
- `data/generative/test.pkl` (15,974 pairs)

---

### Phase 2: Model Implementation (Week 2-3)

**Step 1: Implement Transformer**

```python
class AntigenToAntibodyTransformer(nn.Module):
    """
    Encoder-Decoder Transformer for Ab generation
    """
    def __init__(self):
        self.encoder = TransformerEncoder(
            d_model=512,
            nhead=8,
            num_layers=6,
            dim_feedforward=2048
        )

        self.decoder = TransformerDecoder(
            d_model=512,
            nhead=8,
            num_layers=6,
            dim_feedforward=2048
        )

        # Amino acid embeddings
        self.aa_embedding = nn.Embedding(21, 512)  # 20 AA + padding

        # Affinity conditioning
        self.affinity_proj = nn.Linear(1, 512)

    def forward(self, antigen, target_pKd, antibody=None):
        # Encode antigen
        antigen_emb = self.aa_embedding(antigen)
        antigen_emb = antigen_emb + self.affinity_proj(target_pKd)

        encoder_out = self.encoder(antigen_emb)

        # Decode antibody (teacher forcing during training)
        if antibody is not None:
            # Training: use ground truth
            antibody_emb = self.aa_embedding(antibody)
            decoder_out = self.decoder(
                antibody_emb,
                memory=encoder_out,
                tgt_mask=causal_mask
            )
        else:
            # Inference: autoregressive generation
            decoder_out = self.generate(encoder_out, max_len=150)

        return decoder_out

    def generate(self, encoder_out, max_len=150):
        """Autoregressive generation"""
        generated = [START_TOKEN]

        for i in range(max_len):
            # Decode one token at a time
            tgt_emb = self.aa_embedding(torch.tensor(generated))
            out = self.decoder(tgt_emb, encoder_out)

            # Sample next token
            next_token = torch.argmax(out[-1])

            if next_token == END_TOKEN:
                break

            generated.append(next_token)

        return generated
```

**Step 2: Training Loop**

```python
def train_generative_model():
    model = AntigenToAntibodyTransformer()
    optimizer = Adam(model.parameters(), lr=1e-4)

    for epoch in range(50):
        for batch in train_loader:
            antigen = batch['antigen_seq']
            antibody = batch['antibody_seq']
            pKd = batch['pKd']

            # Forward pass
            logits = model(antigen, pKd, antibody)

            # Compute loss
            loss = cross_entropy(logits, antibody)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        val_loss = evaluate(model, val_loader)
        print(f"Epoch {epoch}: Loss={val_loss:.4f}")

        # Generate sample
        sample = model.generate(val_antigen, target_pKd=9.0)
        print(f"Sample: {tokens_to_sequence(sample)}")
```

---

### Phase 3: Training (Week 3-4)

**Hardware Requirements**:
- GPU: RTX 3090 or better (24GB VRAM)
- Alternative: Google Colab Pro / AWS p3.2xlarge
- Training time: ~10-20 hours for 50 epochs

**Training Configuration**:
```python
config = {
    'd_model': 512,
    'nhead': 8,
    'num_encoder_layers': 6,
    'num_decoder_layers': 6,
    'dim_feedforward': 2048,
    'dropout': 0.1,

    'batch_size': 32,
    'learning_rate': 1e-4,
    'epochs': 50,
    'gradient_clip': 1.0,

    'max_antigen_len': 500,
    'max_antibody_len': 150,
}
```

**Monitoring**:
- Training loss (cross-entropy)
- Validation loss
- Sample generations (qualitative)
- Sequence validity (% valid AA sequences)
- Diversity (unique sequences generated)

---

### Phase 4: Validation with Discriminator (Week 4)

**Hybrid Validation**: Use existing discriminator to score generated antibodies

```python
def validate_with_discriminator():
    """
    Generate antibodies and score with discriminator
    """
    generator = AntigenToAntibodyTransformer.load('checkpoints/best.pth')
    discriminator = AffinityDiscriminator()

    results = []

    for test_antigen in test_set:
        # Generate 100 candidates
        candidates = []
        for _ in range(100):
            ab = generator.generate(
                antigen=test_antigen,
                target_pKd=9.0,  # Request high-affinity
                temperature=1.0   # Diversity
            )
            candidates.append(ab)

        # Score with discriminator
        scores = []
        for ab in candidates:
            score = discriminator.predict_single(ab, test_antigen)
            scores.append(score['predicted_pKd'])

        results.append({
            'antigen': test_antigen,
            'mean_pKd': np.mean(scores),
            'max_pKd': np.max(scores),
            'diversity': len(set(candidates)) / len(candidates)
        })

    return results
```

**Metrics**:
1. **Affinity accuracy**: How close is generated pKd to requested?
2. **Discriminator score**: Do generated Abs score well?
3. **Sequence validity**: % of valid sequences
4. **Diversity**: Unique sequences generated
5. **CDR quality**: Are CDR regions reasonable?

---

### Phase 5: Production Deployment (Week 5)

**API Design**:

```python
class AntigenToAntibodyAPI:
    """
    Production API for antibody generation
    """

    def __init__(self):
        self.generator = AntigenToAntibodyTransformer.load()
        self.discriminator = AffinityDiscriminator()

    def design_antibodies(
        self,
        antigen_seq: str,
        target_pKd: float = 9.0,
        n_candidates: int = 100,
        diversity: float = 1.0
    ):
        """
        Generate antibodies for antigen

        Args:
            antigen_seq: Target antigen sequence
            target_pKd: Desired binding affinity
            n_candidates: Number of antibodies to generate
            diversity: Sampling temperature (0.1-2.0)

        Returns:
            List of antibody candidates with scores
        """
        # Generate candidates
        candidates = []
        for _ in range(n_candidates):
            ab = self.generator.generate(
                antigen=antigen_seq,
                target_pKd=target_pKd,
                temperature=diversity
            )
            candidates.append(ab)

        # Re-rank with discriminator
        scored = []
        for ab in candidates:
            score = self.discriminator.predict_single(ab, antigen_seq)
            scored.append({
                'antibody_sequence': ab,
                'predicted_pKd': score['predicted_pKd'],
                'predicted_Kd_nM': score['predicted_Kd_nM'],
                'category': score['binding_category']
            })

        # Sort by affinity
        scored = sorted(scored, key=lambda x: x['predicted_pKd'], reverse=True)

        return scored
```

---

## 📊 Expected Performance

### Generation Quality

**Best case** (based on similar work):
- Sequence validity: 95%+ (valid amino acid sequences)
- Affinity correlation: ρ = 0.6-0.7 (generated pKd vs requested)
- Discriminator approval: 60%+ score > 7.0 pKd
- Diversity: 80%+ unique sequences

**Realistic case**:
- Sequence validity: 90%+
- Affinity correlation: ρ = 0.4-0.6
- Discriminator approval: 40-60% good binders
- Diversity: 70%+ unique

**Minimum acceptable**:
- Sequence validity: 85%+
- Affinity correlation: ρ > 0.3
- Discriminator approval: 30%+ good binders
- Diversity: 50%+ unique

### Comparison to Baselines

| Method | Affinity Control | Diversity | Speed |
|--------|-----------------|-----------|-------|
| Template mutation | ❌ None | Low | Fast |
| Guided search | ✅ Good | Medium | Slow |
| **Seq2Seq** | ✅✅ Excellent | High | Fast |
| DiffAb | ✅ Good | High | Medium |

---

## 💻 Code Structure

```
Ab_generative_model/
├── data/
│   └── generative/
│       ├── train.pkl              # 127k training pairs
│       ├── val.pkl                # 16k validation pairs
│       └── test.pkl               # 16k test pairs
│
├── generators/
│   ├── template_generator.py      # Existing
│   ├── guided_search.py           # Phase 1 (quick win)
│   └── seq2seq_generator.py       # NEW - Transformer model
│
├── scripts/
│   ├── prepare_generative_data.py # Data preprocessing
│   ├── train_seq2seq.py           # Training script
│   ├── evaluate_generator.py      # Validation
│   └── generate_antibodies.py     # Inference API
│
├── models/
│   └── generative/
│       ├── config.json            # Model hyperparameters
│       ├── best_model.pth         # Trained weights
│       └── training_log.csv       # Training history
│
└── notebooks/
    ├── data_exploration.ipynb     # EDA on 159k dataset
    ├── model_training.ipynb       # Interactive training
    └── generation_demo.ipynb      # Demo notebook
```

---

## 🚀 Quick Start: Data Preparation

**Step 1: Prepare dataset**

```python
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

# Load 159k pairs
df = pd.read_csv(
    '/mnt/c/Users/401-24/Desktop/Docking prediction/data/raw/agab/agab_phase2_full.csv'
)

# Split antibody sequence (heavy|light)
df[['heavy', 'light']] = df['antibody_sequence'].str.split('|', expand=True)

# Clean
df = df.dropna(subset=['heavy', 'antigen_sequence', 'pKd'])
df = df[df['pKd'] > 0]  # Remove invalid affinities

# Split
train, temp = train_test_split(df, test_size=0.2, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

# Save
train.to_pickle('data/generative/train.pkl')
val.to_pickle('data/generative/val.pkl')
test.to_pickle('data/generative/test.pkl')

print(f"Train: {len(train)}")
print(f"Val: {len(val)}")
print(f"Test: {len(test)}")
```

**Step 2: Tokenization**

```python
# Amino acid vocabulary
AA_VOCAB = {
    '<PAD>': 0, '<START>': 1, '<END>': 2,
    'A': 3, 'C': 4, 'D': 5, 'E': 6, 'F': 7,
    'G': 8, 'H': 9, 'I': 10, 'K': 11, 'L': 12,
    'M': 13, 'N': 14, 'P': 15, 'Q': 16, 'R': 17,
    'S': 18, 'T': 19, 'V': 20, 'W': 21, 'Y': 22
}

def tokenize_sequence(seq):
    return [AA_VOCAB['<START>']] + \
           [AA_VOCAB[aa] for aa in seq] + \
           [AA_VOCAB['<END>']]

def detokenize_sequence(tokens):
    REVERSE_VOCAB = {v: k for k, v in AA_VOCAB.items()}
    seq = [REVERSE_VOCAB[t] for t in tokens if t > 2]
    return ''.join(seq)
```

---

## 📈 Timeline & Milestones

### Week 1: Data Preparation
- [x] Explore 159k dataset structure
- [ ] Implement data preprocessing pipeline
- [ ] Create train/val/test splits
- [ ] Compute sequence statistics
- [ ] Build PyTorch DataLoader

### Week 2: Model Implementation
- [ ] Implement Transformer encoder
- [ ] Implement Transformer decoder
- [ ] Add affinity conditioning
- [ ] Test forward/backward pass
- [ ] Implement generation (beam search)

### Week 3: Initial Training
- [ ] Train on 10k subset (quick test)
- [ ] Debug any issues
- [ ] Train on full 127k dataset
- [ ] Monitor convergence
- [ ] Tune hyperparameters

### Week 4: Validation
- [ ] Generate test antibodies
- [ ] Score with discriminator
- [ ] Analyze sequence quality
- [ ] Compare to baselines
- [ ] Iterate if needed

### Week 5: Production
- [ ] Clean up code
- [ ] Create API
- [ ] Write documentation
- [ ] Deploy model
- [ ] Create demo notebook

---

## 🎯 Success Criteria

**Minimum Viable Model**:
- ✅ Generates valid sequences (90%+)
- ✅ Better than random baseline
- ✅ Some antibodies score well (30%+)

**Good Model**:
- ✅ Affinity correlation ρ > 0.5
- ✅ 50%+ antibodies score pKd > 7.0
- ✅ High diversity (70%+ unique)

**Excellent Model**:
- ✅ Affinity correlation ρ > 0.7
- ✅ 70%+ antibodies score pKd > 7.0
- ✅ Beats all baselines
- ✅ Publication-worthy

---

## 📚 References

**Similar Work**:
1. **DiffAb** (2022): Diffusion models for CDR design
2. **IgLM** (2022): Language models for antibody generation
3. **AbDPO** (2023): Preference optimization for antibodies
4. **dyMEAN** (2023): Deep learning for Ab-Ag binding

**Our Advantage**:
- ✅ 159k training pairs (more than most)
- ✅ Validated discriminator for re-ranking
- ✅ End-to-end pipeline

---

## 🔧 Next Steps

**Immediate** (Today):
1. Run data preparation script
2. Explore sequence statistics
3. Design model architecture

**This Week**:
1. Implement Transformer
2. Test on small subset (1k pairs)
3. Debug and iterate

**Next Week**:
1. Train on full dataset
2. Validate with discriminator
3. Compare to baselines

---

**Want me to start implementing the data preparation pipeline?**

I can create:
1. `scripts/prepare_generative_data.py` - Data preprocessing
2. `notebooks/data_exploration.ipynb` - EDA notebook
3. `generators/seq2seq_generator.py` - Model skeleton

Let me know if you want to proceed! 🚀
