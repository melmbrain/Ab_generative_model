# Quick Start Guide

Get your first virus-specific antibody library in 10 minutes!

---

## Installation (5 minutes)

### Step 1: Install Dependencies

```bash
cd /mnt/c/Users/401-24/Desktop/Ab_generative_model

# Install Python packages
pip install torch transformers pandas numpy scikit-learn

# Or use requirements.txt
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
python -c "import torch; import transformers; print('✅ Installation successful!')"
```

---

## Generate Your First Library (5 minutes)

### Example: SARS-CoV-2 Spike Protein

#### Step 1: Create Antigen File

Create a file `data/sars_cov2_spike.txt` with your target antigen sequence:

```bash
# Create antigen sequence file
cat > data/sars_cov2_spike.txt << 'EOF'
NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF
EOF
```

#### Step 2: Run Pipeline

```bash
python scripts/generate_and_score.py \
  --antigen data/sars_cov2_spike.txt \
  --n-candidates 50 \
  --output data/results/sars_cov2
```

#### Step 3: View Results

```bash
# View top 10 candidates
head -n 11 data/results/sars_cov2/top_50_candidates.csv | column -t -s,

# Or open in Excel/LibreOffice
# File: data/results/sars_cov2/top_50_candidates.csv
```

**Expected Output**:
```
rank  id                predicted_pKd  predicted_Kd_nM  binding_category
1     template_2_v34   8.95           11.2             good
2     template_1_v67   8.82           15.1             good
3     template_3_v12   8.76           17.4             good
...
```

---

## Understanding the Results

### Output Files

After running, you'll get:

```
data/results/sars_cov2/
├── all_candidates_scored.csv      # All 50 candidates
├── top_50_candidates.csv          # Top 50 (same in this case)
├── statistics.json                # Summary statistics
└── antigen_sequence.txt           # Your antigen sequence (for reference)
```

### Interpreting Scores

**pKd** (binding affinity):
- **> 9.0**: Excellent (picomolar/sub-nanomolar) → Synthesize first!
- **7.5-9.0**: Good (nanomolar) → Strong candidates
- **6.0-7.5**: Moderate (micromolar) → Consider if diverse
- **< 6.0**: Weak → Skip

**Kd (nM)** (dissociation constant):
- **< 10 nM**: Very strong binding
- **10-100 nM**: Strong binding
- **100-1000 nM**: Moderate binding
- **> 1000 nM**: Weak binding

---

## Next Steps

### 1. Analyze Diversity

Check that your top candidates are diverse (not all from same template):

```python
import pandas as pd

df = pd.read_csv('data/results/sars_cov2/top_50_candidates.csv')

# Count candidates per template
print(df['template_id'].value_counts())

# Good: Mix of templates
# template_1: 18
# template_2: 17
# template_3: 15

# Bad: All from one template
# template_2: 50  ← Need more diversity!
```

### 2. Select for Experimental Testing

Select 10-20 candidates:
- **Top scorers** (highest pKd)
- **Diverse templates** (from different templates)
- **Mix of categories** (excellent + good)

```python
# Example selection strategy
top_10 = df.head(10)  # Top 10 by score
diverse = df.groupby('template_id').head(3)  # Top 3 from each template
final = pd.concat([top_10, diverse]).drop_duplicates().head(15)
```

### 3. Export for Synthesis

```python
# Export sequences for DNA synthesis
synthesis_order = final[['id', 'antibody_heavy', 'antibody_light', 'predicted_pKd']]
synthesis_order.to_csv('data/results/sars_cov2/synthesis_order.csv', index=False)
```

### 4. Test Experimentally

Methods for validation:
- **ELISA**: Binding affinity measurement
- **SPR/BLI**: Kinetics (kon, koff, KD)
- **Flow cytometry**: Cell-based binding
- **Neutralization assay**: Functional testing (for viruses)

---

## Customization

### Generate More Candidates

```bash
# Generate 500 candidates instead of 50
python scripts/generate_and_score.py \
  --antigen data/sars_cov2_spike.txt \
  --n-candidates 500 \
  --output data/results/sars_cov2_large
```

### Adjust Mutation Level

```bash
# More conservative (2 mutations per variant)
python scripts/generate_and_score.py \
  --antigen data/my_virus.txt \
  --n-candidates 100 \
  --mutations 2 \
  --output data/results/conservative

# More aggressive (5 mutations per variant)
python scripts/generate_and_score.py \
  --antigen data/my_virus.txt \
  --n-candidates 100 \
  --mutations 5 \
  --output data/results/aggressive
```

### Use Custom Templates

Create your own template library:

```csv
# data/templates/my_templates.csv
id,name,heavy_chain,light_chain
my_ab1,Custom Ab 1,EVQLQQS...,DIQMTQS...
my_ab2,Custom Ab 2,QVQLVES...,EIVLTQS...
```

Then use it:

```python
from generators import TemplateGenerator

gen = TemplateGenerator(template_library='data/templates/my_templates.csv')
candidates = gen.generate(n_candidates=100)
```

---

## Python API Usage

For more control, use the Python API directly:

```python
from discriminator import AffinityDiscriminator
from generators import TemplateGenerator
import pandas as pd

# 1. Load antigen
with open('data/sars_cov2_spike.txt') as f:
    antigen = f.read().strip()

# 2. Generate candidates
generator = TemplateGenerator()
candidates = generator.generate(n_candidates=100, mutations_per_variant=3)

# 3. Score all candidates
discriminator = AffinityDiscriminator()
results = []

for cand in candidates:
    pred = discriminator.predict_single(
        antibody_seq=cand['full_sequence'],
        antigen_seq=antigen
    )
    results.append({
        'id': cand['id'],
        'pKd': pred['predicted_pKd'],
        'Kd_nM': pred['predicted_Kd_nM'],
        'sequence': cand['full_sequence']
    })

# 4. Rank and save
df = pd.DataFrame(results)
df = df.sort_values('pKd', ascending=False)
df.to_csv('results.csv', index=False)

print(f"Top candidate: pKd = {df.iloc[0]['pKd']:.2f}")
```

---

## Troubleshooting

### Issue: "Model not found"

```
FileNotFoundError: Model not found: models/agab_phase2_model.pth
```

**Solution**: Make sure you're in the correct directory

```bash
cd /mnt/c/Users/401-24/Desktop/Ab_generative_model
ls models/  # Should show agab_phase2_model.pth
```

### Issue: "ESM-2 model download slow"

First time running will download ESM-2 (~140 MB). This is normal.

**Speed it up**: Run once, model is cached for future use.

### Issue: "Out of memory"

Reduce batch size or number of candidates:

```bash
# Generate in smaller batches
python scripts/generate_and_score.py --n-candidates 20
```

---

## Performance Tips

### Speed Up Scoring

Use GPU if available:

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Scoring is automatic on GPU if available
```

### Parallel Processing

For large libraries (1000+ candidates), process in batches:

```python
# Process in batches of 100
for i in range(0, len(candidates), 100):
    batch = candidates[i:i+100]
    scores = score_batch(batch)
    save_batch_results(scores, f'batch_{i//100}')
```

---

## What's Next?

1. ✅ **Optimize parameters**: Try different mutation levels
2. ✅ **Add custom templates**: Use your own antibodies
3. ✅ **Test multiple viruses**: Screen against variants
4. ✅ **Integrate DiffAb**: Add advanced generative models (see `docs/ADDING_GENERATORS.md`)
5. ✅ **Validate experimentally**: Synthesize and test top candidates

---

## Getting Help

- **Documentation**: See `docs/` directory
- **Examples**: See `examples/` directory
- **Issues**: Check GitHub issues or open a new one

---

**You're ready to design antibody libraries for any virus target!** 🚀
