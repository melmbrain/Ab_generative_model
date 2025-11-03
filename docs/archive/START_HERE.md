# 🚀 START HERE - Ab_generative_model

**Welcome to your clean, organized antibody generation project!**

---

## ✅ What You Have

A complete, production-ready system for designing virus-specific antibody libraries.

### File Structure

```
Ab_generative_model/
├── 📖 START_HERE.md          ← You are here!
├── 📖 README.md               ← Full documentation
├── 📖 PROJECT_SUMMARY.md      ← Technical overview
├── 📋 requirements.txt        ← Dependencies
│
├── discriminator/             ← Phase 2 scoring model (ρ=0.85)
│   ├── affinity_discriminator.py
│   └── __init__.py
│
├── generators/                ← Antibody generation
│   ├── template_generator.py  ← Works now!
│   └── __init__.py
│
├── models/                    ← Trained models
│   ├── agab_phase2_model.pth  ← Best model (2.5 MB)
│   └── agab_phase2_results.json
│
├── data/
│   ├── templates/            ← 3 antibody templates
│   ├── results/              ← Your results go here
│   └── example_antigen.txt   ← Test file
│
├── scripts/
│   └── generate_and_score.py ← ⭐ MAIN SCRIPT
│
└── docs/
    └── QUICK_START.md        ← Step-by-step guide
```

---

## 🎯 What This Does

**Input**: Your virus antigen sequence

**Output**: Ranked antibody candidates predicted to bind

**How**:
1. Generate 100-1000 antibody variants (template mutations)
2. Score each with Phase 2 model (Spearman 0.85)
3. Rank by predicted binding affinity
4. Export top candidates for lab testing

---

## ⚡ Quick Start (5 Minutes)

### Step 1: Install Dependencies

```bash
cd /mnt/c/Users/401-24/Desktop/Ab_generative_model
pip install -r requirements.txt
```

### Step 2: Test with Example

```bash
python scripts/generate_and_score.py \
  --antigen data/example_antigen.txt \
  --n-candidates 50 \
  --output data/results/test_run
```

### Step 3: Check Results

```bash
# View top 10
head -n 11 data/results/test_run/top_50_candidates.csv
```

**Expected**: List of antibodies with predicted pKd scores

---

## 📚 Next Steps

1. **Read documentation**:
   - `README.md` - Complete guide
   - `docs/QUICK_START.md` - Detailed tutorial
   - `PROJECT_SUMMARY.md` - Technical details

2. **Try your virus**:
   - Create antigen file: `data/my_virus.txt`
   - Run: `python scripts/generate_and_score.py --antigen data/my_virus.txt`
   - Review: `data/results/my_virus/top_50_candidates.csv`

3. **Customize**:
   - Add your antibody templates
   - Adjust mutation parameters
   - Integrate DiffAb or IgLM (future)

---

## 🔑 Key Files

| File | Purpose |
|------|---------|
| `scripts/generate_and_score.py` | ⭐ Main pipeline (run this!) |
| `discriminator/affinity_discriminator.py` | Scoring model (ρ=0.85) |
| `generators/template_generator.py` | Generate antibody variants |
| `models/agab_phase2_model.pth` | Best trained model (7k samples) |
| `data/example_antigen.txt` | Test antigen sequence |
| `docs/QUICK_START.md` | Step-by-step tutorial |

---

## 💡 Example Usage

### Command Line

```bash
# Generate 100 antibody candidates for SARS-CoV-2
python scripts/generate_and_score.py \
  --antigen data/sars_cov2_spike.txt \
  --n-candidates 100 \
  --output data/results/sars_cov2
```

### Python API

```python
from discriminator import AffinityDiscriminator
from generators import TemplateGenerator

# 1. Generate candidates
gen = TemplateGenerator()
candidates = gen.generate(n_candidates=100)

# 2. Score with discriminator
disc = AffinityDiscriminator()
for cand in candidates:
    score = disc.predict_single(
        cand['full_sequence'],
        your_virus_antigen
    )
    print(f"{cand['id']}: pKd = {score['predicted_pKd']}")
```

---

## 🎓 Understanding Results

**pKd** (higher is better):
- **> 9**: Excellent (picomolar) → Synthesize first!
- **7.5-9**: Good (nanomolar)
- **6-7.5**: Moderate (micromolar)
- **< 6**: Weak

**Kd (nM)** (lower is better):
- **< 10 nM**: Very strong
- **10-100 nM**: Strong
- **100-1000 nM**: Moderate
- **> 1000 nM**: Weak

---

## 🆚 Comparison: Old vs New Project

### Old Project (`Docking prediction/`)

❌ 1.7+ GB size (lots of training data)
❌ Multiple model versions (confusing)
❌ Mixed code and data
❌ Hard to find what you need
❌ No clear starting point

### New Project (`Ab_generative_model/`)

✅ < 10 MB (clean, essential files only)
✅ Single best model (Spearman 0.85)
✅ Organized structure
✅ Clear documentation
✅ Ready to use immediately

---

## 🚨 Common Issues

### "Model not found"

**Solution**: Make sure you're in the right directory

```bash
cd /mnt/c/Users/401-24/Desktop/Ab_generative_model
ls models/  # Should show agab_phase2_model.pth
```

### "Out of memory"

**Solution**: Reduce candidates

```bash
python scripts/generate_and_score.py --n-candidates 20
```

### "Slow first run"

**Normal!** First run downloads ESM-2 model (~140 MB). Subsequent runs are fast (cached).

---

## 📊 Model Performance

```
Phase 2 Model (7k baseline):
  Spearman ρ: 0.8501  ← Excellent!
  Pearson r:  0.9461
  R²:         0.8779
  RMSE:       1.32 pKd

Trained on: 7,015 Ab-Ag pairs
Architecture: Multi-head attention
Features: ESM-2 embeddings + PCA-150
```

**Why this model?**
- Best performance (0.85 vs 0.55 for older models)
- Optimal size (7k vs 159k failed model)
- Production-ready

---

## 🔄 Workflow

```
1. Input virus antigen
        ↓
2. Generate 100-1000 antibody variants
        ↓
3. Score each with Phase 2 model
        ↓
4. Rank by predicted pKd
        ↓
5. Export top 10-50 candidates
        ↓
6. Synthesize and test in lab
        ↓
7. Iterate based on results
```

---

## 🎯 Success Checklist

- [✅] Project installed and organized
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Test run completed (`scripts/generate_and_score.py`)
- [ ] Example results reviewed
- [ ] Documentation read (`README.md`, `QUICK_START.md`)
- [ ] Ready to use with your virus target!

---

## 📞 Getting Help

1. **Check documentation**: `README.md`, `docs/QUICK_START.md`
2. **Review examples**: `data/example_antigen.txt`
3. **Check project summary**: `PROJECT_SUMMARY.md`

---

## 🚀 You're Ready!

Everything is set up and ready to use. Start with:

```bash
python scripts/generate_and_score.py \
  --antigen data/example_antigen.txt \
  --n-candidates 50 \
  --output data/results/first_test
```

Then check the results in `data/results/first_test/top_50_candidates.csv`

**Good luck with your antibody library design!** 🧬

---

**Project Version**: 1.0.0
**Created**: 2025-10-31
**Status**: ✅ Production Ready
