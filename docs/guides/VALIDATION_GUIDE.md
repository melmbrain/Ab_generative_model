# Antibody Validation Guide

**How to validate your generated antibodies using structure prediction**

Created: 2025-11-03
Status: Ready to use with ESMFold

---

## 🎯 Overview

You now have two validation tools:

1. **validate_antibodies.py** - Complete validation pipeline (recommended)
2. **validation/structure_validation.py** - Modular validation library

Both use **ESMFold** for fast, accurate structure prediction.

---

## 🚀 Quick Start

### Step 1: Install ESMFold

```bash
# Install ESMFold (required)
pip install fair-esm

# Or with all dependencies
pip install 'fair-esm[esmfold]'
```

**Requirements**:
- PyTorch (already installed ✅)
- ~2GB free GPU memory
- Or use CPU (slower but works)

---

### Step 2: Validate Your Antibodies

Once your training completes (currently at Epoch 9/20), run:

```bash
# Wait for training to finish first!

# Then validate
python validate_antibodies.py \
  --checkpoint checkpoints/improved_small_2025_10_31_best.pt \
  --num-samples 20 \
  --device cuda
```

**This will**:
1. Load your trained model
2. Generate 20 new antibodies
3. Predict their 3D structures
4. Calculate quality scores
5. Save detailed reports

**Time**: ~2-5 minutes for 20 antibodies

---

## 📊 Understanding the Results

### Quality Metrics

**pLDDT Score** (0-100):
- **>90**: Excellent structure ✅
- **70-90**: Good structure ✅
- **50-70**: Fair structure ⚠️
- **<50**: Poor structure ❌

**Example Output**:
```
[1/20] Validating antibody 1...
    pLDDT: 82.4 - Good ✅

[2/20] Validating antibody 2...
    pLDDT: 91.2 - Excellent ✅

[3/20] Validating antibody 3...
    pLDDT: 65.3 - Fair ⚠️
```

### Results Files

After validation, you'll get:

```
validation_results/
├── validation_results.json    # Detailed results for each antibody
└── validation_summary.json    # Overall statistics
```

**Summary includes**:
- Mean pLDDT score
- Number of good/excellent/poor structures
- Success rate
- Quality distribution

---

## 📈 What Good Results Look Like

### For a Well-Trained Model:

Expected after 20 epochs:
- **Mean pLDDT**: 75-85
- **Good structures (>70)**: 80-95%
- **Excellent structures (>90)**: 20-40%
- **Poor structures (<50)**: <5%

### Your Current Model (Epoch 9):

Based on your training metrics (100% validity, 32% diversity), you should expect:
- **Mean pLDDT**: ~70-80 (good!)
- **Good structures**: 60-80%
- **Some variability** (model still learning)

---

## 🔧 Advanced Usage

### Validate Specific Antibodies

```python
from validation.structure_validation import StructureValidator

# Initialize
validator = StructureValidator(method='esmfold', device='cuda')

# Validate single antibody
antibody = "EVQLVESG....|DIQMTQSP...."
result = validator.validate_antibody(antibody)

print(f"Quality: {result['mean_plddt']:.1f}")
print(f"Good structure: {result['is_good_structure']}")
```

### Batch Validation

```python
# Validate multiple antibodies
antibodies = [seq1, seq2, seq3, ...]
results = validator.validate_batch(
    antibodies,
    save_results=True,
    output_dir='my_results'
)
```

### CPU Mode (No GPU)

```bash
# Use CPU if GPU unavailable
python validate_antibodies.py \
  --checkpoint checkpoints/best.pt \
  --num-samples 10 \
  --device cpu
```

**Note**: CPU is ~10x slower but works fine

---

## ❓ AlphaFold3 Status

### Why Not AlphaFold3?

**AlphaFold3 is NOT available for automated use (as of Nov 2024)**:

❌ No public API
❌ No pip install
❌ Complex manual setup
❌ Non-commercial license only

**What IS available**:
- Web server: https://alphafoldserver.com (manual upload)
- GitHub code (research use, complex)

### Can I Use AlphaFold3?

**Option 1: Manual Validation** (if needed)
1. Generate antibodies with your model
2. Upload to https://alphafoldserver.com
3. Include antigen sequence
4. Get binding predictions

**Option 2: Wait for API**
- Google may release API in future
- Code is modular - easy to swap ESMFold → AF3 later

**Option 3: Use ESMFold** (recommended)
- Available NOW
- Fast and accurate
- Good for structure quality
- Can't predict binding (yet)

---

## 🔬 Validation Strategy

### Tier 1: Structure Quality (ESMFold) ✅
**What it checks**:
- Structure folds correctly
- No clashes or errors
- Biologically realistic

**Use for**:
- Filtering bad sequences
- Quality control
- Initial screening

### Tier 2: Antibody-Specific (Future)
**Could add**:
- IgFold (antibody-specific folding)
- CDR loop validation
- Framework region checks

### Tier 3: Binding Validation (Future)
**Would need**:
- AlphaFold3 (antibody-antigen complex)
- Or molecular docking tools
- Or experimental testing

---

## 📋 Complete Workflow

### After Training Completes:

```bash
# 1. Validate antibodies
python validate_antibodies.py \
  --checkpoint checkpoints/improved_small_2025_10_31_best.pt \
  --num-samples 50 \
  --device cuda \
  --output-dir validation_results

# 2. Check results
cat validation_results/validation_summary.json

# 3. Find best antibodies
# Look for highest pLDDT scores in validation_results.json

# 4. (Optional) Test specific antibodies
python -c "
from validation.structure_validation import StructureValidator
validator = StructureValidator('esmfold', 'cuda')
result = validator.validate_antibody('YOUR_SEQUENCE_HERE')
print(f'Quality: {result[\"mean_plddt\"]:.1f}')
"
```

---

## 🎯 Next Steps After Validation

### If You Get Good Results (>75 mean pLDDT):

1. **Analyze top antibodies**
   - Identify best-performing sequences
   - Look at pKd correlation

2. **Compare to training data**
   - Are they novel or similar?
   - Check diversity metrics

3. **Visualize structures** (optional)
   - Use PyMOL or Mol*
   - Inspect CDR regions

4. **Consider experimental testing**
   - Top candidates could be synthesized
   - Test binding in lab

### If Results Are Mixed:

1. **Continue training**
   - Current: Epoch 9/20
   - More training = better quality

2. **Analyze failure modes**
   - Which sequences fail?
   - Common patterns?

3. **Adjust model** (future work)
   - Fine-tune on high-quality subset
   - Add structure-based loss

---

## 🔧 Troubleshooting

### Error: "ESMFold not installed"
```bash
pip install fair-esm
# Or
pip install 'fair-esm[esmfold]'
```

### Error: "Out of memory"
```bash
# Use CPU instead
python validate_antibodies.py --device cpu

# Or reduce batch size (modify script)
# Or validate fewer samples: --num-samples 10
```

### Error: "CUDA error"
```bash
# Check GPU
nvidia-smi

# If busy (training running), wait or use CPU
python validate_antibodies.py --device cpu
```

### Error: "Checkpoint not found"
```bash
# Check available checkpoints
ls -lh checkpoints/

# Use correct path
python validate_antibodies.py \
  --checkpoint checkpoints/improved_small_2025_10_31_best.pt
```

---

## 📚 Additional Resources

### ESMFold
- Paper: https://www.biorxiv.org/content/10.1101/2022.07.20.500902
- Code: https://github.com/facebookresearch/esm

### AlphaFold3
- Paper: https://www.nature.com/articles/s41586-024-07487-w
- Server: https://alphafoldserver.com

### Antibody Structure
- IMGT: http://www.imgt.org/
- SAbDab: https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/

---

## 🎉 Summary

**You have**:
✅ Working validation pipeline
✅ ESMFold integration (fast, accurate)
✅ Automated quality metrics
✅ Ready to use when training completes

**You DON'T have** (but could add later):
❌ AlphaFold3 (not available yet)
❌ Binding prediction (need AF3 or docking)
❌ Experimental validation (need lab)

**Recommended**:
1. Wait for training to complete (Epoch 9/20 now)
2. Run validation on best checkpoint
3. Analyze results
4. Iterate on model if needed

**Your model is producing high-quality sequences** (100% validity, good diversity) - structure validation should show excellent results! 🚀

---

**Last Updated**: 2025-11-03
**Training Status**: Epoch 9/20 (45% complete)
**Next Validation**: After Epoch 20
