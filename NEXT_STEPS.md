# Next Steps - Model Improvement

**Date**: 2025-01-15
**Status**: ✅ Analysis Complete, Ready to Proceed

---

## What We Just Did (Last Hour)

### ✅ Completed:

1. **Analyzed Training Data** (`analyze_current_model.py`)
   - Dataset: 158,135 antibody-antigen pairs
   - Mean affinity: 7.54 pKd (good range)
   - Only 10 unique antigens in sample (limited diversity)
   - **Good news**: Large dataset, working pipeline

2. **Identified Critical Gaps**
   - ❌ No epitope prediction (can't process new viruses)
   - ❌ Affinity uncalibrated
   - ❌ No binding validation
   - ✅ Structure validation working
   - ✅ Literature validation working (unique feature!)

3. **Created Epitope Predictor** (`epitope_predictor.py`)
   - Local method: Works immediately (hydrophilicity-based)
   - IEDB API: Available as fallback
   - ⚠️  Needs tuning: Missed 2 known SARS-CoV-2 epitopes

---

## Priority 1: Fix Epitope Prediction (TODAY)

### Problem:
Current predictor missed known epitopes `YQAGSTPCNGVEG` and `GKIADYNYKLPDDFT`

### Solution: Lower threshold

**Action**: Edit epitope_predictor.py and test with threshold=0.35

```bash
# Test with different thresholds
python3 -c "
from epitope_predictor import EpitopePredictor

# Load SARS-CoV-2 spike
with open('sars_cov2_spike.fasta') as f:
    lines = f.readlines()
    sequence = ''.join([line.strip() for line in lines if not line.startswith('>')])

for threshold in [0.3, 0.35, 0.4, 0.45, 0.5]:
    print(f'\n=== Threshold: {threshold} ===')
    predictor = EpitopePredictor(method='local', threshold=threshold)
    epitopes = predictor.predict(sequence, top_k=20)

    # Check if known epitopes are found
    known = ['YQAGSTPCNGVEG', 'GKIADYNYKLPDDFT']
    for k in known:
        found = any(k in ep['sequence'] for ep in epitopes)
        print(f'  {k}: {'FOUND' if found else 'MISSED'}')
"
```

**Expected**: Threshold ~0.35 should catch both known epitopes

**Time**: 30 minutes

---

## Priority 2: Integrate Epitope Predictor into Pipeline (TODAY)

### Action: Update pipeline to use real epitope prediction

**File to create**: `run_pipeline_v2.py`

```python
"""
Enhanced Pipeline v2 with Epitope Prediction

Improvements over v1:
1. Real epitope prediction (not hardcoded)
2. Configurable threshold
3. Top-K epitope selection
"""

from epitope_predictor import EpitopePredictor
from generators.antibody_generator import AntibodyGenerator
from validation.structure_validation import IgFoldValidator
import json

def run_pipeline_v2(antigen_sequence, virus_name, target_pkd=9.5,
                   epitope_threshold=0.35, top_k_epitopes=5,
                   top_k_antibodies=3):
    """
    Complete pipeline with epitope prediction

    Args:
        antigen_sequence: Full virus protein sequence
        virus_name: Name of virus (for literature validation)
        target_pkd: Desired binding affinity
        epitope_threshold: Epitope prediction threshold (0-1)
        top_k_epitopes: Number of epitopes to target
        top_k_antibodies: Number of antibodies to generate per epitope

    Returns:
        List of generated antibodies with validation
    """

    print("="*80)
    print(f"ANTIBODY GENERATION PIPELINE V2")
    print("="*80)
    print(f"\nVirus: {virus_name}")
    print(f"Antigen length: {len(antigen_sequence)} aa")
    print(f"Target affinity: pKd = {target_pkd}")

    # Step 1: Predict epitopes
    print(f"\n{'─'*80}")
    print("STEP 1: Epitope Prediction")
    print(f"{'─'*80}")

    predictor = EpitopePredictor(method='local', threshold=epitope_threshold)
    epitopes = predictor.predict(antigen_sequence, top_k=top_k_epitopes * 2)

    print(f"\nPredicted {len(epitopes)} epitopes")

    # Step 2: Literature validation (optional, use existing code)
    # ... existing code from epitope_to_antibody_pipeline.py ...

    # Step 3: Generate antibodies
    # ... existing code ...

    # Step 4: Structure validation
    # ... existing IgFold validation ...

    return results
```

**Time**: 1 hour

---

## Priority 3: Create Benchmark Dataset (TOMORROW)

### Goal: Test model on known antibody-antigen pairs

**Data Sources**:
1. **CoV-AbDab** (SARS-CoV-2 antibodies with structures)
   - URL: http://opig.stats.ox.ac.uk/webapps/covabdab/
   - Download CSV
   - Extract: epitope, heavy, light, neutralization data

**Action**: Create script

```python
# create_benchmark.py
import pandas as pd
import requests

# Download CoV-AbDab
url = "http://opig.stats.ox.ac.uk/webapps/covabdab/static/downloads/CoV-AbDab_260125.csv"
df = pd.read_csv(url)

# Filter for complete data
df = df.dropna(subset=['Heavy V-REGION', 'Light V-REGION', 'Protein + Epitope'])

# Extract epitopes
# ... process epitope annotations ...

# Save benchmark
benchmark = df[['Protein + Epitope', 'Heavy V-REGION', 'Light V-REGION', 'Neutralising Vs']]
benchmark.to_csv('benchmark/sars_cov2_known_antibodies.csv', index=False)

print(f"Created benchmark with {len(benchmark)} antibodies")
```

**Time**: 2-3 hours

---

## This Week's Schedule

### Day 1 (TODAY):
- [x] Analyze training data ✅
- [x] Create epitope predictor ✅
- [ ] Tune epitope predictor threshold (30 min)
- [ ] Test on SARS-CoV-2, verify known epitopes found (30 min)
- [ ] Update pipeline to use epitope predictor (1 hour)

**Deliverable**: Working epitope prediction integrated

---

### Day 2 (TOMORROW):
- [ ] Download CoV-AbDab data (30 min)
- [ ] Create benchmark dataset script (2 hours)
- [ ] Extract 50-100 known antibody pairs (1 hour)
- [ ] Test model on benchmark (1 hour)

**Deliverable**: Benchmark dataset + initial performance metrics

---

### Days 3-4 (THIS WEEK):
- [ ] Analyze benchmark results
- [ ] If affinity correlation poor → Add calibration
- [ ] If epitope accuracy poor → Adjust threshold/method
- [ ] Document findings

**Deliverable**: Model performance report, calibration parameters (if needed)

---

### Day 5 (END OF WEEK):
- [ ] Run full pipeline v2 on SARS-CoV-2
- [ ] Generate top 3 antibody candidates
- [ ] Review all validation metrics
- [ ] **DECISION**: Proceed to synthesis or iterate?

**Deliverable**: 3 synthesis-ready antibodies OR list of issues to fix

---

## Week 2 Plan (Next Week)

### If Week 1 Success (R² > 0.4, epitopes validated):
1. Add AlphaFold-Multimer docking (Days 1-2)
2. Add developability scoring (Day 3)
3. Integrate all into production pipeline (Day 4)
4. Final test + prepare synthesis order (Day 5)

**Outcome**: Ready for synthesis (~$2000 for 3 antibodies)

---

### If Week 1 Needs Improvement (R² < 0.4):
1. Debug affinity prediction (Days 1-2)
2. Try alternative epitope predictors (Day 3)
3. Expand benchmark dataset (Day 4)
4. Re-test and iterate (Day 5)

**Outcome**: Better computational validation before synthesis

---

## Success Criteria (End of Week 1)

### Must Have:
- [  ] Epitope predictor finds ≥2 known SARS-CoV-2 epitopes
- [  ] Benchmark dataset with ≥50 known antibodies created
- [  ] Model tested on benchmark

### Nice to Have:
- [  ] Affinity correlation R² > 0.4
- [  ] Sequence recovery score > 40
- [  ] Pipeline v2 generates candidates in <10 minutes

### Red Flags:
- 🚨 Epitope predictor accuracy <50%
- 🚨 Model generates invalid sequences on benchmark
- 🚨 All predicted affinities off by >2 log units

---

## Quick Commands

### Run Today's Tasks:

```bash
# 1. Tune epitope threshold
python3 epitope_predictor.py  # Check output

# 2. Test on SARS-CoV-2
python3 -c "
from epitope_predictor import EpitopePredictor

with open('sars_cov2_spike.fasta') as f:
    seq = ''.join([l.strip() for l in f if not l.startswith('>')])

predictor = EpitopePredictor(threshold=0.35)
epitopes = predictor.predict(seq, top_k=10)

for i, ep in enumerate(epitopes, 1):
    print(f'{i}. {ep['sequence'][:30]}... (score: {ep['score']:.2f})')
"

# 3. Check if known epitopes found
grep -i "YQAGSTPCNGVEG" # Should appear in output
grep -i "GKIADYNYKLPDDFT" # Should appear
```

---

## Files Created Today

| File | Purpose | Status |
|------|---------|--------|
| `analyze_current_model.py` | Training data analysis | ✅ Complete |
| `epitope_predictor.py` | B-cell epitope prediction | ✅ Complete, needs tuning |
| `analysis_results/model_improvement_analysis.json` | Analysis results | ✅ Saved |
| `analysis_results/ANALYSIS_REPORT.md` | Analysis report | ✅ Saved |
| `NEXT_STEPS.md` | This file | ✅ Complete |

---

## Files to Create This Week

| File | Purpose | Deadline |
|------|---------|----------|
| `run_pipeline_v2.py` | Enhanced pipeline with epitope prediction | Day 1 |
| `create_benchmark.py` | Download + process benchmark data | Day 2 |
| `benchmark/sars_cov2_known_antibodies.csv` | Test dataset | Day 2 |
| `benchmark_model.py` | Test model performance | Day 2 |
| `WEEK1_RESULTS.md` | Week summary + metrics | Day 5 |

---

## Questions to Answer This Week

1. **Epitope Prediction**:
   - What threshold works best for SARS-CoV-2?
   - How many epitopes to generate antibodies for?

2. **Model Performance**:
   - Can model generate antibodies similar to known ones?
   - What's the affinity prediction error?

3. **Synthesis Decision**:
   - Are computational metrics good enough?
   - Which 3 candidates to synthesize first?

---

## Budget Tracker

### Spent So Far: $0
- Week 1: Computational only

### Week 2 Decision Point: $0-2000
- If metrics good → Order 3 antibodies ($600-1200 each)
- If metrics poor → Continue improvement ($0)

### Total Budget: ~$11,000
- Pilot experiment (6 antibodies): $5,700
- Full validation (3 viruses): $3,000
- Contingency: $2,300

---

## Contact for Help

**Stuck on**:
- Epitope prediction → Check IEDB documentation
- Model testing → Review benchmark creation script
- Synthesis decision → Review decision criteria in strategy doc

**Resources**:
- Full strategy: `MODEL_IMPROVEMENT_STRATEGY.md`
- Current results: `STRUCTURE_VALIDATION_REPORT.md`
- Pipeline summary: `FINAL_RESULTS_SUMMARY.md`

---

**Last Updated**: 2025-01-15
**Next Review**: End of Day 1 (after threshold tuning)
**Status**: 🟢 On track, proceeding with Day 1 tasks
