# Week 2 - Getting Started Guide

**Date**: 2025-11-05
**Status**: Ready to begin
**Duration**: 5 days planned

---

## Week 1 Results Summary

### ✅ Achievements

**Sequence Recovery Test** (50 antibodies):
- **Overall Similarity**: 50.4% ✅ (target: ≥40%)
- **Heavy Chain**: 61.5% similarity, 42.2% identity ✅ Excellent
- **Light Chain**: 39.3% similarity, 23.1% identity ⚠️ Needs work
- **Validity**: 100% (all sequences valid) ✅ Perfect

**Benchmark**:
- 10,522 SARS-CoV-2 antibodies (CoV-AbDab)
- 67.9% target RBD
- All with complete sequences

**Pipeline**:
- Epitope predictor v2: 50% recall
- Pipeline v2: Fully functional
- Integration tests: 5/5 passing

### 🔍 Critical Issue Identified

**Light Chain Length Problem**:
```
Real (benchmark):     108.6 aa average
Generated (model):    177.0 aa average
Training data:        201.0 aa average
Error:                +68.4 aa (63% too long!)
```

**Root Cause**: Training data format mismatch
- Training data likely has full-length light chains
- Benchmark (CoV-AbDab) has V-regions only
- Model learned to generate full-length format

**Impact**:
- Reduces light chain similarity scores
- Not a validity issue (sequences are valid)
- Affects overall recovery metric
- **Must fix before synthesis**

---

## Week 2 Goals

### Primary Objectives

1. **Investigate & Fix Light Chain Issue** (Priority 1)
   - Analyze training data format
   - Determine if reprocessing or fine-tuning needed
   - Target: <20 aa length error

2. **Add Binding Prediction** (Priority 2)
   - Integrate AlphaFold-Multimer or ColabFold
   - Predict antibody-antigen binding
   - Filter candidates by interface pLDDT (>70)

3. **Prepare Synthesis Candidates** (Priority 3)
   - Select top 3-6 antibodies
   - Create comprehensive validation report
   - **Decision**: Order synthesis or iterate?

### Success Criteria

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Light chain error | <20 aa | 56.8 aa | ❌ Must fix |
| Overall similarity | >40% | 50.4% | ✅ Passing |
| Binding prediction | Implemented | N/A | 📋 TODO |
| Synthesis candidates | 3-6 ready | 0 | 📋 TODO |

---

## Day-by-Day Plan

### Day 1 (Today): Investigate Light Chain Issue

**Time**: 2-3 hours

**Tasks**:
1. Analyze training data light chain distribution
2. Check if format is full-length vs V-region
3. Compare with benchmark format
4. Identify solution (reprocess, fine-tune, or adjust generation)

**Script to Create**: `analyze_light_chain_issue.py`

**Deliverable**: `LIGHT_CHAIN_ANALYSIS.md` with findings and recommendation

---

### Day 2: Fix Light Chain Issue

**Time**: 2-4 hours (depends on Day 1 findings)

**Option A**: If Training Data Needs Reprocessing
- Extract V-regions from training data
- Regenerate training statistics
- May need retraining (long, skip for now)

**Option B**: If Model Can Be Adjusted
- Modify generation parameters
- Test truncation post-generation
- Validate on benchmark subset

**Option C**: If Issue is Minor
- Document for synthesis selection
- Focus on heavy chain quality
- Proceed with binding prediction

**Deliverable**: Solution implemented and tested

---

### Day 3: Setup Binding Prediction

**Time**: 3-4 hours

**Tool**: ColabFold (easier than full AlphaFold-Multimer)

**Tasks**:
1. Install ColabFold locally OR use Google Colab
2. Create `binding_predictor.py` wrapper
3. Test on 5 generated antibody-epitope pairs
4. Parse output (pLDDT, pTM, interface quality)

**Deliverable**: `binding_predictor.py` working

---

### Day 4: Integrate & Test Binding Prediction

**Time**: 2-3 hours

**Tasks**:
1. Integrate into pipeline v3
2. Test on top 20 generated antibodies
3. Rank by combined metrics:
   - Epitope score (>0.65)
   - Sequence recovery (>0.50)
   - Structure quality (pRMSD <2.0)
   - **Binding prediction (pLDDT >70)** ← NEW

**Deliverable**: Pipeline v3 with binding prediction

---

### Day 5: Select Synthesis Candidates

**Time**: 2-3 hours

**Tasks**:
1. Run full pipeline on SARS-CoV-2 spike
2. Generate 10-20 antibody candidates
3. Apply all filters and rank
4. Select top 3-6 for synthesis
5. Create synthesis report

**Deliverable**: `SYNTHESIS_CANDIDATES.md`

**Decision Point**: Order synthesis (~$1,800-3,600) or iterate Week 3?

---

## Starting with Day 1

Let's begin by investigating the light chain issue. Here's the first script:

### Script 1: Analyze Light Chain Issue

```python
# analyze_light_chain_issue.py
"""
Investigate why generated light chains are too long

Compares:
1. Training data light chain lengths
2. Benchmark light chain lengths
3. Generated light chain lengths

Checks:
- Full-length vs V-region format
- Constant region presence
- Distribution analysis
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Load training data stats
train_stats_file = Path('data/generative/data_stats.json')
with open(train_stats_file) as f:
    train_stats = json.load(f)

print("Training Data Stats:")
print(f"  Light chain length: {train_stats['antibody_light_length']}")

# Load sample training data
train_file = Path('data/generative/train.json')
with open(train_file) as f:
    train_data = json.load(f)

# Analyze first 1000 training examples
sample_size = min(1000, len(train_data))
train_light_lengths = [len(train_data[i]['antibody_light'])
                       for i in range(sample_size)]

print(f"\nTraining Data Sample (n={sample_size}):")
print(f"  Mean: {np.mean(train_light_lengths):.1f} aa")
print(f"  Std:  {np.std(train_light_lengths):.1f} aa")
print(f"  Min:  {np.min(train_light_lengths)} aa")
print(f"  Max:  {np.max(train_light_lengths)} aa")

# Load benchmark data
benchmark_file = Path('benchmark/benchmark_dataset.json')
with open(benchmark_file) as f:
    benchmark_data = json.load(f)

benchmark_light_lengths = [ab['light_length'] for ab in benchmark_data]

print(f"\nBenchmark Data (CoV-AbDab, n={len(benchmark_data)}):")
print(f"  Mean: {np.mean(benchmark_light_lengths):.1f} aa")
print(f"  Std:  {np.std(benchmark_light_lengths):.1f} aa")
print(f"  Min:  {np.min(benchmark_light_lengths)} aa")
print(f"  Max:  {np.max(benchmark_light_lengths)} aa")

# Load test results
results_file = Path('benchmark/sequence_recovery_results.json')
with open(results_file) as f:
    results = json.load(f)

generated_light_lengths = [r['generated_light_length']
                          for r in results['individual_results']]

print(f"\nGenerated Data (Test Results, n={len(generated_light_lengths)}):")
print(f"  Mean: {np.mean(generated_light_lengths):.1f} aa")
print(f"  Std:  {np.std(generated_light_lengths):.1f} aa")
print(f"  Min:  {np.min(generated_light_lengths)} aa")
print(f"  Max:  {np.max(generated_light_lengths)} aa")

# Analysis
print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

print(f"\nLength Comparison:")
print(f"  Training:   {np.mean(train_light_lengths):.1f} aa")
print(f"  Generated:  {np.mean(generated_light_lengths):.1f} aa")
print(f"  Benchmark:  {np.mean(benchmark_light_lengths):.1f} aa")

print(f"\nDifferences:")
print(f"  Generated - Benchmark: {np.mean(generated_light_lengths) - np.mean(benchmark_light_lengths):.1f} aa")
print(f"  Training - Benchmark:  {np.mean(train_light_lengths) - np.mean(benchmark_light_lengths):.1f} aa")

# Check for constant region
# Typical V-region: ~110 aa
# Full-length with C-region: ~210 aa
V_REGION_LENGTH = 110
FULL_LENGTH = 210

print(f"\nFormat Detection:")
if np.mean(train_light_lengths) > 180:
    print(f"  Training data: FULL-LENGTH (with constant region)")
elif np.mean(train_light_lengths) < 130:
    print(f"  Training data: V-REGION only")
else:
    print(f"  Training data: MIXED or INTERMEDIATE")

if np.mean(benchmark_light_lengths) > 180:
    print(f"  Benchmark: FULL-LENGTH")
elif np.mean(benchmark_light_lengths) < 130:
    print(f"  Benchmark: V-REGION only")
else:
    print(f"  Benchmark: MIXED")

print(f"\n{'='*80}")
print("RECOMMENDATION")
print("="*80)

if abs(np.mean(train_light_lengths) - np.mean(benchmark_light_lengths)) > 50:
    print("\n⚠️  FORMAT MISMATCH DETECTED")
    print(f"\nTraining data appears to use a different format than benchmark.")
    print(f"Difference: {abs(np.mean(train_light_lengths) - np.mean(benchmark_light_lengths)):.1f} aa")

    print(f"\nOptions:")
    print(f"1. Reprocess training data to match benchmark format")
    print(f"2. Truncate generated sequences at V-region boundary (~110 aa)")
    print(f"3. Fine-tune model on V-region-only data")
    print(f"4. Focus on heavy chain for synthesis (light chains less critical)")

    print(f"\nRecommended: Option 2 (truncate at ~110 aa) for quick fix")
else:
    print("\n✅ Formats appear similar")
    print(f"Length difference may be due to other factors")
```

**Run this first**:
```bash
python analyze_light_chain_issue.py
```

This will tell us exactly what the problem is.

---

## Alternative: If Time is Limited

### Fast Track to Week 2 Completion

If the light chain issue takes too long:

**Plan B: Focus on Heavy Chains Only**
1. Heavy chains are performing well (42% identity)
2. Many therapeutic antibodies use single-domain formats
3. Can proceed with synthesis focusing on heavy chain quality
4. Light chain issue can be addressed in future iterations

**Advantages**:
- Faster path to synthesis
- Heavy chain is the primary CDR3-containing chain
- Still valuable for validation
- Can iterate on light chains later

---

## Resources & Files

### Key Files to Review

1. **Training Data**:
   - `data/generative/train.json` - Training sequences
   - `data/generative/data_stats.json` - Statistics

2. **Benchmark**:
   - `benchmark/benchmark_dataset.json` - 10,522 antibodies
   - `benchmark/BENCHMARK_REPORT.md` - Statistics

3. **Test Results**:
   - `benchmark/sequence_recovery_results.json` - 50 test results
   - `benchmark/test_run.log` - Full test log

4. **Pipeline**:
   - `run_pipeline_v2.py` - Current pipeline
   - `epitope_predictor_v2.py` - Epitope prediction

### Week 1 Documentation

- `WEEK1_COMPLETION_SUMMARY.md` - Comprehensive summary
- `DAY1_COMPLETION_SUMMARY.md` - Day 1 details
- `DAY2_PROGRESS_SUMMARY.md` - Day 2 details

---

## Questions to Answer (Day 1)

1. **What is the exact format of training data light chains?**
   - Full-length (with constant region)?
   - V-region only?
   - Mixed?

2. **What is the format of benchmark light chains?**
   - CoV-AbDab uses which format?

3. **Is this fixable without retraining?**
   - Can we truncate generated sequences?
   - Post-processing solution?
   - Or need model adjustment?

4. **Impact on synthesis decision?**
   - Can we proceed with current model?
   - Or need to fix first?

---

## Success Metrics for Week 2

### Must Have (Blockers)
- [ ] Light chain length error <20 aa OR decision to proceed with heavy-only
- [ ] Binding prediction working (at least basic implementation)
- [ ] 3-6 synthesis candidates identified

### Nice to Have
- [ ] Light chain fully fixed
- [ ] Full AlphaFold-Multimer integration
- [ ] 10+ validated candidates

### Decision Criteria (End of Week 2)

**Proceed to Synthesis IF**:
1. ✅ Light chain issue resolved OR documented as acceptable
2. ✅ Binding prediction shows >70% pLDDT for top candidates
3. ✅ At least 3 candidates with:
   - Epitope score >0.65
   - Sequence recovery >0.50
   - Structure validation passing
   - Binding prediction good

**Iterate Week 3 IF**:
- Light chain issue is severe and unfixable quickly
- Binding prediction shows poor results
- Not enough high-quality candidates

---

## Getting Started Now

### Step 1: Create Analysis Script

```bash
# Create the analysis script
cat > analyze_light_chain_issue.py << 'EOF'
[Script content from above]
EOF

# Run it
python3 analyze_light_chain_issue.py
```

### Step 2: Review Results

Based on output, decide:
- Quick fix possible? → Implement Day 1
- Need reprocessing? → Plan for Days 1-2
- Major issue? → Consider Plan B (heavy-chain focus)

### Step 3: Update Plan

After Day 1 analysis:
- Update WEEK2_GETTING_STARTED.md with findings
- Adjust Days 2-5 based on Day 1 results
- Document decision and rationale

---

## Budget Reminder

**Current**: $0 spent (Week 1 was all computational)

**Week 2**: Still $0 (no synthesis yet)

**Synthesis Decision** (end of Week 2):
- 3 antibodies: ~$1,800-3,600
- 6 antibodies: ~$3,600-7,200

**Remaining Budget**: $11,200 total

---

## Need Help?

### Common Issues

**Issue**: Training data format unclear
**Solution**: Check first 10 sequences manually, compare to known antibody structures

**Issue**: Analysis script errors
**Solution**: Check file paths, ensure all data files present

**Issue**: Uncertain about next steps
**Solution**: Review WEEK1_COMPLETION_SUMMARY.md, focus on light chain issue first

### Documentation

- `WEEK1_COMPLETION_SUMMARY.md` - Full Week 1 results
- `PIPELINE_V2_QUICK_START.md` - Pipeline usage guide
- `MODEL_IMPROVEMENT_STRATEGY.md` - 6-month roadmap

---

## Ready to Start?

**Run this to begin Day 1**:

```bash
# 1. Create analysis script
python3 analyze_light_chain_issue.py

# 2. Review output

# 3. Document findings in LIGHT_CHAIN_ANALYSIS.md

# 4. Decide on solution approach

# 5. Proceed to Day 2 implementation
```

---

**Status**: 📋 Ready to begin Week 2 Day 1
**Next**: Analyze light chain issue
**Time Estimate**: 2-3 hours for Day 1

Good luck! 🚀
