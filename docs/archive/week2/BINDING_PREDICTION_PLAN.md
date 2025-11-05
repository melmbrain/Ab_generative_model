# Binding Prediction Setup Plan

**Date**: 2025-11-05
**Objective**: Add antibody-antigen binding prediction to pipeline
**Time Estimate**: 2-4 hours

---

## 🎯 GOALS

### Primary Goal
Predict binding quality for generated antibody-epitope pairs

### Success Criteria
- [ ] Can predict antibody-antigen complex structure
- [ ] Extract binding quality metrics (pLDDT, pTM, interface quality)
- [ ] Test on 2-3 generated antibodies
- [ ] Create simple scoring system

---

## 🔬 TOOL OPTIONS ANALYSIS

### Option 1: ColabFold (RECOMMENDED)

**Pros**:
- ✅ Faster than full AlphaFold (~10x)
- ✅ Can run on Google Colab (free GPU)
- ✅ Good accuracy for antibody-antigen complexes
- ✅ Outputs pLDDT (confidence) and pTM (quality)
- ✅ Well-documented for antibodies

**Cons**:
- ❌ Requires Google Colab or complex local setup
- ❌ Each prediction takes 2-5 minutes
- ❌ Colab has usage limits

**Best for**: Our current needs (screening 3-6 candidates)

### Option 2: ESMFold

**Pros**:
- ✅ Very fast (~seconds per protein)
- ✅ Easy to install locally
- ✅ Outputs pLDDT scores
- ✅ We already have experience with it

**Cons**:
- ❌ Less accurate for complexes
- ❌ Designed for single chains, not complexes
- ❌ No pTM score

**Best for**: Quick screening, individual chains

### Option 3: AlphaFold-Multimer (Full)

**Pros**:
- ✅ Most accurate
- ✅ Best for antibody-antigen complexes
- ✅ Published benchmarks available

**Cons**:
- ❌ Very slow (30-60 min per complex)
- ❌ Requires massive databases (>2TB)
- ❌ Complex installation

**Best for**: Final validation before synthesis (not screening)

---

## 📋 RECOMMENDED APPROACH

### Phase 1: Quick Implementation (TODAY - Day 2 Afternoon)

**Use ESMFold for initial screening**

**Why**:
- Already have ESMFold experience from Week 1
- Very fast (can test 20 antibodies in minutes)
- Good enough for filtering out bad candidates
- Can score:
  - Individual antibody structure quality (pLDDT)
  - Sequence-structure compatibility
  - Basic validity

**Implementation**:
1. Use ESMFold to validate antibody structures (already tested)
2. Score based on pLDDT of CDR regions
3. Add simple epitope-antibody sequence analysis
4. Create ranking system

**Time**: 1-2 hours

### Phase 2: Proper Binding Prediction (Day 3 or later)

**Setup ColabFold via Google Colab**

**Why**:
- More accurate binding prediction
- No local installation needed
- Free GPU access
- Good for 3-6 final candidates

**Implementation**:
1. Create Google Colab notebook
2. Format antibody-epitope pairs for ColabFold
3. Run predictions on Colab
4. Download and parse results
5. Extract interface pLDDT and pTM scores

**Time**: 2-3 hours

---

## 🚀 PHASE 1 IMPLEMENTATION (TODAY)

### Step 1: Enhanced Structure Validation (30 min)

Create `enhanced_structure_validator.py`:

```python
from validation.structure_validation import IgFoldValidator
import numpy as np

class EnhancedStructureValidator:
    """
    Enhanced validator with binding site analysis
    """

    def __init__(self):
        self.igfold = IgFoldValidator()

    def score_binding_potential(self, heavy_chain, light_chain, epitope):
        """
        Score binding potential based on:
        1. CDR structure quality
        2. Sequence complementarity
        3. Overall structure confidence
        """
        # Validate structure
        result = self.igfold.validate(heavy_chain, light_chain)

        # Score CDR regions (higher pLDDT = better)
        cdr_score = self.score_cdr_quality(result)

        # Score sequence complementarity
        complementarity = self.score_complementarity(heavy_chain, light_chain, epitope)

        # Combined score
        binding_score = 0.5 * cdr_score + 0.3 * complementarity + 0.2 * (result.plddt_heavy / 100)

        return {
            'binding_score': binding_score,
            'cdr_score': cdr_score,
            'complementarity': complementarity,
            'structure_confidence': result.plddt_heavy,
            'is_valid': result.is_valid
        }
```

### Step 2: Sequence-Based Binding Prediction (30 min)

Create `sequence_binding_scorer.py`:

```python
class SequenceBindingScorer:
    """
    Predict binding potential from sequences
    (Fast, no structure prediction needed)
    """

    def score_epitope_antibody_match(self, heavy, light, epitope):
        """
        Score based on:
        1. CDR3 length compatibility
        2. Charge complementarity
        3. Hydrophobicity matching
        """
        # Extract CDR3 regions (approximate)
        cdr3_heavy = self.extract_cdr3_heavy(heavy)
        cdr3_light = self.extract_cdr3_light(light)

        # Score charge complementarity
        charge_score = self.score_charges(cdr3_heavy, cdr3_light, epitope)

        # Score hydrophobicity
        hydro_score = self.score_hydrophobicity(cdr3_heavy, cdr3_light, epitope)

        # Combined
        return (charge_score + hydro_score) / 2
```

### Step 3: Simple Ranking System (30 min)

```python
def rank_antibodies(antibodies, epitopes):
    """
    Rank antibodies by combined score:
    1. Epitope prediction score (30%)
    2. Structure quality (30%)
    3. Sequence binding score (20%)
    4. Diversity (20%)
    """
    scores = []

    for ab in antibodies:
        score = (
            0.3 * ab['epitope_score'] +
            0.3 * ab['structure_plddt'] / 100 +
            0.2 * ab['binding_score'] +
            0.2 * ab['diversity_score']
        )
        scores.append((ab, score))

    # Sort by score descending
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    return ranked
```

---

## 🎯 PHASE 2 IMPLEMENTATION (Day 3)

### ColabFold via Google Colab

**Notebook Setup**:

```python
# Install ColabFold
!pip install -q colabfold

# Prepare antibody-epitope complex
antibody = f"{heavy_chain}:{light_chain}"
complex_sequence = f"{antibody}:{epitope}"

# Run ColabFold
from colabfold.batch import predict_structure

results = predict_structure(
    sequence=complex_sequence,
    job_name="antibody_epitope",
    num_models=1,  # Fast version
    use_templates=False
)

# Extract metrics
plddt = results['plddt']
ptm = results['ptm']
interface_plddt = calculate_interface_plddt(results, antibody, epitope)

print(f"pTM: {ptm:.3f}")
print(f"Interface pLDDT: {interface_plddt:.1f}")
print(f"Binding quality: {'Good' if ptm > 0.7 and interface_plddt > 70 else 'Poor'}")
```

**Integration**:
- Upload antibody sequences to Colab
- Run predictions
- Download results
- Parse and integrate scores

---

## 📊 SCORING SYSTEM

### Combined Ranking Score

```
Total Score =
    0.25 × Epitope Score (0-1) +
    0.25 × Structure Quality (pLDDT/100) +
    0.25 × Binding Score (sequence-based) +
    0.15 × Diversity Score +
    0.10 × Similarity Score
```

### Thresholds for Synthesis

**Must Have**:
- Epitope score > 0.65
- Structure pLDDT > 70
- Binding score > 0.5
- Diversity: Not duplicate

**Nice to Have**:
- Epitope validated with citations
- pRMSD < 2.0 Å
- Binding score > 0.7

---

## ⏱️ TIME ESTIMATES

### Phase 1 (Today - Day 2 Afternoon)

| Task | Time | Priority |
|------|------|----------|
| Enhanced structure validator | 30 min | HIGH |
| Sequence binding scorer | 30 min | HIGH |
| Simple ranking system | 30 min | HIGH |
| Test on 5-10 antibodies | 30 min | HIGH |
| **TOTAL** | **2 hours** | |

### Phase 2 (Day 3 or later)

| Task | Time | Priority |
|------|------|----------|
| Setup Google Colab | 30 min | MEDIUM |
| Create ColabFold notebook | 1 hour | MEDIUM |
| Test on 2-3 complexes | 1 hour | MEDIUM |
| Parse results | 30 min | MEDIUM |
| **TOTAL** | **3 hours** | |

---

## 🎯 DELIVERABLES

### Today (Phase 1)
- [ ] `enhanced_structure_validator.py`
- [ ] `sequence_binding_scorer.py`
- [ ] `rank_antibodies.py`
- [ ] Test results on 10 antibodies
- [ ] `BINDING_SCORING_RESULTS.md`

### Day 3 (Phase 2)
- [ ] `colabfold_binding_prediction.ipynb` (Colab notebook)
- [ ] `parse_colabfold_results.py`
- [ ] Integration into pipeline v4
- [ ] Test on top 3-6 candidates

---

## 💡 DECISION: PHASE 1 TODAY

### Rationale

**Pros of starting with Phase 1**:
1. ✅ Can implement TODAY (2 hours)
2. ✅ Uses existing tools (ESMFold)
3. ✅ Good enough for initial screening
4. ✅ Can test on many candidates quickly
5. ✅ Provides ranking for synthesis decision

**Pros of waiting for Phase 2**:
1. ✅ More accurate binding prediction
2. ✅ Industry-standard tool (ColabFold)
3. ❌ But takes longer to setup
4. ❌ Slower per prediction
5. ❌ Only needed for final 3-6 candidates

**Decision**: **Implement Phase 1 today, Phase 2 on Day 3**

### Plan

**Today (Day 2 Afternoon)**:
1. Implement sequence-based binding scoring
2. Enhance structure validation
3. Create ranking system
4. Test on 10-20 antibodies
5. Select top candidates

**Day 3**:
1. Setup ColabFold for top 3-6 candidates
2. Run proper binding prediction
3. Validate scoring system
4. Make final synthesis decision

---

## ✅ SUCCESS CRITERIA (Day 2 Afternoon)

- [ ] Can score antibodies for binding potential
- [ ] Tested on at least 10 antibodies
- [ ] Clear ranking system implemented
- [ ] Top 3-6 candidates identified
- [ ] Documentation complete

---

## 🎓 FALLBACK PLAN

### If Phase 1 Takes Too Long

**Minimal viable approach**:
1. Use only structure quality (pLDDT from ESMFold)
2. Rank by: epitope score + structure quality
3. Select top 3-6 for synthesis
4. Validate binding experimentally

**Time**: 1 hour

### If We Need More Accuracy

**Alternative**:
1. Use online AlphaFold servers
2. Submit top 3 candidates manually
3. Wait for results (few hours)
4. Make decision based on results

**Time**: Variable (server-dependent)

---

## 🎯 BOTTOM LINE

### Today's Plan: PHASE 1 IMPLEMENTATION

**Goal**: Create sequence-based binding scoring system

**Tasks** (2 hours):
1. Enhance structure validation with CDR analysis
2. Implement sequence binding scorer
3. Create ranking system
4. Test on 10-20 antibodies

**Expected Output**:
- Ranked list of antibodies
- Top 3-6 candidates for synthesis
- Binding scores for all candidates

**Next Steps** (Day 3):
- Setup ColabFold for final validation
- Run binding prediction on top candidates
- Make synthesis decision

---

**Status**: 🟡 **READY TO START PHASE 1**
**Time Available**: 2-3 hours
**Confidence**: High (using proven tools)

🚀 **Let's implement Phase 1 binding scoring!**
