# Evolution Plan: From Discriminator to Antigen-Conditioned Generator

**Goal**: Enable the model to predict/design antibody sequences when given an antigen sequence

**Current State**: Discriminator (scorer) + Template generator (NOT antigen-conditioned)

**Target State**: Antigen-conditioned antibody generation

---

## Approach 1: Guided Evolutionary Search (IMMEDIATE)

**Timeline**: 1-2 weeks
**Complexity**: Low
**Uses existing**: Discriminator + Template generator

### Implementation

```python
class AntiGenGuidedGenerator:
    """
    Evolutionary search guided by discriminator

    Given antigen → Generate optimized antibodies
    """

    def __init__(self):
        self.discriminator = AffinityDiscriminator()
        self.base_generator = TemplateGenerator()

    def design_for_antigen(self, antigen_seq: str, n_iterations=50):
        # 1. Generate initial population
        population = self.base_generator.generate(n_candidates=100)

        # 2. Evolutionary loop
        for iteration in range(n_iterations):
            # Score all against target antigen
            scores = []
            for ab in population:
                result = self.discriminator.predict_single(
                    ab['full_sequence'],
                    antigen_seq
                )
                scores.append(result['predicted_pKd'])

            # Select top 20%
            top_candidates = select_top_k(population, scores, k=20)

            # Mutate top candidates to create new population
            population = []
            for candidate in top_candidates:
                variants = mutate_around(candidate, n_variants=5)
                population.extend(variants)

        # 3. Return best
        return rank_by_affinity(population, antigen_seq)
```

### Pros
- ✅ Works with existing models
- ✅ No new training needed
- ✅ Interpretable (see evolution process)
- ✅ Can implement in 1-2 weeks

### Cons
- ❌ Slower (requires iterations)
- ❌ May not explore full sequence space
- ❌ Computationally expensive (many discriminator calls)

---

## Approach 2: DiffAb Integration (SHORT-TERM)

**Timeline**: 1-2 months
**Complexity**: Medium
**New dependency**: DiffAb model

### What is DiffAb?

**Paper**: "Antigen-Specific Antibody Design and Optimization with Diffusion-Based Generative Models" (2022)

**What it does**:
- Input: Antigen structure (or sequence)
- Output: CDR-H3 sequences designed to bind the antigen
- Method: Diffusion models (like DALL-E for proteins)

### Integration Strategy

```python
class DiffAbDiscriminatorHybrid:
    """
    DiffAb generates → Discriminator re-ranks
    """

    def __init__(self):
        self.diffab_model = load_diffab()  # External model
        self.discriminator = AffinityDiscriminator()
        self.template_generator = TemplateGenerator()

    def design_for_antigen(self, antigen_seq: str):
        # 1. DiffAb designs CDR-H3 for antigen
        cdr_h3_variants = self.diffab_model.design_cdr(
            antigen=antigen_seq,
            n_designs=100
        )

        # 2. Graft onto antibody frameworks
        full_antibodies = []
        for cdr in cdr_h3_variants:
            ab = self.template_generator.graft_cdr(
                framework='template_2',
                cdr_h3=cdr
            )
            full_antibodies.append(ab)

        # 3. Re-rank with discriminator
        scores = self.discriminator.predict_batch(
            full_antibodies,
            [antigen_seq] * len(full_antibodies)
        )

        # 4. Return top-ranked
        return rank_and_return(full_antibodies, scores)
```

### Setup Required

```bash
# Install DiffAb
git clone https://github.com/luost26/diffab
cd diffab
pip install -r requirements.txt
python setup.py install

# Download pretrained weights
wget https://zenodo.org/record/7340272/files/diffab_models.zip
```

### Pros
- ✅ State-of-the-art CDR design
- ✅ Truly antigen-conditioned
- ✅ Published, validated approach
- ✅ Combines strengths: DiffAb (generation) + Discriminator (ranking)

### Cons
- ❌ Requires DiffAb installation
- ❌ Needs antigen structure (or prediction)
- ❌ More complex pipeline

---

## Approach 3: IgLM Integration (MEDIUM-TERM)

**Timeline**: 2-3 months
**Complexity**: Medium-High
**New dependency**: IgLM language model

### What is IgLM?

**Paper**: "Generative Language Modeling for Antibody Design" (2022)

**What it does**:
- Pre-trained on 500M+ antibody sequences
- Can infill CDR regions given framework
- Can be conditioned on properties (though not directly on antigen)

### Integration Strategy

```python
class IgLMGuidedGenerator:
    """
    IgLM generates → Discriminator guides
    """

    def design_for_antigen(self, antigen_seq: str):
        # 1. Use IgLM to generate diverse CDRs
        antibody_candidates = []
        for i in range(100):
            ab = iglm.infill_cdrs(
                framework='template_2',
                temperature=1.0  # Diversity
            )
            antibody_candidates.append(ab)

        # 2. Score with discriminator
        scores = discriminator.predict_batch(
            antibody_candidates,
            [antigen_seq] * 100
        )

        # 3. Use top scorers as seeds
        top_seeds = select_top_k(antibody_candidates, scores, k=10)

        # 4. Generate more variants around top seeds
        refined = []
        for seed in top_seeds:
            variants = iglm.generate_variants(seed, n=10)
            refined.extend(variants)

        # 5. Final re-ranking
        final_scores = discriminator.predict_batch(refined, antigen_seq)

        return rank_and_return(refined, final_scores)
```

---

## Approach 4: Train Custom Generator (LONG-TERM)

**Timeline**: 3-6 months
**Complexity**: High
**Requires**: ML expertise, GPU resources

### Architecture

```
Encoder-Decoder Transformer

Input:  [Antigen sequence] + [Target pKd]
         ↓
Encoder (ESM-2 or custom Transformer)
         ↓
Latent representation (512 dims)
         ↓
Decoder (Transformer)
         ↓
Output: [Antibody sequence]
```

### Training Data

**Available**: 7,015 Ab-Ag pairs from discriminator training

**Training approach**:
1. Train encoder-decoder on Ab-Ag pairs
2. Loss: Sequence reconstruction + pKd prediction
3. Use discriminator as reward in RL fine-tuning

### Pros
- ✅ Fully custom, optimized for your data
- ✅ End-to-end antigen → antibody
- ✅ Can incorporate other constraints (developability, etc.)

### Cons
- ❌ Requires significant ML expertise
- ❌ Needs GPU resources for training
- ❌ 3-6 months development time
- ❌ Risk: May not outperform existing methods

---

## Recommended Roadmap

### Phase 1: Immediate (1-2 weeks)
**✅ Implement Guided Evolutionary Search**
- Use existing discriminator + template generator
- Iterative optimization for target antigen
- Validate on test antigens

**Deliverable**: Working antigen-conditioned generator

---

### Phase 2: Short-term (1-2 months)
**✅ Integrate DiffAb**
- Install and test DiffAb
- Build hybrid pipeline: DiffAb → Discriminator
- Benchmark against Phase 1 approach

**Deliverable**: State-of-the-art CDR design

---

### Phase 3: Medium-term (3-6 months)
**✅ Compare IgLM vs Custom Model**
- Option A: Integrate IgLM (faster)
- Option B: Train custom generator (more control)
- Benchmark all approaches

**Deliverable**: Production-ready antigen-conditioned system

---

## Implementation Priority

**Start with Approach 1 (Guided Search)** because:
1. ✅ Works immediately with existing code
2. ✅ Proves concept end-to-end
3. ✅ Establishes baseline for comparisons
4. ✅ Low risk, high value

**Then add Approach 2 (DiffAb)** because:
1. ✅ State-of-the-art CDR design
2. ✅ Published, validated method
3. ✅ Discriminator adds value (re-ranking)
4. ✅ Reasonable complexity

**Consider Approach 3/4 later** based on:
- Performance of Approaches 1 & 2
- Resource availability
- Specific application needs

---

## Code Structure for Evolution

```
Ab_generative_model/
├── discriminator/              # Existing
│   └── affinity_discriminator.py
│
├── generators/
│   ├── template_generator.py   # Existing
│   ├── guided_search.py        # NEW (Phase 1)
│   ├── diffab_generator.py     # NEW (Phase 2)
│   ├── iglm_generator.py       # NEW (Phase 3)
│   └── custom_generator.py     # NEW (Phase 3, optional)
│
├── scripts/
│   ├── generate_and_score.py   # Existing
│   └── antigen_to_antibody.py  # NEW - Main script
│
└── models/
    ├── agab_phase2_model.pth   # Existing discriminator
    ├── diffab_weights/         # NEW (Phase 2)
    └── custom_gen_weights/     # NEW (Phase 3)
```

---

## Expected Performance

### Approach 1: Guided Search
- **Generation time**: 5-10 minutes (50 iterations)
- **Quality**: Good (leverages discriminator)
- **Diversity**: Moderate (limited by template space)

### Approach 2: DiffAb + Discriminator
- **Generation time**: 1-2 minutes (100 designs)
- **Quality**: Excellent (state-of-the-art CDR design)
- **Diversity**: High (diffusion model explores well)

### Approach 3: IgLM + Discriminator
- **Generation time**: 2-3 minutes
- **Quality**: Very good (pre-trained on 500M sequences)
- **Diversity**: Very high (language model)

### Approach 4: Custom Generator
- **Generation time**: < 1 second (direct inference)
- **Quality**: Depends on training (unknown)
- **Diversity**: Controllable via sampling temperature

---

## Next Steps

**Want me to implement Approach 1 (Guided Search) for you?**

I can create:
1. `generators/guided_search.py` - Evolutionary search implementation
2. `scripts/antigen_to_antibody.py` - Command-line interface
3. Example usage and benchmarks

This would give you **antigen-conditioned antibody generation** in 1-2 hours of coding.

**Let me know if you want me to proceed!** 🚀
