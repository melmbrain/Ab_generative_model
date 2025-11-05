# Antibody Generation Model v2.0

**Affinity-Conditioned Transformer for Therapeutic Antibody Discovery**

[![Version](https://img.shields.io/badge/version-2.0--synthesis--ready-blue)]()
[![Status](https://img.shields.io/badge/status-synthesis--ready-brightgreen)]()
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![PyTorch 2.5+](https://img.shields.io/badge/pytorch-2.5+-ee4c2c.svg)]()

## 🎯 Latest: 5 SARS-CoV-2 Antibodies Ready for Synthesis!

**v2.0-synthesis-ready** (2025-11-05): Production pipeline generates 5 synthesis-ready antibodies targeting SARS-CoV-2 spike protein with 100% success rate.

**Top 2 Candidates**:
- **Ab_2** (Score: 0.640) - Primary candidate, RBD-adjacent epitope
- **Ab_5** (Score: 0.639) - Secondary candidate, best binding score
- **Estimated Cost**: $1,200-2,400 for synthesis
- **Status**: Ready for experimental validation

📄 See [SYNTHESIS_CANDIDATES_FINAL.md](SYNTHESIS_CANDIDATES_FINAL.md) for full details.

---

## Overview

A complete end-to-end pipeline for therapeutic antibody discovery:
1. **Epitope Prediction** - Identifies immunogenic regions in viral antigens
2. **Antibody Generation** - Transformer model generates diverse, high-affinity antibodies
3. **Binding Prediction** - Fast sequence-based screening
4. **Multi-Criteria Ranking** - Automated candidate selection

**Key Features**:
- 🎯 **100% diversity** - No mode collapse (solved via sampling)
- ✅ **Correct lengths** - 109 aa light chains (V-region only)
- ⚡ **Fast generation** - 5 antibodies in 5 seconds
- 🧬 **High quality** - 92.63 mean pLDDT structure scores
- 📊 **Automated ranking** - Multi-criteria scoring system

---

## Quick Start

### Generate Antibodies for Any Virus

```bash
python3 run_pipeline_v3.py \
  --antigen-file sars_cov2_spike.fasta \
  --virus-name "SARS-CoV-2" \
  --antigen-name "spike protein" \
  --top-k-epitopes 5 \
  --temperature 0.5 \
  --output-dir results/my_antibodies
```

See [QUICK_START.md](QUICK_START.md) for detailed usage guide.

---

## Performance Metrics

### v2.0 (Week 2 - Current)

| Metric | Value | Status |
|--------|-------|--------|
| Generation time | 5.1 seconds (5 antibodies) | ⚡ Fast |
| Diversity | 100% (5/5 unique) | ✅ Perfect |
| Synthesis-ready | 100% (5/5 passing) | ✅ Perfect |
| Epitope scores | 0.697-0.740 | ✅ High |
| Binding scores | 0.599-0.655 | ✅ Good |

**Top Candidate Quality**:
- Overall score: **0.640** (Ab_2)
- Epitope score: **0.727**
- Binding score: **0.632**
- Diversity score: **0.628**

### v1.0 (Week 1 - Training)

| Metric | Value |
|--------|-------|
| Architecture | Transformer Seq2Seq |
| Parameters | 5,616,153 |
| Training Data | 158,337 pairs |
| Final Val Loss | 0.6532 |
| Mean pLDDT | 92.63 ± 15.98 |

---

## What's New in v2.0

### Week 2 Day 1: Critical Fixes
✅ **Diversity Fix**: 6% → 100% diversity via sampling (T=0.5)  
✅ **Light Chain Fix**: 177aa → 109aa (V-region only)

### Week 2 Day 2: Production Pipeline
✅ **Pipeline v3**: Complete end-to-end system  
✅ **Binding Prediction**: Fast sequence-based scoring  
✅ **Multi-Criteria Ranking**: Automated candidate selection  
✅ **5 Synthesis Candidates**: SARS-CoV-2 spike protein

---

## Installation

```bash
# Clone repository
git clone https://github.com/melmbrain/Ab_generative_model.git
cd Ab_generative_model

# Install dependencies
pip install -r requirements.txt
```

---

## Documentation

- **Quick Start**: [QUICK_START.md](QUICK_START.md) - Generate antibodies in minutes
- **Synthesis Candidates**: [SYNTHESIS_CANDIDATES_FINAL.md](SYNTHESIS_CANDIDATES_FINAL.md)
- **Week 2 Summary**: [WEEK2_DAY2_COMPLETE.md](WEEK2_DAY2_COMPLETE.md)
- **Next Steps**: [NEXT_STEPS.md](NEXT_STEPS.md) - Future work

---

## Citation

```bibtex
@software{antibody_generation_v2,
  title={Affinity-Conditioned Transformer for Therapeutic Antibody Discovery},
  version={2.0-synthesis-ready},
  year={2025},
  url={https://github.com/melmbrain/Ab_generative_model}
}
```

---

**Last Updated**: 2025-11-05  
**Version**: 2.0-synthesis-ready  
**Status**: Production-ready with synthesis candidates
