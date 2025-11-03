"""
Generators for creating novel antibody sequences

Currently implemented:
- TemplateGenerator: Mutate CDR regions of known templates
- AminoAcidTokenizer: Tokenizer for protein sequences
- Seq2SeqGenerator: Transformer-based sequence generation

Future implementations:
- DiffAbGenerator: Antigen-conditioned diffusion models
- IgLMGenerator: Language model-based generation
"""

# Note: Imports are done lazily to avoid requiring all dependencies
# Import specific classes when needed:
#   from generators.template_generator import TemplateGenerator
#   from generators.tokenizer import AminoAcidTokenizer
#   from generators.seq2seq_generator import Seq2SeqGenerator

__all__ = []
