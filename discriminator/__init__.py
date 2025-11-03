"""
Discriminator module for scoring antibody-antigen binding affinity

Uses Phase 2 model (Spearman 0.85) trained on 7k Ab-Ag pairs
"""

from .affinity_discriminator import AffinityDiscriminator

__all__ = ['AffinityDiscriminator']
