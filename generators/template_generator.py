"""
Template-Based Antibody Generator

Generates antibody variants by mutating CDR regions of known templates.
Simple but effective approach - works immediately without external dependencies.

Usage:
    from generators.template_generator import TemplateGenerator

    gen = TemplateGenerator()
    candidates = gen.generate(n_candidates=100)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
import random


class TemplateGenerator:
    """
    Generate antibody variants from templates by CDR mutations

    This is a simple but proven approach used in industry.
    """

    def __init__(self, template_library: str = 'data/templates/template_custom.csv'):
        """
        Initialize generator with template library

        Args:
            template_library: Path to CSV with antibody templates
        """
        self.template_path = Path(template_library)

        if not self.template_path.exists():
            print(f"⚠️  Template library not found: {template_library}")
            print(f"   Using built-in default templates")
            self.templates = self._get_default_templates()
        else:
            self.templates = pd.read_csv(self.template_path)

        print(f"✅ Loaded {len(self.templates)} antibody templates")

        # Amino acid alphabet
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

        # CDR regions (approximate positions for human antibodies)
        # Heavy chain CDRs
        self.cdr_h1 = (31, 35)  # CDR-H1
        self.cdr_h2 = (50, 65)  # CDR-H2
        self.cdr_h3 = (95, 102) # CDR-H3 (most important for binding!)

        # Light chain CDRs
        self.cdr_l1 = (24, 34)  # CDR-L1
        self.cdr_l2 = (50, 56)  # CDR-L2
        self.cdr_l3 = (89, 97)  # CDR-L3

    def _get_default_templates(self) -> pd.DataFrame:
        """Built-in default antibody templates"""
        return pd.DataFrame([
            {
                'id': 'template_1',
                'name': 'Generic IgG1',
                'heavy_chain': 'EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDIQYGNYYYGMDVWGQGTTVTVSS',
                'light_chain': 'DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK'
            },
            {
                'id': 'template_2',
                'name': 'High-affinity variant',
                'heavy_chain': 'QVQLVQSGAEVKKPGASVKVSCKASGYTFTNYYMHWVRQAPGQGLEWMGWINPNSGGTNYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYCARDGYHGSWFAYWGQGTLVTVSS',
                'light_chain': 'EIVLTQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLIYGASSRATGIPDRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGSSPLTFGGGTKVEIK'
            },
            {
                'id': 'template_3',
                'name': 'Stabilized framework',
                'heavy_chain': 'DVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSGISWNSGSIGYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKDRSTGWSEYFDYWGQGTLVTVSS',
                'light_chain': 'DIQLTQSPSFLSASVGDRVTITCRASQGISSALAWYQQKPGKAPKLLIYDASSLESGVPSRFSGSGSGTEFTLTISSLQPEDFATYYCQQFNSYPLTFGQGTKVEIK'
            }
        ])

    def generate(self,
                 n_candidates: int = 100,
                 mutations_per_variant: int = 3,
                 focus_on_cdr3: bool = True) -> List[Dict]:
        """
        Generate antibody variants

        Args:
            n_candidates: Number of variants to generate
            mutations_per_variant: Number of amino acid mutations per variant
            focus_on_cdr3: If True, 70% of mutations in CDR-H3 (most important)

        Returns:
            List of antibody candidate dictionaries
        """
        print(f"🧬 Generating {n_candidates} antibody variants...")
        print(f"   Mutations per variant: {mutations_per_variant}")
        print(f"   Focus on CDR-H3: {focus_on_cdr3}")

        candidates = []
        variants_per_template = n_candidates // len(self.templates)

        for idx, template in self.templates.iterrows():
            heavy = template.get('heavy_chain', '')
            light = template.get('light_chain', '')
            template_id = template.get('id', f'template_{idx}')

            for i in range(variants_per_template):
                # Mutate sequences
                mutant_heavy = self._mutate_sequence(
                    heavy,
                    n_mutations=mutations_per_variant,
                    focus_cdr3=focus_on_cdr3,
                    chain='heavy'
                )

                mutant_light = self._mutate_sequence(
                    light,
                    n_mutations=mutations_per_variant // 2,  # Fewer mutations in light chain
                    focus_cdr3=False,
                    chain='light'
                ) if light else ''

                candidates.append({
                    'id': f'{template_id}_v{i+1}',
                    'antibody_heavy': mutant_heavy,
                    'antibody_light': mutant_light,
                    'full_sequence': mutant_heavy + 'XXX' + mutant_light if mutant_light else mutant_heavy,
                    'template_id': template_id,
                    'source': 'template_mutation',
                    'n_mutations': mutations_per_variant
                })

        print(f"✅ Generated {len(candidates)} variants from {len(self.templates)} templates")

        return candidates

    def _mutate_sequence(self,
                         sequence: str,
                         n_mutations: int,
                         focus_cdr3: bool,
                         chain: str = 'heavy') -> str:
        """
        Introduce mutations in CDR regions

        Args:
            sequence: Original sequence
            n_mutations: Number of mutations to introduce
            focus_cdr3: Focus 70% of mutations in CDR3
            chain: 'heavy' or 'light'

        Returns:
            Mutated sequence
        """
        if not sequence or len(sequence) == 0:
            return sequence

        seq_list = list(sequence)
        seq_len = len(sequence)

        # Define CDR regions based on chain type
        if chain == 'heavy':
            cdr_regions = [self.cdr_h1, self.cdr_h2, self.cdr_h3]
            cdr3_idx = 2
        else:
            cdr_regions = [self.cdr_l1, self.cdr_l2, self.cdr_l3]
            cdr3_idx = 2

        mutations_made = 0

        # If focus_cdr3, do 70% of mutations in CDR3
        if focus_cdr3:
            cdr3_mutations = int(n_mutations * 0.7)
            other_mutations = n_mutations - cdr3_mutations

            # Mutate CDR3
            mutations_made += self._mutate_region(
                seq_list,
                cdr_regions[cdr3_idx],
                cdr3_mutations,
                seq_len
            )

            # Mutate other CDRs
            for i, region in enumerate(cdr_regions):
                if i != cdr3_idx:
                    mutations_made += self._mutate_region(
                        seq_list,
                        region,
                        other_mutations // 2,
                        seq_len
                    )
        else:
            # Distribute mutations evenly across all CDRs
            mutations_per_cdr = n_mutations // len(cdr_regions)

            for region in cdr_regions:
                mutations_made += self._mutate_region(
                    seq_list,
                    region,
                    mutations_per_cdr,
                    seq_len
                )

        return ''.join(seq_list)

    def _mutate_region(self,
                       seq_list: List[str],
                       region: tuple,
                       n_mutations: int,
                       seq_len: int) -> int:
        """
        Mutate a specific region of the sequence

        Args:
            seq_list: Sequence as list of characters (modified in-place)
            region: (start, end) tuple
            n_mutations: Number of mutations
            seq_len: Total sequence length

        Returns:
            Number of mutations actually made
        """
        start, end = region

        # Check if region is valid
        if start >= seq_len or end > seq_len:
            return 0

        mutations_made = 0

        for _ in range(n_mutations):
            # Pick random position in region
            pos = random.randint(start, min(end - 1, seq_len - 1))

            # Get current amino acid
            original = seq_list[pos]

            # Pick different amino acid
            new_aa = random.choice([aa for aa in self.amino_acids if aa != original])

            # Mutate
            seq_list[pos] = new_aa
            mutations_made += 1

        return mutations_made

    def generate_smart_variants(self,
                                antibody_seq: str,
                                n_variants: int = 10,
                                conservative: bool = True) -> List[Dict]:
        """
        Generate smart variants of a specific antibody

        Args:
            antibody_seq: Starting antibody sequence (heavy + XXX + light)
            n_variants: Number of variants to generate
            conservative: If True, use conservative amino acid substitutions

        Returns:
            List of variant dictionaries
        """
        # Split heavy and light chains
        if 'XXX' in antibody_seq:
            heavy, light = antibody_seq.split('XXX')
        else:
            heavy, light = antibody_seq, ''

        variants = []

        for i in range(n_variants):
            # Conservative mutations use similar amino acids
            if conservative:
                mutant_heavy = self._conservative_mutate(heavy, n_mutations=2, chain='heavy')
                mutant_light = self._conservative_mutate(light, n_mutations=1, chain='light') if light else ''
            else:
                mutant_heavy = self._mutate_sequence(heavy, n_mutations=3, focus_cdr3=True, chain='heavy')
                mutant_light = self._mutate_sequence(light, n_mutations=2, focus_cdr3=False, chain='light') if light else ''

            variants.append({
                'id': f'smart_variant_{i+1}',
                'antibody_heavy': mutant_heavy,
                'antibody_light': mutant_light,
                'full_sequence': mutant_heavy + 'XXX' + mutant_light if mutant_light else mutant_heavy,
                'source': 'smart_mutation',
                'conservative': conservative
            })

        return variants

    def _conservative_mutate(self, sequence: str, n_mutations: int, chain: str) -> str:
        """
        Make conservative mutations (similar amino acids)

        Conservative groups:
        - Small: G, A, S
        - Hydrophobic: V, L, I, M
        - Aromatic: F, Y, W
        - Positive: K, R, H
        - Negative: D, E
        - Polar: N, Q, S, T
        - Special: C, P
        """
        conservative_groups = {
            'G': ['A', 'S'], 'A': ['G', 'S'], 'S': ['G', 'A', 'T'],
            'V': ['L', 'I'], 'L': ['V', 'I', 'M'], 'I': ['V', 'L'], 'M': ['L'],
            'F': ['Y'], 'Y': ['F', 'W'], 'W': ['Y'],
            'K': ['R'], 'R': ['K', 'H'], 'H': ['R'],
            'D': ['E'], 'E': ['D'],
            'N': ['Q'], 'Q': ['N'],
            'T': ['S'],
            'C': ['S'],
            'P': ['G']
        }

        seq_list = list(sequence)
        seq_len = len(sequence)

        # Get CDR regions
        if chain == 'heavy':
            cdrs = [self.cdr_h1, self.cdr_h2, self.cdr_h3]
        else:
            cdrs = [self.cdr_l1, self.cdr_l2, self.cdr_l3]

        mutations_made = 0

        while mutations_made < n_mutations:
            # Pick random CDR
            cdr_start, cdr_end = random.choice(cdrs)

            if cdr_start >= seq_len:
                continue

            # Pick position in CDR
            pos = random.randint(cdr_start, min(cdr_end - 1, seq_len - 1))
            original = seq_list[pos]

            # Get conservative substitutions
            if original in conservative_groups and conservative_groups[original]:
                new_aa = random.choice(conservative_groups[original])
                seq_list[pos] = new_aa
                mutations_made += 1

        return ''.join(seq_list)


def main():
    """Example usage"""
    print("="*70)
    print("Template-Based Antibody Generator")
    print("="*70)

    # Initialize
    generator = TemplateGenerator()

    # Generate 10 candidates
    candidates = generator.generate(n_candidates=10, mutations_per_variant=3)

    print("\nGenerated Candidates:")
    for i, cand in enumerate(candidates[:5]):  # Show first 5
        print(f"\n{i+1}. {cand['id']}")
        print(f"   Heavy: {cand['antibody_heavy'][:50]}...")
        print(f"   Light: {cand['antibody_light'][:50]}...")
        print(f"   Template: {cand['template_id']}")

    print("\n" + "="*70)
    print(f"✅ Generated {len(candidates)} antibody candidates")
    print("="*70)


if __name__ == '__main__':
    main()
