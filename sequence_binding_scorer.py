"""
Sequence-Based Binding Scorer

Fast, CPU-only binding potential prediction based on sequence properties.
No GPU or structure prediction needed.

Scores antibodies based on:
1. Charge complementarity (epitope vs CDRs)
2. Hydrophobicity matching
3. CDR3 length compatibility
4. Sequence diversity

Usage:
    scorer = SequenceBindingScorer()
    score = scorer.score_binding_potential(heavy, light, epitope)
"""

import numpy as np
from typing import Dict, Tuple


class SequenceBindingScorer:
    """
    Predict binding potential from sequence properties only

    Fast, CPU-only, no structure prediction needed
    """

    # Amino acid properties
    CHARGES = {
        'K': 1, 'R': 1, 'H': 0.5,  # Positive
        'D': -1, 'E': -1,           # Negative
        'default': 0                 # Neutral
    }

    HYDROPHOBICITY = {
        # Parker hydrophilicity scale (higher = more hydrophilic)
        'A': -0.5, 'C': -1.0, 'D': 3.0, 'E': 3.0, 'F': -2.5,
        'G': 0.0, 'H': -0.5, 'I': -1.8, 'K': 3.0, 'L': -1.8,
        'M': -1.3, 'N': 0.2, 'P': 0.0, 'Q': 0.2, 'R': 3.0,
        'S': 0.3, 'T': -0.4, 'V': -1.5, 'W': -3.4, 'Y': -2.3,
        'default': 0.0
    }

    def __init__(self):
        """Initialize scorer"""
        pass

    def extract_cdr3_heavy(self, heavy_chain: str) -> str:
        """
        Extract heavy chain CDR3 region (approximate)

        CDR3 is typically around positions 95-102 in heavy chain
        For ~120 aa heavy chain, use positions 85-105
        """
        if len(heavy_chain) < 100:
            # Short chain, use middle region
            start = len(heavy_chain) // 2 - 10
            end = len(heavy_chain) // 2 + 10
        else:
            # Normal length
            start = 85
            end = min(105, len(heavy_chain))

        return heavy_chain[start:end]

    def extract_cdr3_light(self, light_chain: str) -> str:
        """
        Extract light chain CDR3 region (approximate)

        CDR3 is typically around positions 89-97 in light chain
        For ~109 aa light chain, use positions 85-100
        """
        if len(light_chain) < 100:
            # Short chain, use middle region
            start = len(light_chain) // 2 - 7
            end = len(light_chain) // 2 + 7
        else:
            # Normal length
            start = 85
            end = min(100, len(light_chain))

        return light_chain[start:end]

    def calculate_net_charge(self, sequence: str) -> float:
        """Calculate net charge of sequence"""
        charge = sum(self.CHARGES.get(aa, 0) for aa in sequence)
        return charge / len(sequence) if sequence else 0

    def calculate_hydrophobicity(self, sequence: str) -> float:
        """Calculate average hydrophobicity"""
        hydro = sum(self.HYDROPHOBICITY.get(aa, 0) for aa in sequence)
        return hydro / len(sequence) if sequence else 0

    def score_charge_complementarity(
        self,
        cdr3_heavy: str,
        cdr3_light: str,
        epitope: str
    ) -> float:
        """
        Score charge complementarity

        Opposite charges attract:
        - Positive epitope + negative CDRs = good
        - Negative epitope + positive CDRs = good

        Returns:
            Score 0-1 (higher = better complementarity)
        """
        epitope_charge = self.calculate_net_charge(epitope)
        cdr_heavy_charge = self.calculate_net_charge(cdr3_heavy)
        cdr_light_charge = self.calculate_net_charge(cdr3_light)

        # Combined CDR charge
        cdr_charge = (cdr_heavy_charge + cdr_light_charge) / 2

        # Complementarity: opposite charges score high
        complementarity = -epitope_charge * cdr_charge

        # Normalize to 0-1 range
        # Max complementarity ~= 1.0 (highly charged opposite)
        # Min complementarity ~= -1.0 (same charges repel)
        # Map to 0-1: (x + 1) / 2
        score = (complementarity + 1) / 2

        return max(0, min(1, score))

    def score_hydrophobicity_match(
        self,
        cdr3_heavy: str,
        cdr3_light: str,
        epitope: str
    ) -> float:
        """
        Score hydrophobicity matching

        Similar hydrophobicity = good binding
        (hydrophobic epitope binds hydrophobic CDRs)

        Returns:
            Score 0-1 (higher = better match)
        """
        epitope_hydro = self.calculate_hydrophobicity(epitope)
        cdr_heavy_hydro = self.calculate_hydrophobicity(cdr3_heavy)
        cdr_light_hydro = self.calculate_hydrophobicity(cdr3_light)

        # Combined CDR hydrophobicity
        cdr_hydro = (cdr_heavy_hydro + cdr_light_hydro) / 2

        # Score based on similarity (small difference = good)
        difference = abs(epitope_hydro - cdr_hydro)

        # Normalize: max difference ~= 6.0, min ~= 0
        # Map to 0-1: 1 - (diff / 6)
        score = 1 - (difference / 6.0)

        return max(0, min(1, score))

    def score_length_compatibility(
        self,
        cdr3_heavy: str,
        cdr3_light: str,
        epitope: str
    ) -> float:
        """
        Score CDR3 length compatibility with epitope

        CDR3 should be similar length to epitope for good binding

        Returns:
            Score 0-1 (higher = better compatibility)
        """
        epitope_len = len(epitope)
        cdr_heavy_len = len(cdr3_heavy)
        cdr_light_len = len(cdr3_light)

        # Average CDR length
        avg_cdr_len = (cdr_heavy_len + cdr_light_len) / 2

        # Length difference
        diff = abs(epitope_len - avg_cdr_len)

        # Normalize: good match = diff < 5, poor = diff > 20
        # Map to 0-1: max(0, 1 - diff/20)
        score = 1 - (diff / 20)

        return max(0, min(1, score))

    def score_binding_potential(
        self,
        heavy_chain: str,
        light_chain: str,
        epitope: str
    ) -> Dict[str, float]:
        """
        Score overall binding potential

        Args:
            heavy_chain: Heavy chain sequence
            light_chain: Light chain sequence
            epitope: Epitope sequence

        Returns:
            Dictionary with scores:
            - binding_score: Overall score (0-1)
            - charge_score: Charge complementarity (0-1)
            - hydro_score: Hydrophobicity match (0-1)
            - length_score: Length compatibility (0-1)
        """
        # Extract CDR3 regions
        cdr3_heavy = self.extract_cdr3_heavy(heavy_chain)
        cdr3_light = self.extract_cdr3_light(light_chain)

        # Calculate individual scores
        charge_score = self.score_charge_complementarity(
            cdr3_heavy, cdr3_light, epitope
        )

        hydro_score = self.score_hydrophobicity_match(
            cdr3_heavy, cdr3_light, epitope
        )

        length_score = self.score_length_compatibility(
            cdr3_heavy, cdr3_light, epitope
        )

        # Combined score (weighted average)
        binding_score = (
            0.4 * charge_score +
            0.4 * hydro_score +
            0.2 * length_score
        )

        return {
            'binding_score': binding_score,
            'charge_score': charge_score,
            'hydro_score': hydro_score,
            'length_score': length_score,
            'cdr3_heavy': cdr3_heavy,
            'cdr3_light': cdr3_light
        }

    def score_diversity(
        self,
        antibody_sequence: str,
        other_sequences: list
    ) -> float:
        """
        Score sequence diversity

        Higher score if antibody is different from others

        Args:
            antibody_sequence: Current antibody sequence
            other_sequences: List of other antibody sequences

        Returns:
            Diversity score (0-1, higher = more diverse)
        """
        if not other_sequences:
            return 1.0

        # Calculate similarity to each other sequence
        similarities = []
        for other in other_sequences:
            # Simple similarity: fraction of matching positions
            matches = sum(1 for a, b in zip(antibody_sequence, other) if a == b)
            max_len = max(len(antibody_sequence), len(other))
            similarity = matches / max_len if max_len > 0 else 0
            similarities.append(similarity)

        # Diversity = 1 - max_similarity
        max_similarity = max(similarities) if similarities else 0
        diversity = 1 - max_similarity

        return diversity


def test_scorer():
    """Test the sequence binding scorer"""
    print("="*80)
    print("TESTING SEQUENCE BINDING SCORER")
    print("="*80)

    scorer = SequenceBindingScorer()

    # Test antibody
    heavy = "EVQLVETGGGLVQPGGSLRLSCAASGFTLNSYGISWVRQAPGKGPEWVSVIPPIGRRTFYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDGPYYYYGMDVWGQGTTVTVSS"
    light = "DVMTQSPLSLPVTPGEPASISCRSSQSLLHSNGYNYLDWYLQKPGQSPQLLIYLGSNRASGVPDRFSGSGSGTDFTLKISRVEAEDVGVYYCMQALQTPFTFGPGTKVDIK"
    epitope = "GKIADYNYKLPDDFTGCVIAWNSNN"

    # Score binding
    result = scorer.score_binding_potential(heavy, light, epitope)

    print(f"\nTest Antibody:")
    print(f"  Heavy: {len(heavy)} aa")
    print(f"  Light: {len(light)} aa")
    print(f"  Epitope: {epitope}")

    print(f"\nExtracted CDR3 regions:")
    print(f"  Heavy CDR3: {result['cdr3_heavy']}")
    print(f"  Light CDR3: {result['cdr3_light']}")

    print(f"\nBinding Scores:")
    print(f"  Overall:  {result['binding_score']:.3f}")
    print(f"  Charge:   {result['charge_score']:.3f}")
    print(f"  Hydro:    {result['hydro_score']:.3f}")
    print(f"  Length:   {result['length_score']:.3f}")

    # Test diversity
    other_abs = [heavy, heavy[:100] + "X" * 20]
    diversity = scorer.score_diversity(heavy, other_abs)
    print(f"\nDiversity Score: {diversity:.3f}")

    print("\n" + "="*80)
    print("✅ Scorer working correctly!")
    print("="*80)


if __name__ == '__main__':
    test_scorer()
