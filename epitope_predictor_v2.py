"""
Improved B-Cell Epitope Predictor v2

Uses sliding window approach instead of continuous region extraction.
This better matches how epitopes are actually identified.

Usage:
    from epitope_predictor_v2 import EpitopePredictorV2

    predictor = EpitopePredictorV2()
    epitopes = predictor.predict(antigen_sequence)
"""

import numpy as np
from typing import List, Dict


class EpitopePredictorV2:
    """
    Improved epitope predictor using sliding window
    """

    def __init__(self, method='sliding_window', window_sizes=[10, 13, 15], threshold=0.5):
        """
        Initialize epitope predictor

        Args:
            method: 'sliding_window' for windowed scoring
            window_sizes: List of window sizes to try
            threshold: Score threshold (0-1)
        """
        self.method = method
        self.window_sizes = window_sizes
        self.threshold = threshold

        # Parker hydrophilicity scale
        self.hydrophilicity = {
            'A': 2.1, 'R': 4.2, 'N': 4.4, 'D': 10.0, 'C': 1.4,
            'Q': 4.1, 'E': 7.8, 'G': 2.5, 'H': 2.1, 'I': -8.0,
            'L': -9.2, 'K': 5.7, 'M': -4.2, 'F': -9.2, 'P': -0.2,
            'S': 3.0, 'T': 3.2, 'W': -10.0, 'Y': -1.9, 'V': -3.7
        }

        print(f"Epitope Predictor V2 initialized:")
        print(f"  Method: {method}")
        print(f"  Window sizes: {window_sizes}")
        print(f"  Threshold: {threshold}")

    def predict(self, sequence: str, top_k: int = 10) -> List[Dict]:
        """
        Predict epitopes using sliding window

        Args:
            sequence: Antigen amino acid sequence
            top_k: Return top K epitopes

        Returns:
            List of epitope dictionaries
        """
        print(f"\nPredicting epitopes with sliding window...")
        print(f"  Sequence length: {len(sequence)} aa")

        all_candidates = []

        # Try different window sizes
        for window_size in self.window_sizes:
            candidates = self._sliding_window_scan(sequence, window_size)
            all_candidates.extend(candidates)

        # Remove duplicates and overlaps
        all_candidates = self._remove_overlaps(all_candidates)

        # Sort by score
        all_candidates.sort(key=lambda x: x['score'], reverse=True)

        # Filter by threshold
        filtered = [c for c in all_candidates if c['score'] >= self.threshold]

        print(f"  Found {len(filtered)} epitopes above threshold {self.threshold}")

        return filtered[:top_k]

    def _sliding_window_scan(self, sequence: str, window_size: int) -> List[Dict]:
        """
        Scan sequence with sliding window
        """
        candidates = []

        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i+window_size]

            # Calculate hydrophilicity score
            scores = []
            for aa in window:
                score = self.hydrophilicity.get(aa, 0.0)
                normalized = (score + 10) / 20
                scores.append(max(0, min(1, normalized)))

            mean_score = np.mean(scores)

            candidates.append({
                'sequence': window,
                'position': f"{i}-{i+window_size}",
                'start': i,
                'end': i+window_size,
                'length': window_size,
                'score': float(mean_score),
                'method': 'sliding_window'
            })

        return candidates

    def _remove_overlaps(self, candidates: List[Dict], overlap_threshold=0.7) -> List[Dict]:
        """
        Remove highly overlapping epitopes, keeping higher-scoring ones
        """
        # Sort by score (descending)
        sorted_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)

        kept = []

        for candidate in sorted_candidates:
            # Check overlap with already kept candidates
            overlaps = False

            for kept_candidate in kept:
                # Calculate overlap
                start1, end1 = candidate['start'], candidate['end']
                start2, end2 = kept_candidate['start'], kept_candidate['end']

                overlap_start = max(start1, start2)
                overlap_end = min(end1, end2)
                overlap_len = max(0, overlap_end - overlap_start)

                # Calculate overlap fraction
                min_len = min(candidate['length'], kept_candidate['length'])
                overlap_fraction = overlap_len / min_len if min_len > 0 else 0

                if overlap_fraction > overlap_threshold:
                    overlaps = True
                    break

            if not overlaps:
                kept.append(candidate)

        return kept


def test_predictor_v2():
    """
    Test improved predictor
    """
    print("="*80)
    print("EPITOPE PREDICTOR V2 TEST")
    print("="*80)

    # Load SARS-CoV-2 spike protein
    from pathlib import Path

    spike_file = Path('sars_cov2_spike.fasta')

    if spike_file.exists():
        with open(spike_file) as f:
            lines = f.readlines()
            sequence = ''.join([line.strip() for line in lines if not line.startswith('>')])
    else:
        print("\n⚠️  sars_cov2_spike.fasta not found, using RBD region")
        sequence = "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF"

    print(f"\nTest sequence length: {len(sequence)} aa")

    # Known epitopes to find
    known_epitopes = [
        'YQAGSTPCNGVEG',
        'GKIADYNYKLPDDFT',
    ]

    # Test different thresholds
    for threshold in [0.50, 0.55, 0.60]:
        print(f"\n{'─'*80}")
        print(f"THRESHOLD: {threshold}")
        print(f"{'─'*80}")

        predictor = EpitopePredictorV2(threshold=threshold)
        epitopes = predictor.predict(sequence, top_k=20)

        # Check for known epitopes
        found_count = 0
        for known in known_epitopes:
            found = False
            for ep in epitopes:
                if known in ep['sequence'] or ep['sequence'] in known:
                    print(f"  ✅ FOUND: {known} in {ep['sequence']} (score: {ep['score']:.3f})")
                    found = True
                    found_count += 1
                    break

            if not found:
                # Check if it's in sequence
                if known in sequence:
                    # Find its position
                    pos = sequence.find(known)
                    # Calculate its score
                    scores = []
                    for aa in known:
                        score = predictor.hydrophilicity.get(aa, 0.0)
                        normalized = (score + 10) / 20
                        scores.append(max(0, min(1, normalized)))
                    mean_score = np.mean(scores)

                    print(f"  ❌ MISSED: {known} (pos {pos}, score: {mean_score:.3f}, threshold: {threshold})")
                else:
                    print(f"  ❓ N/A: {known} (not in test sequence)")

        print(f"\n  Summary: Found {found_count}/{len(known_epitopes)} known epitopes")
        print(f"  Total predicted: {len(epitopes)}")

        if len(epitopes) > 0:
            print(f"\n  Top 5 predictions:")
            for i, ep in enumerate(epitopes[:5], 1):
                print(f"    {i}. {ep['sequence']} (score: {ep['score']:.3f}, pos: {ep['position']})")


if __name__ == '__main__':
    test_predictor_v2()
