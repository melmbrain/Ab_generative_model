"""
B-Cell Epitope Predictor

This module predicts B-cell epitopes (antibody binding sites) in antigen sequences.

Two options:
1. IEDB BepiPred-2.0 API (easy, online)
2. Local scoring based on hydrophilicity, accessibility, flexibility

Usage:
    from epitope_predictor import EpitopePredictor

    predictor = EpitopePredictor(method='iedb')  # or 'local'
    epitopes = predictor.predict(antigen_sequence)
"""

import requests
import time
import numpy as np
from typing import List, Dict
from io import StringIO


class EpitopePredictor:
    """
    Predict B-cell epitopes in protein sequences
    """

    def __init__(self, method='iedb', min_length=8, max_length=25, threshold=0.5):
        """
        Initialize epitope predictor

        Args:
            method: 'iedb' for BepiPred-2.0 API, 'local' for simple scoring
            min_length: Minimum epitope length (aa)
            max_length: Maximum epitope length (aa)
            threshold: Score threshold (0-1)
        """
        self.method = method
        self.min_length = min_length
        self.max_length = max_length
        self.threshold = threshold

        print(f"Epitope Predictor initialized:")
        print(f"  Method: {method}")
        print(f"  Length range: {min_length}-{max_length} aa")
        print(f"  Threshold: {threshold}")

    def predict(self, sequence: str, top_k: int = 10) -> List[Dict]:
        """
        Predict epitopes in sequence

        Args:
            sequence: Antigen amino acid sequence
            top_k: Return top K epitopes

        Returns:
            List of epitope dictionaries with:
            - sequence: epitope sequence
            - position: start-end position
            - score: confidence score (0-1)
            - method: prediction method used
        """
        if self.method == 'iedb':
            return self._predict_iedb(sequence, top_k)
        else:
            return self._predict_local(sequence, top_k)

    def _predict_iedb(self, sequence: str, top_k: int) -> List[Dict]:
        """
        Use IEDB BepiPred-2.0 API for prediction

        Reference: http://tools.iedb.org/bcell/help/#bepipred-2.0
        """
        print(f"Predicting epitopes with IEDB BepiPred-2.0...")
        print(f"  Sequence length: {len(sequence)} aa")

        try:
            # IEDB BepiPred-2.0 API endpoint
            url = "http://tools-cluster-interface.iedb.org/tools_api/bcell/"

            params = {
                'method': 'Bepipred-2.0',
                'sequence_text': sequence,
            }

            # Make request
            print(f"  Sending request to IEDB...")
            response = requests.post(url, data=params, timeout=60)

            if response.status_code != 200:
                print(f"  ⚠️  IEDB API returned status {response.status_code}")
                print(f"  Falling back to local prediction")
                return self._predict_local(sequence, top_k)

            # Parse response
            scores = self._parse_iedb_response(response.text, sequence)

            if scores is None:
                print(f"  ⚠️  Could not parse IEDB response")
                print(f"  Falling back to local prediction")
                return self._predict_local(sequence, top_k)

            print(f"  ✅ Received {len(scores)} scores from IEDB")

        except Exception as e:
            print(f"  ❌ Error calling IEDB API: {e}")
            print(f"  Falling back to local prediction")
            return self._predict_local(sequence, top_k)

        # Extract epitopes from scores
        epitopes = self._extract_epitopes_from_scores(scores, sequence)

        # Sort by score and return top K
        epitopes.sort(key=lambda x: x['score'], reverse=True)

        return epitopes[:top_k]

    def _parse_iedb_response(self, response_text: str, sequence: str):
        """
        Parse IEDB API response

        Response format:
        Position    Residue Score
        1           M       0.352
        2           A       0.421
        ...
        """
        try:
            lines = response_text.strip().split('\n')

            # Skip header
            if len(lines) < 2:
                return None

            scores = []
            for line in lines[1:]:  # Skip header line
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        position = int(parts[0])
                        residue = parts[1]
                        score = float(parts[2])
                        scores.append(score)
                    except (ValueError, IndexError):
                        continue

            if len(scores) != len(sequence):
                print(f"  ⚠️  Score count ({len(scores)}) doesn't match sequence length ({len(sequence)})")
                return None

            return np.array(scores)

        except Exception as e:
            print(f"  Error parsing IEDB response: {e}")
            return None

    def _predict_local(self, sequence: str, top_k: int) -> List[Dict]:
        """
        Simple local prediction based on physicochemical properties

        Uses Parker hydrophilicity scale as a proxy for surface exposure
        """
        print(f"Predicting epitopes with local method...")
        print(f"  Sequence length: {len(sequence)} aa")

        # Parker hydrophilicity scale
        hydrophilicity = {
            'A': 2.1, 'R': 4.2, 'N': 4.4, 'D': 10.0, 'C': 1.4,
            'Q': 4.1, 'E': 7.8, 'G': 2.5, 'H': 2.1, 'I': -8.0,
            'L': -9.2, 'K': 5.7, 'M': -4.2, 'F': -9.2, 'P': -0.2,
            'S': 3.0, 'T': 3.2, 'W': -10.0, 'Y': -1.9, 'V': -3.7
        }

        # Calculate per-residue scores
        scores = []
        for aa in sequence:
            score = hydrophilicity.get(aa, 0.0)
            # Normalize to 0-1 range (Parker scale is roughly -10 to +10)
            normalized = (score + 10) / 20
            scores.append(max(0, min(1, normalized)))  # Clip to [0, 1]

        scores = np.array(scores)

        # Smooth with sliding window
        window_size = 7
        smoothed = np.convolve(scores, np.ones(window_size)/window_size, mode='same')

        # Extract epitopes
        epitopes = self._extract_epitopes_from_scores(smoothed, sequence)

        # Sort by score
        epitopes.sort(key=lambda x: x['score'], reverse=True)

        print(f"  Found {len(epitopes)} candidate epitopes")

        return epitopes[:top_k]

    def _extract_epitopes_from_scores(self, scores: np.ndarray, sequence: str) -> List[Dict]:
        """
        Extract continuous high-scoring regions as epitopes
        """
        epitopes = []
        in_epitope = False
        epitope_start = 0

        for i, score in enumerate(scores):
            if score > self.threshold and not in_epitope:
                # Start of potential epitope
                epitope_start = i
                in_epitope = True

            elif (score <= self.threshold or i == len(scores) - 1) and in_epitope:
                # End of epitope
                epitope_end = i if score <= self.threshold else i + 1
                epitope_length = epitope_end - epitope_start

                # Check length constraints
                if self.min_length <= epitope_length <= self.max_length:
                    epitope_seq = sequence[epitope_start:epitope_end]
                    epitope_score = float(np.mean(scores[epitope_start:epitope_end]))

                    epitopes.append({
                        'sequence': epitope_seq,
                        'position': f"{epitope_start}-{epitope_end}",
                        'start': epitope_start,
                        'end': epitope_end,
                        'length': epitope_length,
                        'score': epitope_score,
                        'method': self.method
                    })

                in_epitope = False

        return epitopes


def test_predictor():
    """
    Test epitope predictor on SARS-CoV-2 spike protein
    """
    print("="*80)
    print("EPITOPE PREDICTOR TEST")
    print("="*80)

    # Load SARS-CoV-2 spike protein
    from pathlib import Path

    spike_file = Path('sars_cov2_spike.fasta')

    if not spike_file.exists():
        print("\n⚠️  sars_cov2_spike.fasta not found")
        print("   Using truncated sequence for demo")

        # Use RBD region (residues 319-541)
        sequence = "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF"
    else:
        # Load from file
        with open(spike_file) as f:
            lines = f.readlines()
            sequence = ''.join([line.strip() for line in lines if not line.startswith('>')])

    print(f"\nTest sequence length: {len(sequence)} aa")

    # Test both methods
    for method in ['local', 'iedb']:
        print(f"\n{'─'*80}")
        print(f"Testing {method.upper()} method")
        print(f"{'─'*80}")

        predictor = EpitopePredictor(method=method, threshold=0.5)
        epitopes = predictor.predict(sequence, top_k=10)

        print(f"\nTop 10 predicted epitopes:")
        for i, ep in enumerate(epitopes, 1):
            print(f"\n{i}. {ep['sequence']}")
            print(f"   Position: {ep['position']}")
            print(f"   Length: {ep['length']} aa")
            print(f"   Score: {ep['score']:.3f}")
            print(f"   Method: {ep['method']}")

        # Check if known epitopes are found
        known_epitopes = [
            'YQAGSTPCNGVEG',  # Known SARS-CoV-2 epitope (RBD)
            'GKIADYNYKLPDDFT',  # Another known epitope
        ]

        print(f"\n{'─'*80}")
        print(f"Validation: Known epitopes")
        print(f"{'─'*80}")

        for known in known_epitopes:
            # Check if found
            found = any(known in ep['sequence'] or ep['sequence'] in known
                       for ep in epitopes)

            if found:
                print(f"  ✅ Found: {known}")
            else:
                # Check if it's in the sequence at all
                if known in sequence:
                    print(f"  ⚠️  Missed: {known} (in sequence but not predicted)")
                else:
                    print(f"  ❓ N/A: {known} (not in test sequence)")


if __name__ == '__main__':
    test_predictor()
