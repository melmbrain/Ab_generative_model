"""
Antibody Ranking System

Ranks generated antibodies for synthesis based on multiple criteria:
1. Epitope prediction score
2. Binding potential (sequence-based)
3. Sequence diversity
4. Optional: Structure quality (if available)

Usage:
    from rank_antibodies import AntibodyRanker

    ranker = AntibodyRanker()
    ranked = ranker.rank_antibodies(antibodies, epitopes)
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sequence_binding_scorer import SequenceBindingScorer


class AntibodyRanker:
    """
    Rank antibodies for synthesis based on multiple criteria
    """

    def __init__(self):
        """Initialize ranker"""
        self.binding_scorer = SequenceBindingScorer()

    def calculate_combined_score(
        self,
        antibody: Dict,
        epitope: Dict,
        all_antibodies: List[Dict],
        weights: Dict = None
    ) -> Tuple[float, Dict]:
        """
        Calculate combined score for antibody

        Args:
            antibody: Antibody dict with heavy/light chains
            epitope: Epitope dict with sequence and score
            all_antibodies: All antibodies (for diversity calculation)
            weights: Score weights (optional)

        Returns:
            (total_score, score_breakdown)
        """
        if weights is None:
            weights = {
                'epitope': 0.30,
                'binding': 0.30,
                'diversity': 0.20,
                'structure': 0.20
            }

        scores = {}

        # 1. Epitope score (higher = better epitope prediction)
        scores['epitope'] = epitope.get('score', 0.5)

        # 2. Binding score (sequence-based)
        binding_result = self.binding_scorer.score_binding_potential(
            antibody['heavy_chain'],
            antibody['light_chain'],
            epitope['sequence']
        )
        scores['binding'] = binding_result['binding_score']
        scores['binding_details'] = binding_result

        # 3. Diversity score
        other_sequences = [
            ab['full_sequence']
            for ab in all_antibodies
            if ab.get('id') != antibody.get('id')
        ]
        scores['diversity'] = self.binding_scorer.score_diversity(
            antibody['full_sequence'],
            other_sequences
        )

        # 4. Structure score (if available)
        if 'plddt_heavy' in antibody and antibody['plddt_heavy'] is not None:
            # Use average pLDDT as structure score
            plddt_avg = (antibody['plddt_heavy'] + antibody.get('plddt_light', 0)) / 2
            scores['structure'] = plddt_avg / 100  # Normalize to 0-1
        else:
            # No structure info, use neutral score
            scores['structure'] = 0.5

        # Calculate weighted total
        total_score = sum(
            weights[key] * scores[key]
            for key in weights.keys()
        )

        return total_score, scores

    def rank_antibodies(
        self,
        antibodies: List[Dict],
        epitopes: List[Dict] = None,
        weights: Dict = None,
        top_k: int = None
    ) -> List[Tuple[Dict, float, Dict]]:
        """
        Rank antibodies by combined score

        Args:
            antibodies: List of antibody dicts
            epitopes: List of epitope dicts (matched by index)
            weights: Score weights (optional)
            top_k: Return only top K (optional)

        Returns:
            List of (antibody, score, breakdown) tuples, sorted by score
        """
        if epitopes is None:
            # Create dummy epitopes if not provided
            epitopes = [
                {
                    'sequence': ab.get('epitope_sequence', 'UNKNOWN'),
                    'score': ab.get('epitope_score', 0.5)
                }
                for ab in antibodies
            ]

        ranked = []

        for i, antibody in enumerate(antibodies):
            # Get matching epitope
            epitope_idx = min(i, len(epitopes) - 1)
            epitope = epitopes[epitope_idx]

            # Calculate score
            total_score, breakdown = self.calculate_combined_score(
                antibody, epitope, antibodies, weights
            )

            ranked.append((antibody, total_score, breakdown))

        # Sort by score (descending)
        ranked.sort(key=lambda x: x[1], reverse=True)

        # Return top K if requested
        if top_k is not None:
            ranked = ranked[:top_k]

        return ranked

    def print_ranking_report(
        self,
        ranked: List[Tuple[Dict, float, Dict]],
        output_file: Path = None
    ):
        """
        Print and optionally save ranking report

        Args:
            ranked: Ranked antibodies from rank_antibodies()
            output_file: Optional file to save report
        """
        report_lines = []

        def print_and_save(line):
            print(line)
            report_lines.append(line)

        print_and_save("=" * 80)
        print_and_save("ANTIBODY RANKING REPORT")
        print_and_save("=" * 80)
        print_and_save(f"\nTotal antibodies: {len(ranked)}")

        for rank, (ab, score, breakdown) in enumerate(ranked, 1):
            print_and_save(f"\n{'─' * 80}")
            print_and_save(f"Rank {rank}: {ab.get('id', 'Unknown')} (Score: {score:.3f})")
            print_and_save(f"{'─' * 80}")

            # Antibody info
            print_and_save(f"Epitope: {ab.get('epitope_sequence', 'N/A')[:30]}...")
            print_and_save(f"Heavy: {len(ab.get('heavy_chain', ''))} aa")
            print_and_save(f"Light: {len(ab.get('light_chain', ''))} aa")

            # Score breakdown
            print_and_save(f"\nScore Breakdown:")
            print_and_save(f"  Epitope:   {breakdown['epitope']:.3f}")
            print_and_save(f"  Binding:   {breakdown['binding']:.3f}")
            print_and_save(f"  Diversity: {breakdown['diversity']:.3f}")
            print_and_save(f"  Structure: {breakdown['structure']:.3f}")

            # Binding details
            if 'binding_details' in breakdown:
                bd = breakdown['binding_details']
                print_and_save(f"\nBinding Details:")
                print_and_save(f"  Charge complementarity: {bd['charge_score']:.3f}")
                print_and_save(f"  Hydrophobicity match:   {bd['hydro_score']:.3f}")
                print_and_save(f"  Length compatibility:   {bd['length_score']:.3f}")

            # Quality indicators
            quality = []
            if breakdown['epitope'] > 0.65:
                quality.append("✅ High epitope score")
            if breakdown['binding'] > 0.6:
                quality.append("✅ Good binding potential")
            if breakdown['diversity'] > 0.7:
                quality.append("✅ Highly diverse")

            if quality:
                print_and_save(f"\nQuality Indicators:")
                for q in quality:
                    print_and_save(f"  {q}")

        print_and_save(f"\n{'=' * 80}")
        print_and_save("SYNTHESIS RECOMMENDATIONS")
        print_and_save("=" * 80)

        # Top 3 recommendations
        top_3 = ranked[:3]
        print_and_save(f"\nTop 3 Candidates for Synthesis:")
        for i, (ab, score, breakdown) in enumerate(top_3, 1):
            print_and_save(f"{i}. {ab.get('id', 'Unknown')} (Score: {score:.3f})")
            print_and_save(f"   Epitope: {ab.get('epitope_sequence', 'N/A')[:30]}...")
            print_and_save(f"   Length: {len(ab.get('heavy_chain', ''))} + {len(ab.get('light_chain', ''))} aa")

        # Synthesis criteria check
        print_and_save(f"\n{'─' * 80}")
        print_and_save("Synthesis Criteria Check:")
        print_and_save("─" * 80)

        for i, (ab, score, breakdown) in enumerate(top_3, 1):
            criteria_met = []
            criteria_failed = []

            if breakdown['epitope'] >= 0.65:
                criteria_met.append("Epitope score ≥0.65")
            else:
                criteria_failed.append(f"Epitope score {breakdown['epitope']:.3f} < 0.65")

            if breakdown['binding'] >= 0.5:
                criteria_met.append("Binding score ≥0.5")
            else:
                criteria_failed.append(f"Binding score {breakdown['binding']:.3f} < 0.5")

            if breakdown['diversity'] >= 0.3:
                criteria_met.append("Diverse sequence")
            else:
                criteria_failed.append("Low diversity")

            print_and_save(f"\n{ab.get('id', 'Unknown')}:")
            if criteria_met:
                for c in criteria_met:
                    print_and_save(f"  ✅ {c}")
            if criteria_failed:
                for c in criteria_failed:
                    print_and_save(f"  ⚠️  {c}")

        print_and_save(f"\n{'=' * 80}")

        # Save to file if requested
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write('\n'.join(report_lines))
            print(f"\n💾 Report saved to: {output_file}")

    def save_ranked_results(
        self,
        ranked: List[Tuple[Dict, float, Dict]],
        output_file: Path
    ):
        """
        Save ranked results to JSON

        Args:
            ranked: Ranked antibodies
            output_file: Output JSON file
        """
        results = []

        for rank, (ab, score, breakdown) in enumerate(ranked, 1):
            result = {
                'rank': rank,
                'antibody': ab,
                'total_score': float(score),
                'scores': {
                    'epitope': float(breakdown['epitope']),
                    'binding': float(breakdown['binding']),
                    'diversity': float(breakdown['diversity']),
                    'structure': float(breakdown['structure'])
                }
            }

            if 'binding_details' in breakdown:
                bd = breakdown['binding_details']
                result['binding_details'] = {
                    'charge_score': float(bd['charge_score']),
                    'hydro_score': float(bd['hydro_score']),
                    'length_score': float(bd['length_score'])
                }

            results.append(result)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({
                'total_antibodies': len(ranked),
                'ranked_antibodies': results
            }, f, indent=2)

        print(f"💾 Ranked results saved to: {output_file}")


def main():
    """Test the ranker with example antibodies"""
    print("=" * 80)
    print("TESTING ANTIBODY RANKER")
    print("=" * 80)

    # Example antibodies
    antibodies = [
        {
            'id': 'Ab_1',
            'epitope_sequence': 'GKIADYNYKLPDDFTGCVIAWNSNN',
            'epitope_score': 0.74,
            'heavy_chain': 'EVQLVETGGGLVQPGGSLRLSCAASGFTLNSYGISWVRQAPGKGPEWVSVIPPIGRRTFYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDGPYYYYGMDVWGQGTTVTVSS',
            'light_chain': 'DVMTQSPLSLPVTPGEPASISCRSSQSLLHSNGYNYLDWYLQKPGQSPQLLIYLGSNRASGVPDRFSGSGSGTDFTLKISRVEAEDVGVYYCMQALQTPFTFGPGTKVDIK',
            'full_sequence': 'EVQLVETGGGLVQPGGSLRLSCAASGFTLNSYGISWVRQAPGKGPEWVSVIPPIGRRTFYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDGPYYYYGMDVWGQGTTVTVSS|DVMTQSPLSLPVTPGEPASISCRSSQSLLHSNGYNYLDWYLQKPGQSPQLLIYLGSNRASGVPDRFSGSGSGTDFTLKISRVEAEDVGVYYCMQALQTPFTFGPGTKVDIK'
        },
        {
            'id': 'Ab_2',
            'epitope_sequence': 'AVEQDKNTQEVF',
            'epitope_score': 0.72,
            'heavy_chain': 'QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAVIWYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAREPLYYFDYWGQGTLVTVSS',
            'light_chain': 'DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK',
            'full_sequence': 'QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAVIWYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAREPLYYFDYWGQGTLVTVSS|DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK'
        },
        {
            'id': 'Ab_3',
            'epitope_sequence': 'GKIADYNYKLPDDFTGCVIAWNSNN',
            'epitope_score': 0.74,
            'heavy_chain': 'EVQLVESGGGLVQPGGSLRLSCAASGFTFSNYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDRSGYFDYWGQGTLVTVSS',
            'light_chain': 'EIVLTQSPGTLSLSPGERATLSCRASQSVSSYLAWYQQKPGQAPRLLIYGASSRATGIPDRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGSSPITFGQGTRLEIK',
            'full_sequence': 'EVQLVESGGGLVQPGGSLRLSCAASGFTFSNYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDRSGYFDYWGQGTLVTVSS|EIVLTQSPGTLSLSPGERATLSCRASQSVSSYLAWYQQKPGQAPRLLIYGASSRATGIPDRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGSSPITFGQGTRLEIK'
        }
    ]

    # Epitopes
    epitopes = [
        {'sequence': 'GKIADYNYKLPDDFTGCVIAWNSNN', 'score': 0.74},
        {'sequence': 'AVEQDKNTQEVF', 'score': 0.72},
        {'sequence': 'GKIADYNYKLPDDFTGCVIAWNSNN', 'score': 0.74}
    ]

    # Rank antibodies
    ranker = AntibodyRanker()
    ranked = ranker.rank_antibodies(antibodies, epitopes)

    # Print report
    ranker.print_ranking_report(ranked)

    # Save results
    ranker.save_ranked_results(
        ranked,
        Path('results/test_ranking_results.json')
    )

    print("\n✅ Ranker test complete!")


if __name__ == '__main__':
    main()
