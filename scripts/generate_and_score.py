"""
Main Script: Generate and Score Antibody Candidates

Production-ready pipeline for virus-specific antibody library design.

Usage:
    python scripts/generate_and_score.py --antigen virus_sequence.txt --output results/

Workflow:
    1. Generate antibody candidates (template-based)
    2. Score with Phase 2 discriminator (Spearman ρ = 0.85)
    3. Rank and save top candidates
"""

import argparse
import sys
import logging
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from discriminator import AffinityDiscriminator
from generators import TemplateGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_antigen_sequence(antigen_file: str) -> str:
    """
    Load antigen sequence from file

    Args:
        antigen_file: Path to antigen sequence file (.txt or .fasta)

    Returns:
        Clean antigen sequence

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If sequence is invalid
    """
    path = Path(antigen_file)

    if not path.exists():
        raise FileNotFoundError(
            f"Antigen file not found: {antigen_file}\n"
            f"Current directory: {Path.cwd()}"
        )

    # Read file
    try:
        with open(path, 'r') as f:
            content = f.read().strip()
    except Exception as e:
        raise RuntimeError(f"Failed to read file: {e}")

    # Remove FASTA header if present
    if content.startswith('>'):
        lines = content.split('\n')
        sequence = ''.join(lines[1:])
    else:
        sequence = content.replace('\n', '').replace(' ', '').upper()

    # Validate
    if not sequence:
        raise ValueError("Antigen file is empty")

    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    invalid = set(sequence) - valid_aa
    if invalid:
        raise ValueError(f"Invalid amino acids in antigen: {invalid}")

    if len(sequence) < 10:
        raise ValueError(f"Antigen too short ({len(sequence)} aa). Minimum: 10 aa")

    return sequence


def print_header():
    """Print pipeline header"""
    print("\n" + "="*80)
    print(" "*20 + "🧬 ANTIBODY LIBRARY GENERATION PIPELINE")
    print("="*80)
    print(" "*15 + "Production-Ready Virus-Specific Antibody Design")
    print(" "*20 + "Powered by Phase 2 Model (Spearman ρ = 0.85)")
    print("="*80 + "\n")


def print_summary_table(stats: Dict):
    """Print results summary table"""
    print("\n" + "="*80)
    print("📊 RESULTS SUMMARY")
    print("="*80)

    print(f"\n{'Metric':<30} {'Value':<20}")
    print("-" * 50)
    print(f"{'Total candidates generated':<30} {stats['total_generated']:<20}")
    print(f"{'Successfully scored':<30} {stats['successfully_scored']:<20}")
    print(f"{'Success rate':<30} {stats['successfully_scored']/stats['total_generated']*100:.1f}%")
    print()
    print(f"{'Top pKd':<30} {stats['top_pKd']:.2f}")
    print(f"{'Top Kd':<30} {stats['top_Kd_nM']:.1f} nM")
    print(f"{'Mean pKd':<30} {stats['mean_pKd']:.2f} ± {stats['std_pKd']:.2f}")
    print(f"{'Median pKd':<30} {stats['median_pKd']:.2f}")
    print()
    print(f"{'Excellent binders (pKd > 9)':<30} {stats['excellent_count']:<20}")
    print(f"{'Good binders (pKd 7.5-9)':<30} {stats['good_count']:<20}")
    print(f"{'Moderate binders (pKd 6-7.5)':<30} {stats['moderate_count']:<20}")
    print(f"{'Poor binders (pKd < 6)':<30} {stats['poor_count']:<20}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Generate and score antibody candidates for virus target',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 candidates for SARS-CoV-2 Spike
  python scripts/generate_and_score.py \\
    --antigen data/example_antigen.txt \\
    --n-candidates 100 \\
    --output data/results/sars_cov2

  # Quick test with 20 candidates
  python scripts/generate_and_score.py \\
    --antigen data/example_antigen.txt \\
    --n-candidates 20 \\
    --quick

  # High-throughput screening (500 candidates)
  python scripts/generate_and_score.py \\
    --antigen data/my_virus.txt \\
    --n-candidates 500 \\
    --mutations 4 \\
    --output data/results/large_screen
        """
    )

    # Required arguments
    parser.add_argument('--antigen', type=str, required=True,
                       help='Path to antigen sequence file (.txt or .fasta)')

    # Optional arguments
    parser.add_argument('--n-candidates', type=int, default=100,
                       help='Number of candidates to generate (default: 100)')
    parser.add_argument('--mutations', type=int, default=3,
                       help='Mutations per variant (default: 3)')
    parser.add_argument('--output', type=str, default='data/results',
                       help='Output directory (default: data/results)')
    parser.add_argument('--top-n', type=int, default=50,
                       help='Number of top candidates to save separately (default: 50)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: minimal output')
    parser.add_argument('--quiet', action='store_true',
                       help='Quiet mode: suppress progress bars')

    # Model selection
    parser.add_argument('--model', type=str, default='models/agab_phase2_model.pth',
                       help='Path to discriminator model')

    args = parser.parse_args()

    # Print header
    if not args.quiet:
        print_header()

    logger.info("Configuration:")
    logger.info(f"  Antigen file: {args.antigen}")
    logger.info(f"  Candidates: {args.n_candidates}")
    logger.info(f"  Mutations per variant: {args.mutations}")
    logger.info(f"  Output directory: {args.output}")
    logger.info(f"  Model: {args.model}")

    # ==================== STEP 1: Load Antigen ====================
    logger.info("\n📖 Step 1: Loading antigen sequence...")

    try:
        antigen_seq = load_antigen_sequence(args.antigen)
        logger.info(f"✅ Loaded antigen: {len(antigen_seq)} amino acids")
        logger.info(f"   Preview: {antigen_seq[:60]}{'...' if len(antigen_seq) > 60 else ''}")
    except Exception as e:
        logger.error(f"❌ Failed to load antigen: {e}")
        sys.exit(1)

    # ==================== STEP 2: Generate Candidates ====================
    logger.info(f"\n🧬 Step 2: Generating {args.n_candidates} antibody candidates...")

    try:
        generator = TemplateGenerator()
        candidates = generator.generate(
            n_candidates=args.n_candidates,
            mutations_per_variant=args.mutations,
            focus_on_cdr3=True
        )
        logger.info(f"✅ Generated {len(candidates)} candidates from {len(generator.templates)} templates")

    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        sys.exit(1)

    # ==================== STEP 3: Score Candidates ====================
    logger.info(f"\n📊 Step 3: Scoring candidates with Phase 2 discriminator...")

    try:
        discriminator = AffinityDiscriminator(
            model_path=args.model,
            verbose=not args.quiet
        )

        # Prepare sequences for batch scoring
        antibody_seqs = [c['full_sequence'] for c in candidates]
        antigen_seqs = [antigen_seq] * len(antibody_seqs)

        # Score with progress bar
        predictions = discriminator.predict_batch(
            antibody_seqs,
            antigen_seqs,
            progress_bar=not args.quiet
        )

        # Combine with candidate info
        results = []
        for cand, pred in zip(candidates, predictions):
            if pred.get('status') == 'success':
                results.append({
                    'id': cand['id'],
                    'template_id': cand['template_id'],
                    'predicted_pKd': pred['predicted_pKd'],
                    'predicted_Kd_nM': pred['predicted_Kd_nM'],
                    'predicted_Kd_uM': pred['predicted_Kd_uM'],
                    'binding_category': pred['binding_category'],
                    'interpretation': pred['interpretation'],
                    'antibody_heavy': cand['antibody_heavy'],
                    'antibody_light': cand['antibody_light'],
                    'full_sequence': cand['full_sequence'],
                    'n_mutations': cand['n_mutations'],
                    'source': cand['source'],
                    'timestamp': pred['timestamp']
                })

        logger.info(f"✅ Successfully scored {len(results)}/{len(candidates)} candidates")

    except Exception as e:
        logger.error(f"❌ Scoring failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ==================== STEP 4: Rank and Analyze ====================
    logger.info(f"\n🏆 Step 4: Ranking and analyzing results...")

    if len(results) == 0:
        logger.error("❌ No successful predictions. Cannot continue.")
        sys.exit(1)

    df = pd.DataFrame(results)
    df = df.sort_values('predicted_pKd', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)

    # Calculate statistics
    stats = {
        'total_generated': args.n_candidates,
        'successfully_scored': len(df),
        'timestamp': datetime.now().isoformat(),
        'antigen_file': args.antigen,
        'antigen_length': len(antigen_seq),
        'n_mutations': args.mutations,
        'top_pKd': float(df.iloc[0]['predicted_pKd']),
        'top_Kd_nM': float(df.iloc[0]['predicted_Kd_nM']),
        'mean_pKd': float(df['predicted_pKd'].mean()),
        'median_pKd': float(df['predicted_pKd'].median()),
        'std_pKd': float(df['predicted_pKd'].std()),
        'min_pKd': float(df['predicted_pKd'].min()),
        'max_pKd': float(df['predicted_pKd'].max()),
        'excellent_count': len(df[df['binding_category'] == 'excellent']),
        'good_count': len(df[df['binding_category'] == 'good']),
        'moderate_count': len(df[df['binding_category'] == 'moderate']),
        'poor_count': len(df[df['binding_category'] == 'poor'])
    }

    # Print summary
    if not args.quiet:
        print_summary_table(stats)

    # ==================== STEP 5: Save Results ====================
    logger.info(f"\n💾 Step 5: Saving results...")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save all candidates
        all_file = output_dir / 'all_candidates_scored.csv'
        df.to_csv(all_file, index=False)
        logger.info(f"✅ Saved all candidates ({len(df)}): {all_file}")

        # Save top N
        top_n = min(args.top_n, len(df))
        top_df = df.head(top_n)
        top_file = output_dir / f'top_{top_n}_candidates.csv'
        top_df.to_csv(top_file, index=False)
        logger.info(f"✅ Saved top {top_n}: {top_file}")

        # Save statistics
        stats_file = output_dir / 'statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"✅ Saved statistics: {stats_file}")

        # Save antigen sequence
        antigen_file = output_dir / 'antigen_sequence.txt'
        with open(antigen_file, 'w') as f:
            f.write(f">Target Antigen ({len(antigen_seq)} aa)\n")
            f.write(f">Source: {args.antigen}\n")
            f.write(antigen_seq)
        logger.info(f"✅ Saved antigen sequence: {antigen_file}")

        # Save top 10 summary (human-readable)
        summary_file = output_dir / 'TOP_10_SUMMARY.txt'
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("TOP 10 ANTIBODY CANDIDATES FOR SYNTHESIS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Antigen: {args.antigen} ({len(antigen_seq)} aa)\n")
            f.write(f"Total candidates: {len(df)}\n\n")

            top_10 = df.head(10)
            for idx, row in top_10.iterrows():
                f.write(f"\n{'='*80}\n")
                f.write(f"Rank #{row['rank']}: {row['id']}\n")
                f.write(f"{'='*80}\n")
                f.write(f"Predicted pKd: {row['predicted_pKd']:.2f}\n")
                f.write(f"Predicted Kd:  {row['predicted_Kd_nM']:.1f} nM ({row['predicted_Kd_uM']:.3f} μM)\n")
                f.write(f"Category:      {row['binding_category']}\n")
                f.write(f"{row['interpretation']}\n")
                f.write(f"\nTemplate: {row['template_id']}\n")
                f.write(f"Mutations: {row['n_mutations']}\n\n")
                f.write(f"Heavy Chain ({len(row['antibody_heavy'])} aa):\n{row['antibody_heavy']}\n\n")
                if row['antibody_light']:
                    f.write(f"Light Chain ({len(row['antibody_light'])} aa):\n{row['antibody_light']}\n\n")

        logger.info(f"✅ Saved top 10 summary: {summary_file}")

    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")
        sys.exit(1)

    # ==================== FINAL SUMMARY ====================
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE!")
    print("="*80)

    print(f"\n🏆 Top 10 Candidates:\n")
    print("-" * 80)
    top_10_display = df.head(10)[['rank', 'id', 'predicted_pKd', 'predicted_Kd_nM', 'binding_category', 'template_id']]
    print(top_10_display.to_string(index=False))

    print(f"\n📂 Results Location:")
    print(f"   {output_dir.absolute()}/")
    print(f"\n📄 Key Files:")
    print(f"   • {all_file.name} - All {len(df)} scored candidates")
    print(f"   • {top_file.name} - Top {top_n} candidates")
    print(f"   • TOP_10_SUMMARY.txt - Human-readable top 10")
    print(f"   • statistics.json - Summary statistics")

    print(f"\n🚀 Next Steps:")
    print(f"   1. Review top candidates in: {top_file}")
    print(f"   2. Check diversity (different templates)")
    print(f"   3. Select 10-20 diverse candidates")
    print(f"   4. Synthesize and validate experimentally")
    print(f"   5. Iterate based on experimental results")

    print("\n" + "="*80)
    print(f"Pipeline completed successfully in {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
