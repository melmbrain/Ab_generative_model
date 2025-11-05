"""
Validate Antibody FASTA Files with IgFold

This script takes antibody FASTA files and validates them using IgFold,
which predicts 3D structure and provides pLDDT confidence scores.

IgFold is specifically designed for antibodies and provides:
- Structure prediction (PDB files)
- pLDDT scores (per-residue confidence)
- Overall structure quality assessment

Usage:
    python validate_fasta_with_igfold.py \
        --input results/full_pipeline/antibody_1.fasta \
        --output results/full_pipeline/validation

    # Validate all antibodies in a directory
    python validate_fasta_with_igfold.py \
        --input-dir results/full_pipeline \
        --output results/validation
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

try:
    import torch
    from igfold import IgFoldRunner
    IGFOLD_AVAILABLE = True
except ImportError as e:
    print(f"❌ Error importing IgFold: {e}")
    print("Please install: pip install igfold")
    IGFOLD_AVAILABLE = False
    sys.exit(1)


def parse_fasta(fasta_file: Path):
    """
    Parse FASTA file to extract heavy and light chains

    Returns:
        dict with 'heavy' and 'light' sequences
    """
    sequences = {}
    current_id = None
    current_seq = []

    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence
                if current_id and current_seq:
                    seq = ''.join(current_seq)
                    if 'Heavy' in current_id:
                        sequences['heavy'] = seq
                    elif 'Light' in current_id:
                        sequences['light'] = seq

                # Start new sequence
                current_id = line
                current_seq = []
            else:
                current_seq.append(line)

        # Save last sequence
        if current_id and current_seq:
            seq = ''.join(current_seq)
            if 'Heavy' in current_id:
                sequences['heavy'] = seq
            elif 'Light' in current_id:
                sequences['light'] = seq

    return sequences


def validate_antibody_with_igfold(sequences: dict, output_dir: Path, antibody_name: str):
    """
    Validate antibody structure using IgFold

    Args:
        sequences: Dict with 'heavy' and 'light' chain sequences
        output_dir: Directory to save results
        antibody_name: Name for output files

    Returns:
        dict with validation results
    """
    print(f"\n{'='*80}")
    print(f"Validating {antibody_name} with IgFold")
    print(f"{'='*80}")

    # Check sequences
    if 'heavy' not in sequences or 'light' not in sequences:
        print("❌ Error: FASTA must contain both heavy and light chains")
        return None

    heavy_seq = sequences['heavy']
    light_seq = sequences['light']

    print(f"\n📊 Sequence Information:")
    print(f"   Heavy chain: {len(heavy_seq)} amino acids")
    print(f"   Light chain: {len(light_seq)} amino acids")
    print(f"   Total: {len(heavy_seq) + len(light_seq)} amino acids")

    # Initialize IgFold
    print(f"\n🔧 Initializing IgFold...")

    # Use CPU or GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Using device: {device}")

    try:
        igfold = IgFoldRunner()
    except Exception as e:
        print(f"❌ Error initializing IgFold: {e}")
        return None

    # Predict structure
    print(f"\n🧬 Predicting structure...")
    print(f"   This may take 1-2 minutes...")

    try:
        # IgFold expects sequences parameter
        # Format: list of dicts with 'H' and 'L' keys
        sequences_dict = {
            'H': heavy_seq,  # Heavy chain
            'L': light_seq   # Light chain
        }

        # Create temporary FASTA file
        import tempfile
        temp_fasta = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False)
        temp_fasta.write(f">H\n{heavy_seq}\n")
        temp_fasta.write(f">L\n{light_seq}\n")
        temp_fasta.close()

        # Output PDB path
        output_pdb = output_dir / f"{antibody_name}.pdb"

        # Predict using fold() method
        output = igfold.fold(
            str(output_pdb),  # Output PDB file path
            fasta_file=temp_fasta.name,  # Input FASTA
            do_refine=False,  # Skip refinement for speed
            do_renum=False    # Skip renumbering
        )

        # Clean up temp file
        import os
        os.unlink(temp_fasta.name)

        # Extract results - IgFold returns IgFoldOutput object
        pdb_file = str(output_pdb)

        # IgFold uses pRMSD (predicted RMSD) instead of pLDDT
        # pRMSD is in Angstroms - lower is better
        # Extract pRMSD from output object
        # IgFold uses 4 models, so prmsd has shape [num_residues, 4]
        # Average across models to get per-residue scores
        prmsd_raw = output.prmsd.cpu().numpy()  # Shape: [num_residues, 4]
        prmsd = np.mean(prmsd_raw, axis=-1)  # Shape: [num_residues]

        print(f"✅ Structure prediction complete!")
        print(f"   PDB file: {pdb_file}")

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Analyze pRMSD scores
    print(f"\n📊 Analyzing Structure Quality (pRMSD)...")
    print(f"   Note: IgFold uses pRMSD (predicted RMSD in Å) as confidence metric")
    print(f"         Lower pRMSD = Higher quality (unlike pLDDT)")

    # Overall statistics
    mean_prmsd = float(np.mean(prmsd))
    median_prmsd = float(np.median(prmsd))
    min_prmsd = float(np.min(prmsd))
    max_prmsd = float(np.max(prmsd))
    std_prmsd = float(np.std(prmsd))

    # Quality classification based on pRMSD thresholds
    # Typical thresholds: <1.0Å excellent, <2.0Å good, <3.5Å fair, >=3.5Å poor
    excellent = np.sum(prmsd < 1.0)
    good = np.sum((prmsd >= 1.0) & (prmsd < 2.0))
    fair = np.sum((prmsd >= 2.0) & (prmsd < 3.5))
    poor = np.sum(prmsd >= 3.5)
    total = len(prmsd)

    print(f"\n   Overall Statistics:")
    print(f"   ├─ Mean pRMSD:   {mean_prmsd:.2f} Å")
    print(f"   ├─ Median pRMSD: {median_prmsd:.2f} Å")
    print(f"   ├─ Std Dev:      {std_prmsd:.2f} Å")
    print(f"   ├─ Min pRMSD:    {min_prmsd:.2f} Å")
    print(f"   └─ Max pRMSD:    {max_prmsd:.2f} Å")

    print(f"\n   Quality Distribution:")
    print(f"   ├─ Excellent (<1.0Å):  {excellent:4d} / {total} ({excellent/total*100:5.1f}%)")
    print(f"   ├─ Good (1.0-2.0Å):    {good:4d} / {total} ({good/total*100:5.1f}%)")
    print(f"   ├─ Fair (2.0-3.5Å):    {fair:4d} / {total} ({fair/total*100:5.1f}%)")
    print(f"   └─ Poor (≥3.5Å):       {poor:4d} / {total} ({poor/total*100:5.1f}%)")

    # Overall assessment (lower pRMSD is better!)
    if mean_prmsd < 1.0:
        quality = "EXCELLENT"
        emoji = "🎯"
    elif mean_prmsd < 2.0:
        quality = "GOOD"
        emoji = "✅"
    elif mean_prmsd < 3.5:
        quality = "FAIR"
        emoji = "⚠️"
    else:
        quality = "POOR"
        emoji = "❌"

    print(f"\n   {emoji} Overall Quality: {quality} (Mean pRMSD: {mean_prmsd:.2f} Å)")

    # PDB file is already saved by IgFold
    # Rename it to something more descriptive
    saved_pdb_file = Path(pdb_file)
    new_pdb_file = output_dir / f"{antibody_name}_structure.pdb"

    try:
        if saved_pdb_file.exists() and saved_pdb_file != new_pdb_file:
            import shutil
            shutil.copy(saved_pdb_file, new_pdb_file)
            pdb_file = new_pdb_file
        print(f"\n💾 Structure saved to: {pdb_file}")
    except Exception as e:
        print(f"⚠️  Could not rename PDB: {e}")
        pdb_file = saved_pdb_file

    # Compile results
    results = {
        'antibody_name': antibody_name,
        'sequences': {
            'heavy': heavy_seq,
            'light': light_seq,
            'heavy_length': len(heavy_seq),
            'light_length': len(light_seq),
            'total_length': len(heavy_seq) + len(light_seq)
        },
        'prmsd_scores': {
            'mean': mean_prmsd,
            'median': median_prmsd,
            'std': std_prmsd,
            'min': min_prmsd,
            'max': max_prmsd,
            'per_residue': prmsd.tolist()
        },
        'quality_distribution': {
            'excellent': {'count': int(excellent), 'percentage': float(excellent/total*100)},
            'good': {'count': int(good), 'percentage': float(good/total*100)},
            'fair': {'count': int(fair), 'percentage': float(fair/total*100)},
            'poor': {'count': int(poor), 'percentage': float(poor/total*100)}
        },
        'overall_quality': quality,
        'structure_file': str(pdb_file) if pdb_file else None,
        'confidence_metric': 'pRMSD (Å)',
        'note': 'IgFold uses pRMSD (lower is better) instead of pLDDT (higher is better)'
    }

    # Save results JSON
    json_file = output_dir / f"{antibody_name}_validation.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"📄 Results saved to: {json_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate antibodies with IgFold")

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=str,
                            help='Single FASTA file to validate')
    input_group.add_argument('--input-dir', type=str,
                            help='Directory containing FASTA files')

    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for results')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get input files
    if args.input:
        fasta_files = [Path(args.input)]
    else:
        input_dir = Path(args.input_dir)
        fasta_files = list(input_dir.glob('*.fasta')) + list(input_dir.glob('*.fa'))

    if not fasta_files:
        print("❌ No FASTA files found!")
        return

    print(f"\n{'🧬'*40}")
    print("IGFOLD ANTIBODY STRUCTURE VALIDATION")
    print(f"{'🧬'*40}")
    print(f"\nFound {len(fasta_files)} FASTA file(s) to validate")
    print(f"Output directory: {output_dir}")

    # Validate each file
    all_results = []

    for i, fasta_file in enumerate(fasta_files, 1):
        print(f"\n{'─'*80}")
        print(f"Processing {i}/{len(fasta_files)}: {fasta_file.name}")
        print(f"{'─'*80}")

        # Parse FASTA
        sequences = parse_fasta(fasta_file)

        if not sequences:
            print(f"⚠️  Skipping {fasta_file.name} - could not parse sequences")
            continue

        # Validate
        antibody_name = fasta_file.stem
        result = validate_antibody_with_igfold(sequences, output_dir, antibody_name)

        if result:
            all_results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")

    if all_results:
        print(f"\n✅ Successfully validated {len(all_results)} antibody/antibodies")

        # Overall statistics
        mean_prmsd_all = np.mean([r['prmsd_scores']['mean'] for r in all_results])

        print(f"\n📊 Overall Statistics:")
        print(f"   Average mean pRMSD: {mean_prmsd_all:.2f} Å")

        print(f"\n   Individual Results:")
        for result in all_results:
            emoji = "🎯" if result['prmsd_scores']['mean'] < 1.0 else \
                   "✅" if result['prmsd_scores']['mean'] < 2.0 else \
                   "⚠️" if result['prmsd_scores']['mean'] < 3.5 else "❌"
            print(f"   {emoji} {result['antibody_name']}: {result['prmsd_scores']['mean']:.2f} Å ({result['overall_quality']})")

        # Save summary
        summary_file = output_dir / "validation_summary.json"
        summary = {
            'total_antibodies': len(all_results),
            'average_mean_prmsd': float(mean_prmsd_all),
            'confidence_metric': 'pRMSD (Å)',
            'note': 'Lower pRMSD indicates better structure quality',
            'results': all_results
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n💾 Summary saved to: {summary_file}")

        # Comparison with IgFold quality benchmarks
        print(f"\n{'='*80}")
        print("STRUCTURE QUALITY ASSESSMENT")
        print(f"{'='*80}")
        print(f"\n   IgFold quality benchmarks (pRMSD):")
        print(f"   ├─ Excellent: <1.0 Å (high confidence)")
        print(f"   ├─ Good:      1.0-2.0 Å (good confidence)")
        print(f"   ├─ Fair:      2.0-3.5 Å (moderate confidence)")
        print(f"   └─ Poor:      ≥3.5 Å (low confidence)")
        print(f"\n   Current validation mean pRMSD:  {mean_prmsd_all:.2f} Å (from {len(all_results)} antibodies)")

        if mean_prmsd_all < 1.0:
            print(f"\n   🎯 EXCELLENT! Generated antibodies have high-confidence structures!")
        elif mean_prmsd_all < 2.0:
            print(f"\n   ✅ GOOD! Generated antibodies have good-quality structures")
        elif mean_prmsd_all < 3.5:
            print(f"\n   ⚠️  FAIR! Generated antibodies have moderate-quality structures")
        else:
            print(f"\n   ❌ CONCERN! Generated antibodies have low-confidence structures")

    else:
        print("\n❌ No antibodies were successfully validated")

    print(f"\n{'='*80}")
    print("✅ VALIDATION COMPLETE")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
