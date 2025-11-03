"""
Validate Generated Antibodies with IgFold

IgFold is a specialized deep learning model for predicting antibody structures.
Unlike ESM-2 (general proteins), IgFold is specifically designed for antibodies
and provides accurate structure predictions with pLDDT confidence scores.

This is the BEST validation method for antibody generation models.

Usage:
    # Validate antibodies from a trained model
    python validate_with_igfold.py --checkpoint checkpoints/improved_small_2025_10_31_best.pt

    # Validate with custom settings
    python validate_with_igfold.py --checkpoint checkpoints/best.pt --num-samples 10 --save-pdbs

References:
    Ruffolo et al. (2023) "Fast, accurate antibody structure prediction from deep learning"
    Nature Methods, doi: 10.1038/s41592-022-01490-7
"""

import argparse
import torch
import json
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from generators.transformer_seq2seq import create_model
from generators.tokenizer import AminoAcidTokenizer
from generators.data_loader import AbAgDataset


def load_model_and_generate(checkpoint_path, num_samples=10, device='cuda'):
    """
    Load trained model and generate antibody sequences

    Args:
        checkpoint_path: Path to model checkpoint
        num_samples: Number of antibodies to generate
        device: 'cuda' or 'cpu'

    Returns:
        List of (antibody_heavy, antibody_light, antigen, target_pkd) tuples
    """
    print("\n" + "="*70)
    print("Loading Model and Generating Antibodies")
    print("="*70)

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AminoAcidTokenizer()
    print(f"   Vocab size: {tokenizer.vocab_size}")

    # Load validation data
    print("\n2. Loading validation data...")
    val_dataset = AbAgDataset(
        data_path='data/generative/val.json',
        tokenizer=tokenizer
    )
    print(f"   Validation samples: {len(val_dataset)}")

    # Create model
    print("\n3. Creating model...")
    model = create_model('small', vocab_size=tokenizer.vocab_size, max_src_len=512, max_tgt_len=300)
    print(f"   Parameters: {model.get_model_size():,}")

    # Load checkpoint
    print(f"\n4. Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"   Loaded from epoch {checkpoint['epoch']}")
    print(f"   Validation loss: {checkpoint['val_loss']:.4f}")

    # Generate antibodies
    print(f"\n5. Generating {num_samples} antibodies...")
    generated_antibodies = []

    with torch.no_grad():
        for i in range(min(num_samples, len(val_dataset))):
            # Get sample
            sample = val_dataset[i]

            # Tokenize antigen sequence
            antigen_seq = sample['antigen_sequence']
            antigen_tokens = tokenizer.encode(antigen_seq)
            # Truncate to max length if needed
            if len(antigen_tokens) > 512:
                antigen_tokens = antigen_tokens[:512]
            antigen_tokens_tensor = torch.tensor([antigen_tokens]).to(device)

            # Prepare pKd
            pkd = torch.tensor([[sample['pKd']]]).float().to(device)

            # Generate
            generated = model.generate_greedy(antigen_tokens_tensor, pkd, max_length=300)

            # Decode
            antibody_seq = tokenizer.decode(generated[0].cpu().tolist())

            # Split heavy and light chains
            if '|' in antibody_seq:
                heavy, light = antibody_seq.split('|')
            else:
                # If no separator, assume first ~120 aa is heavy, rest is light
                heavy = antibody_seq[:120]
                light = antibody_seq[120:]

            generated_antibodies.append((
                heavy,
                light,
                antigen_seq,
                float(sample['pKd'])
            ))

            if (i + 1) % 5 == 0:
                print(f"   Generated {i + 1}/{num_samples}...")

    print(f"\n✅ Generated {len(generated_antibodies)} antibodies")

    return generated_antibodies


def validate_with_igfold(antibodies, device='cuda', save_pdbs=False, output_dir='igfold_results'):
    """
    Validate antibodies using IgFold

    IgFold is specifically designed for antibodies and provides:
    - Accurate 3D structure prediction
    - pLDDT confidence scores (0-100, higher is better)
    - Fast prediction (~2-5 seconds per antibody)

    Args:
        antibodies: List of (heavy, light, antigen, pkd) tuples
        device: 'cuda' or 'cpu'
        save_pdbs: Whether to save PDB structure files
        output_dir: Directory to save results

    Returns:
        List of validation results
    """
    print("\n" + "="*70)
    print("Validating Antibodies with IgFold")
    print("="*70)

    try:
        from igfold import IgFoldRunner
        import numpy as np
    except ImportError:
        print("\n❌ IgFold not installed!")
        print("\nTo install:")
        print("  pip install igfold")
        return []

    # Initialize IgFold
    print("\nInitializing IgFold...")
    igfold = IgFoldRunner()
    print("✅ IgFold loaded")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    if save_pdbs:
        pdb_dir = output_path / 'structures'
        pdb_dir.mkdir(exist_ok=True)
        print(f"   PDB structures will be saved to: {pdb_dir}")

    # Validate each antibody
    results = []

    print(f"\nValidating {len(antibodies)} antibodies...")
    print("(This may take a few minutes)\n")

    for i, (heavy, light, antigen, pkd) in enumerate(antibodies):
        print(f"[{i+1}/{len(antibodies)}] Validating antibody {i+1}...")

        try:
            # Predict structure with IgFold
            # Format sequences as dict with chain names
            sequences = {
                'H': heavy,
                'L': light
            }

            # Run IgFold prediction
            # IgFold saves to a temporary PDB file, we need to provide output path
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
                pdb_file = tmp.name

            igfold.fold(
                pdb_file=pdb_file,
                sequences=sequences,
                do_refine=False,  # Don't refine (requires PyRosetta which isn't installed)
                do_renum=False    # Don't renumber (keep original numbering)
            )

            # Read the PDB file
            with open(pdb_file, 'r') as f:
                pdb_string = f.read()

            # Clean up temp file
            import os
            os.unlink(pdb_file)

            # Extract pLDDT scores from PDB (stored in B-factor column)
            plddt_scores = []
            for line in pdb_string.split('\n'):
                if line.startswith('ATOM'):
                    try:
                        # B-factor is in columns 61-66 (0-indexed: 60-66)
                        bfactor = float(line[60:66].strip())
                        plddt_scores.append(bfactor)
                    except:
                        pass
            plddt_scores = np.array(plddt_scores)

            # Convert from fraction (0-1) to percentage (0-100) and clip outliers
            plddt_scores = np.clip(plddt_scores * 100, 0, 100)

            # Calculate metrics
            mean_plddt = float(plddt_scores.mean())
            confident = float((plddt_scores > 70).sum() / len(plddt_scores) * 100)

            # Determine quality grade
            if mean_plddt > 90:
                quality = "Excellent"
                is_good = True
            elif mean_plddt > 70:
                quality = "Good"
                is_good = True
            elif mean_plddt > 50:
                quality = "Fair"
                is_good = False
            else:
                quality = "Poor"
                is_good = False

            result = {
                'antibody_id': i,
                'heavy_chain': heavy,
                'light_chain': light,
                'antigen': antigen[:50] + '...',  # Truncate for readability
                'target_pkd': pkd,
                'heavy_length': len(heavy),
                'light_length': len(light),
                'total_length': len(heavy) + len(light),
                'mean_plddt': mean_plddt,
                'min_plddt': float(plddt_scores.min()),
                'max_plddt': float(plddt_scores.max()),
                'confident_residues_pct': confident,
                'is_good_structure': is_good,
                'quality_grade': quality
            }

            results.append(result)

            # Print result
            status = "✅" if result['is_good_structure'] else "⚠️ "
            print(f"    pLDDT: {mean_plddt:.1f} - {quality} {status}")

            # Save PDB if requested
            if save_pdbs:
                pdb_file = pdb_dir / f'antibody_{i:03d}_plddt{mean_plddt:.0f}.pdb'
                with open(pdb_file, 'w') as f:
                    f.write(pdb_string)

        except Exception as e:
            print(f"    ❌ Error: {e}")
            results.append({
                'antibody_id': i,
                'heavy_chain': heavy,
                'light_chain': light,
                'error': str(e)
            })

    return results


def save_results(results, output_dir='igfold_results'):
    """Save validation results"""
    import numpy as np

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save detailed results
    results_file = output_path / 'igfold_validation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Detailed results saved to: {results_file}")

    # Calculate summary statistics
    valid_results = [r for r in results if 'error' not in r]

    if valid_results:
        plddt_scores = [r['mean_plddt'] for r in valid_results]

        summary = {
            'total_antibodies': len(results),
            'successful_predictions': len(valid_results),
            'failed_predictions': len(results) - len(valid_results),
            'mean_plddt': float(np.mean(plddt_scores)),
            'std_plddt': float(np.std(plddt_scores)),
            'median_plddt': float(np.median(plddt_scores)),
            'min_plddt': float(np.min(plddt_scores)),
            'max_plddt': float(np.max(plddt_scores)),
            'excellent_structures': sum(1 for p in plddt_scores if p > 90),
            'good_structures': sum(1 for p in plddt_scores if p > 70),
            'fair_structures': sum(1 for p in plddt_scores if 50 < p <= 70),
            'poor_structures': sum(1 for p in plddt_scores if p <= 50),
            'quality_distribution': {
                'Excellent (>90)': sum(1 for p in plddt_scores if p > 90),
                'Good (70-90)': sum(1 for p in plddt_scores if 70 < p <= 90),
                'Fair (50-70)': sum(1 for p in plddt_scores if 50 < p <= 70),
                'Poor (<50)': sum(1 for p in plddt_scores if p <= 50)
            }
        }

        # Save summary
        summary_file = output_path / 'igfold_validation_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Print summary
        print("\n" + "="*70)
        print("IgFold Validation Summary")
        print("="*70)
        print(f"Total antibodies:         {summary['total_antibodies']}")
        print(f"Successful predictions:   {summary['successful_predictions']}")
        print(f"Failed predictions:       {summary['failed_predictions']}")
        print(f"\nStructure Quality (pLDDT scores):")
        print(f"  Mean pLDDT:             {summary['mean_plddt']:.2f} ± {summary['std_plddt']:.2f}")
        print(f"  Median pLDDT:           {summary['median_plddt']:.2f}")
        print(f"  Range:                  {summary['min_plddt']:.2f} - {summary['max_plddt']:.2f}")
        print(f"\nQuality Distribution:")
        print(f"  Excellent (>90):        {summary['excellent_structures']} ({summary['excellent_structures']/len(valid_results)*100:.1f}%)")
        print(f"  Good (70-90):           {summary['good_structures']} ({summary['good_structures']/len(valid_results)*100:.1f}%)")
        print(f"  Fair (50-70):           {summary['fair_structures']} ({summary['fair_structures']/len(valid_results)*100:.1f}%)")
        print(f"  Poor (<50):             {summary['poor_structures']} ({summary['poor_structures']/len(valid_results)*100:.1f}%)")
        print("="*70)

        print(f"\n✅ Summary saved to: {summary_file}")

        # Interpretation guide
        print("\n" + "="*70)
        print("How to Interpret IgFold Results")
        print("="*70)
        print("\npLDDT Score Ranges:")
        print("  >90:  Excellent - High confidence, very accurate structure")
        print("  70-90: Good - Good confidence, reliable structure")
        print("  50-70: Fair - Moderate confidence, some uncertainty")
        print("  <50:  Poor - Low confidence, questionable structure")
        print("\nFor antibodies, good models typically achieve:")
        print("  - Mean pLDDT: 75-85")
        print("  - >70% of structures with pLDDT > 70")
        print("  - Natural CDR conformations")
        print("="*70)

    else:
        print("\n⚠️  No valid results to summarize")


def main():
    parser = argparse.ArgumentParser(description='Validate generated antibodies with IgFold')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of antibodies to validate (default: 10)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use: cuda or cpu (default: cuda)')
    parser.add_argument('--output-dir', type=str, default='igfold_results',
                       help='Output directory for results (default: igfold_results)')
    parser.add_argument('--save-pdbs', action='store_true',
                       help='Save PDB structure files')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("Antibody Validation Pipeline (IgFold)")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples: {args.num_samples}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    print(f"Save PDBs: {args.save_pdbs}")

    # Step 1: Generate antibodies
    antibodies = load_model_and_generate(
        args.checkpoint,
        num_samples=args.num_samples,
        device=args.device
    )

    if not antibodies:
        print("\n❌ No antibodies generated. Exiting.")
        return

    # Step 2: Validate with IgFold
    results = validate_with_igfold(
        antibodies,
        device=args.device,
        save_pdbs=args.save_pdbs,
        output_dir=args.output_dir
    )

    if not results:
        print("\n❌ Validation failed. Exiting.")
        return

    # Step 3: Save results
    save_results(results, output_dir=args.output_dir)

    print("\n✅ Validation complete!")
    print(f"\nResults saved in: {args.output_dir}/")
    if args.save_pdbs:
        print(f"PDB structures saved in: {args.output_dir}/structures/")


if __name__ == '__main__':
    main()
