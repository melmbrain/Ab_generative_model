"""
Structure Validation Module for Generated Antibodies

Validates generated antibody sequences using structure prediction tools:
- ESMFold (fast, available now)
- Placeholder for AlphaFold3 (when available)
- IgFold (antibody-specific, optional)

Usage:
    validator = StructureValidator(method='esmfold')
    results = validator.validate_antibody(sequence)
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


class StructureValidator:
    """
    Validate antibody structures using prediction models
    """

    def __init__(self, method='esmfold', device='cuda'):
        """
        Initialize structure validator

        Args:
            method: 'esmfold', 'alphafold3', or 'igfold'
            device: 'cuda' or 'cpu'
        """
        self.method = method
        self.device = device
        self.model = None

        print(f"Initializing {method} validator on {device}...")
        self._load_model()

    def _load_model(self):
        """Load the structure prediction model"""
        if self.method == 'esmfold':
            self._load_esmfold()
        elif self.method == 'alphafold3':
            self._load_alphafold3()
        elif self.method == 'igfold':
            self._load_igfold()
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _load_esmfold(self):
        """Load ESMFold model"""
        try:
            import esm
            print("Loading ESMFold model (this may take a minute)...")
            self.model = esm.pretrained.esmfold_v1()
            self.model = self.model.eval()

            if self.device == 'cuda' and torch.cuda.is_available():
                self.model = self.model.cuda()
                print("✅ ESMFold loaded on GPU")
            else:
                print("✅ ESMFold loaded on CPU (slower)")

        except ImportError:
            print("❌ ESMFold not installed!")
            print("Install with: pip install fair-esm")
            print("Or: pip install 'fair-esm[esmfold]'")
            raise

    def _load_alphafold3(self):
        """Placeholder for AlphaFold3 (not yet available for easy install)"""
        print("⚠️  AlphaFold3 is not yet available for automated use")
        print("Options:")
        print("  1. Use web server: https://alphafoldserver.com")
        print("  2. Wait for official Python package")
        print("  3. Use ESMFold instead (recommended)")
        raise NotImplementedError(
            "AlphaFold3 not available. Use method='esmfold' instead."
        )

    def _load_igfold(self):
        """Load IgFold model (antibody-specific)"""
        try:
            from igfold import IgFoldRunner
            print("Loading IgFold model...")
            self.model = IgFoldRunner()
            print("✅ IgFold loaded")
        except ImportError:
            print("❌ IgFold not installed!")
            print("Install with: pip install igfold")
            raise

    def validate_antibody(self, sequence: str, antigen: str = None) -> Dict:
        """
        Validate a single antibody sequence

        Args:
            sequence: Antibody sequence (heavy|light)
            antigen: Antigen sequence (optional, for binding prediction)

        Returns:
            Dictionary with validation results
        """
        if self.method == 'esmfold':
            return self._validate_esmfold(sequence)
        elif self.method == 'igfold':
            return self._validate_igfold(sequence)
        else:
            raise NotImplementedError(f"Method {self.method} not implemented")

    def _validate_esmfold(self, sequence: str) -> Dict:
        """
        Validate using ESMFold

        Returns structure quality scores
        """
        # Remove separator for structure prediction
        seq_clean = sequence.replace('|', '')

        with torch.no_grad():
            # Predict structure
            output = self.model.infer_pdb(seq_clean)

            # Extract confidence scores (pLDDT)
            # pLDDT: per-residue confidence (0-100)
            # Higher is better, >70 is good, >90 is excellent

        # Parse PDB output to get pLDDT scores
        plddt_scores = self._extract_plddt_from_pdb(output)

        results = {
            'method': 'esmfold',
            'sequence_length': len(seq_clean),
            'mean_plddt': float(np.mean(plddt_scores)),
            'min_plddt': float(np.min(plddt_scores)),
            'max_plddt': float(np.max(plddt_scores)),
            'confident_residues': float(np.sum(plddt_scores > 70) / len(plddt_scores) * 100),
            'excellent_residues': float(np.sum(plddt_scores > 90) / len(plddt_scores) * 100),
            'is_good_structure': float(np.mean(plddt_scores)) > 70,
            'pdb_structure': output
        }

        return results

    def _validate_igfold(self, sequence: str) -> Dict:
        """
        Validate using IgFold (antibody-specific)

        Returns antibody-specific quality scores
        """
        # Split into heavy and light chains
        if '|' in sequence:
            heavy, light = sequence.split('|')
        else:
            # Assume single chain
            heavy = sequence
            light = None

        # Predict structure
        sequences = {'H': heavy}
        if light:
            sequences['L'] = light

        output = self.model.fold(
            fv_heavy_chain=heavy,
            fv_light_chain=light
        )

        results = {
            'method': 'igfold',
            'has_heavy': True,
            'has_light': light is not None,
            'structure': output
        }

        return results

    def _extract_plddt_from_pdb(self, pdb_string: str) -> np.ndarray:
        """
        Extract pLDDT scores from PDB string

        Args:
            pdb_string: PDB format string

        Returns:
            Array of pLDDT scores per residue
        """
        plddt_scores = []

        for line in pdb_string.split('\n'):
            if line.startswith('ATOM'):
                # pLDDT is in the B-factor column (columns 61-66)
                try:
                    plddt = float(line[60:66].strip())
                    plddt_scores.append(plddt)
                except:
                    pass

        # Get unique scores (one per residue, not per atom)
        # Group by residue number
        residue_plddt = {}
        for line in pdb_string.split('\n'):
            if line.startswith('ATOM'):
                res_num = int(line[22:26].strip())
                plddt = float(line[60:66].strip())
                if res_num not in residue_plddt:
                    residue_plddt[res_num] = plddt

        return np.array(list(residue_plddt.values()))

    def validate_batch(self, sequences: List[str],
                      save_results: bool = True,
                      output_dir: str = 'validation_results') -> List[Dict]:
        """
        Validate multiple antibody sequences

        Args:
            sequences: List of antibody sequences
            save_results: Save results to file
            output_dir: Directory to save results

        Returns:
            List of validation results
        """
        results = []

        print(f"\nValidating {len(sequences)} sequences...")

        for i, seq in enumerate(sequences):
            print(f"  [{i+1}/{len(sequences)}] Validating sequence {i+1}...")

            try:
                result = self.validate_antibody(seq)
                result['sequence_id'] = i
                result['sequence'] = seq
                results.append(result)

                # Print summary
                if 'mean_plddt' in result:
                    quality = result['mean_plddt']
                    status = "✅ Good" if quality > 70 else "⚠️  Fair" if quality > 50 else "❌ Poor"
                    print(f"    pLDDT: {quality:.1f} - {status}")

            except Exception as e:
                print(f"    ❌ Error: {e}")
                results.append({
                    'sequence_id': i,
                    'sequence': seq,
                    'error': str(e)
                })

        # Save results
        if save_results:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

            results_file = output_path / 'validation_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"\n✅ Results saved to: {results_file}")

            # Save summary
            self._save_summary(results, output_path)

        return results

    def _save_summary(self, results: List[Dict], output_dir: Path):
        """Save validation summary statistics"""
        valid_results = [r for r in results if 'error' not in r]

        if not valid_results:
            print("No valid results to summarize")
            return

        # Calculate statistics
        plddt_scores = [r.get('mean_plddt', 0) for r in valid_results if 'mean_plddt' in r]

        summary = {
            'total_sequences': len(results),
            'successful': len(valid_results),
            'failed': len(results) - len(valid_results),
            'mean_plddt': float(np.mean(plddt_scores)) if plddt_scores else 0,
            'good_structures': sum(1 for p in plddt_scores if p > 70),
            'excellent_structures': sum(1 for p in plddt_scores if p > 90),
            'poor_structures': sum(1 for p in plddt_scores if p < 50)
        }

        # Save summary
        summary_file = output_dir / 'validation_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Print summary
        print("\n" + "="*60)
        print("Validation Summary")
        print("="*60)
        print(f"Total sequences:        {summary['total_sequences']}")
        print(f"Successful predictions: {summary['successful']}")
        print(f"Failed predictions:     {summary['failed']}")
        if plddt_scores:
            print(f"Mean pLDDT:            {summary['mean_plddt']:.1f}")
            print(f"Good structures (>70): {summary['good_structures']} ({summary['good_structures']/len(plddt_scores)*100:.1f}%)")
            print(f"Excellent (>90):       {summary['excellent_structures']} ({summary['excellent_structures']/len(plddt_scores)*100:.1f}%)")
            print(f"Poor (<50):            {summary['poor_structures']} ({summary['poor_structures']/len(plddt_scores)*100:.1f}%)")
        print("="*60)


def test_validator():
    """Test the structure validator"""
    print("="*70)
    print("Testing Structure Validator")
    print("="*70)

    # Test sequence (example antibody)
    test_sequence = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDGGYAMDYWGQGTLVTVSS|DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIKR"

    try:
        # Initialize validator
        validator = StructureValidator(method='esmfold', device='cuda')

        # Validate
        print(f"\nValidating test sequence...")
        result = validator.validate_antibody(test_sequence)

        # Print results
        print("\nValidation Results:")
        print(f"  Method: {result['method']}")
        print(f"  Sequence length: {result['sequence_length']}")
        print(f"  Mean pLDDT: {result['mean_plddt']:.2f}")
        print(f"  Min pLDDT: {result['min_plddt']:.2f}")
        print(f"  Max pLDDT: {result['max_plddt']:.2f}")
        print(f"  Confident residues (>70): {result['confident_residues']:.1f}%")
        print(f"  Excellent residues (>90): {result['excellent_residues']:.1f}%")
        print(f"  Is good structure: {'✅ Yes' if result['is_good_structure'] else '❌ No'}")

        print("\n✅ Validator test passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Install ESMFold: pip install fair-esm")
        print("  2. Ensure you have enough GPU memory (2GB+)")
        print("  3. If GPU issues, try device='cpu'")


if __name__ == '__main__':
    test_validator()
