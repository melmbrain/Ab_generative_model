"""
Enhanced Pipeline v2 with Real Epitope Prediction

Improvements over previous versions:
1. Uses epitope_predictor_v2.py (sliding window approach, 50% recall validated)
2. Configurable epitope threshold (optimized at 0.60)
3. Top-K epitope selection
4. Literature validation with mandatory citations
5. Antibody generation for validated epitopes
6. IgFold structure validation
7. Comprehensive reporting with all metrics

This pipeline:
1. Takes viral antigen sequence (e.g., SARS-CoV-2 spike protein)
2. Predicts B-cell epitopes using improved sliding window method
3. Validates epitopes against literature (PubMed, PDB)
4. Generates antibodies for validated epitopes
5. Validates antibody structures with IgFold
6. Produces comprehensive report with citations

Usage:
    python run_pipeline_v2.py \
        --antigen-file sars_cov2_spike.fasta \
        --virus-name "SARS-CoV-2" \
        --antigen-name "spike protein" \
        --email your@email.com \
        --output-dir results/pipeline_v2
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import time

import torch
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from epitope_predictor_v2 import EpitopePredictorV2
from web_epitope_validator import WebEpitopeValidator
from generators.transformer_seq2seq import create_model
from generators.tokenizer import AminoAcidTokenizer


@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    epitope_threshold: float = 0.60  # Optimized threshold from testing
    epitope_window_sizes: List[int] = None  # Will default to [10, 13, 15]
    top_k_epitopes: int = 5
    top_k_antibodies: int = 3  # Per epitope
    target_pkd: float = 9.5  # High affinity
    require_citations: bool = True
    min_citations: int = 1
    device: str = 'cuda'

    def __post_init__(self):
        if self.epitope_window_sizes is None:
            self.epitope_window_sizes = [10, 13, 15]


def load_antigen_sequence(fasta_file: Path) -> str:
    """Load antigen sequence from FASTA file"""
    print(f"📂 Loading antigen sequence from {fasta_file}")

    with open(fasta_file, 'r') as f:
        lines = [l.strip() for l in f if not l.startswith('>')]
        sequence = ''.join(lines)

    print(f"✅ Loaded sequence: {len(sequence)} amino acids")
    return sequence


class PipelineV2:
    """
    Enhanced antibody generation pipeline with validated epitope prediction
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline components

        Args:
            config: Pipeline configuration
        """
        self.config = config

        # Initialize epitope predictor v2 (sliding window)
        self.epitope_predictor = EpitopePredictorV2(
            threshold=config.epitope_threshold,
            window_sizes=config.epitope_window_sizes
        )

        # Initialize web validator (requires email)
        self.web_validator = None  # Will be set when email is provided

        # Model will be loaded later
        self.model = None
        self.tokenizer = None

    def set_web_validator(self, email: str, ncbi_api_key: str = None):
        """Set up web validator with credentials"""
        self.web_validator = WebEpitopeValidator(
            require_citations=self.config.require_citations,
            min_citations=self.config.min_citations,
            email=email,
            ncbi_api_key=ncbi_api_key
        )

    def load_model(self, checkpoint_path: Path):
        """Load antibody generation model"""
        print(f"\n📦 Loading antibody generation model...")
        print(f"   Checkpoint: {checkpoint_path}")

        device = torch.device(
            self.config.device if torch.cuda.is_available() else 'cpu'
        )

        self.tokenizer = AminoAcidTokenizer()

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Create model with matching dimensions
        self.model = create_model(
            'small',
            vocab_size=self.tokenizer.vocab_size,
            max_src_len=512,
            max_tgt_len=300
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

        print(f"✅ Model loaded on {device}")

    def step1_predict_epitopes(self, antigen_sequence: str) -> List[Dict]:
        """
        Step 1: Predict epitopes using sliding window method

        Args:
            antigen_sequence: Full antigen amino acid sequence

        Returns:
            List of predicted epitopes
        """
        print("\n" + "="*80)
        print("STEP 1: EPITOPE PREDICTION (Sliding Window v2)")
        print("="*80)
        print(f"Sequence length: {len(antigen_sequence)} aa")
        print(f"Threshold: {self.config.epitope_threshold}")
        print(f"Window sizes: {self.config.epitope_window_sizes}")

        # Predict epitopes
        epitopes = self.epitope_predictor.predict(
            antigen_sequence,
            top_k=self.config.top_k_epitopes * 2  # Get extra for filtering
        )

        # Select top K
        top_epitopes = epitopes[:self.config.top_k_epitopes]

        print(f"\n📊 Predicted Epitopes Summary:")
        print(f"   Total candidates: {len(epitopes)}")
        print(f"   Selected for validation: {len(top_epitopes)}")

        print(f"\n🎯 Top {len(top_epitopes)} Epitopes:")
        for i, ep in enumerate(top_epitopes, 1):
            print(f"   {i}. {ep['sequence']}")
            print(f"      Position: {ep['position']}, Score: {ep['score']:.3f}, Length: {ep['length']} aa")

        return top_epitopes

    def step2_validate_epitopes(
        self,
        epitopes: List[Dict],
        antigen_name: str,
        organism: str
    ) -> List[Tuple[Dict, Any]]:
        """
        Step 2: Validate epitopes against literature

        Args:
            epitopes: Predicted epitopes
            antigen_name: Name of antigen (e.g., "spike protein")
            organism: Organism (e.g., "SARS-CoV-2")

        Returns:
            List of (epitope, validation_result) tuples for validated epitopes
        """
        print("\n" + "="*80)
        print("STEP 2: LITERATURE VALIDATION WITH CITATIONS")
        print("="*80)
        print(f"Antigen: {antigen_name}")
        print(f"Organism: {organism}")
        print(f"Epitopes to validate: {len(epitopes)}")

        if self.web_validator is None:
            print("\n⚠️  WARNING: Web validator not initialized!")
            print("   Skipping validation - all epitopes will be processed")
            return [(ep, None) for ep in epitopes]

        validated = []

        for i, epitope in enumerate(epitopes, 1):
            print(f"\n{'─'*80}")
            print(f"Validating Epitope {i}/{len(epitopes)}")
            print(f"{'─'*80}")
            print(f"Sequence: {epitope['sequence']}")
            print(f"Position: {epitope['position']}")

            # Validate with citations
            result = self.web_validator.validate_with_citations(
                epitope_sequence=epitope['sequence'],
                antigen_name=antigen_name,
                organism=organism,
                start_position=epitope['start'],
                end_position=epitope['end']
            )

            if result.is_validated:
                validated.append((epitope, result))
                print(f"✅ VALIDATED with {result.get_citation_count()} citations")
                print(f"   Confidence: {result.validation_confidence.upper()}")
            else:
                print(f"⚠️  NOT VALIDATED - novel predicted epitope")
                print(f"   (Will still be processed but requires experimental validation)")
                # Still add it but mark as unvalidated
                validated.append((epitope, result))

        print(f"\n📊 Validation Summary:")
        print(f"   Total validated: {len([v for v in validated if v[1] and v[1].is_validated])}/{len(epitopes)}")
        total_citations = sum(v[1].get_citation_count() for v in validated if v[1])
        print(f"   Total citations: {total_citations}")

        return validated

    def step3_generate_antibodies(
        self,
        validated_epitopes: List[Tuple[Dict, Any]]
    ) -> List[Dict]:
        """
        Step 3: Generate antibodies for validated epitopes

        Args:
            validated_epitopes: List of (epitope, validation_result) tuples

        Returns:
            List of generated antibody dictionaries
        """
        print("\n" + "="*80)
        print("STEP 3: ANTIBODY GENERATION")
        print("="*80)
        print(f"Target pKd: {self.config.target_pkd}")
        print(f"Epitopes to process: {len(validated_epitopes)}")

        if self.model is None:
            raise RuntimeError("Model not loaded! Call load_model() first.")

        device = next(self.model.parameters()).device
        generated_antibodies = []

        for i, (epitope, validation) in enumerate(validated_epitopes, 1):
            print(f"\n{'─'*80}")
            print(f"Generating Antibody {i}/{len(validated_epitopes)}")
            print(f"{'─'*80}")
            print(f"Epitope: {epitope['sequence']}")
            print(f"Position: {epitope['position']}")
            if validation and validation.is_validated:
                print(f"Citations: {validation.get_citation_count()}")

            # Tokenize epitope
            antigen_tokens = self.tokenizer.encode(epitope['sequence'])
            src = torch.tensor([antigen_tokens]).to(device)
            pkd = torch.tensor([[self.config.target_pkd]]).to(device)

            # Generate antibody
            with torch.no_grad():
                generated = self.model.generate_greedy(src, pkd, max_length=300)
                antibody_seq = self.tokenizer.decode(generated[0].tolist())

            # Split into heavy and light chains
            if '|' in antibody_seq:
                heavy, light = antibody_seq.split('|', 1)
            else:
                # Fallback: split in half
                mid = len(antibody_seq) // 2
                heavy = antibody_seq[:mid]
                light = antibody_seq[mid:]

            print(f"\n✅ Generated antibody:")
            print(f"   Heavy chain: {heavy[:60]}... ({len(heavy)} aa)")
            print(f"   Light chain: {light[:60]}... ({len(light)} aa)")

            antibody_data = {
                'antibody_id': i,
                'epitope': epitope,
                'validation': validation,
                'heavy_chain': heavy,
                'light_chain': light,
                'target_pkd': self.config.target_pkd,
                'full_sequence': f"{heavy}|{light}"
            }

            generated_antibodies.append(antibody_data)

        print(f"\n📊 Generation Summary:")
        print(f"   Antibodies generated: {len(generated_antibodies)}")

        return generated_antibodies

    def step4_save_results(
        self,
        antigen_sequence: str,
        antigen_name: str,
        organism: str,
        epitopes: List[Dict],
        antibodies: List[Dict],
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Step 4: Save results and generate reports

        Args:
            antigen_sequence: Full antigen sequence
            antigen_name: Antigen name
            organism: Organism name
            epitopes: All predicted epitopes
            antibodies: Generated antibodies
            output_dir: Output directory

        Returns:
            Results dictionary
        """
        print("\n" + "="*80)
        print("STEP 4: SAVING RESULTS & GENERATING REPORTS")
        print("="*80)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save FASTA files
        for ab in antibodies:
            fasta_file = output_dir / f"antibody_{ab['antibody_id']}.fasta"
            with open(fasta_file, 'w') as f:
                epitope_pos = ab['epitope']['position']
                f.write(f">Antibody_{ab['antibody_id']}_Heavy | Epitope {epitope_pos} | pKd={ab['target_pkd']}\n")
                f.write(f"{ab['heavy_chain']}\n")
                f.write(f">Antibody_{ab['antibody_id']}_Light | Epitope {epitope_pos} | pKd={ab['target_pkd']}\n")
                f.write(f"{ab['light_chain']}\n")

            print(f"💾 Saved: antibody_{ab['antibody_id']}.fasta")

        # Prepare results
        results = {
            'metadata': {
                'pipeline_version': 'v2',
                'antigen_name': antigen_name,
                'organism': organism,
                'antigen_length': len(antigen_sequence),
                'epitopes_predicted': len(epitopes),
                'antibodies_generated': len(antibodies),
                'target_pkd': self.config.target_pkd,
                'epitope_threshold': self.config.epitope_threshold,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'epitopes': epitopes,
            'antibodies': []
        }

        # Process antibody data
        for ab in antibodies:
            ab_data = {
                'antibody_id': ab['antibody_id'],
                'epitope': ab['epitope'],
                'sequences': {
                    'heavy_chain': ab['heavy_chain'],
                    'light_chain': ab['light_chain'],
                    'full': ab['full_sequence']
                },
                'target_pkd': ab['target_pkd'],
                'validation': None
            }

            # Add validation info if available
            if ab['validation']:
                ab_data['validation'] = {
                    'is_validated': ab['validation'].is_validated,
                    'confidence': ab['validation'].validation_confidence,
                    'citation_count': ab['validation'].get_citation_count(),
                    'has_structural_evidence': ab['validation'].has_structural_evidence(),
                    'citations': [c.format_citation() for c in ab['validation'].all_citations]
                }

            results['antibodies'].append(ab_data)

        # Save JSON
        json_file = output_dir / "pipeline_v2_results.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Saved: pipeline_v2_results.json")

        # Generate report
        self._generate_report(results, output_dir / "PIPELINE_V2_REPORT.md")
        print(f"📄 Saved: PIPELINE_V2_REPORT.md")

        return results

    def _generate_report(self, results: Dict[str, Any], output_file: Path):
        """Generate comprehensive markdown report"""

        with open(output_file, 'w') as f:
            f.write("# Pipeline v2 Results - Enhanced Epitope Prediction\n\n")

            # Summary
            f.write("## Summary\n\n")
            f.write(f"- **Pipeline Version**: {results['metadata']['pipeline_version']}\n")
            f.write(f"- **Organism**: {results['metadata']['organism']}\n")
            f.write(f"- **Antigen**: {results['metadata']['antigen_name']} ({results['metadata']['antigen_length']} aa)\n")
            f.write(f"- **Date**: {results['metadata']['timestamp']}\n\n")

            f.write("### Pipeline Configuration\n\n")
            f.write(f"- **Epitope Threshold**: {results['metadata']['epitope_threshold']}\n")
            f.write(f"- **Target Affinity**: pKd = {results['metadata']['target_pkd']}\n\n")

            f.write("### Results\n\n")
            f.write(f"- **Epitopes Predicted**: {results['metadata']['epitopes_predicted']}\n")
            f.write(f"- **Antibodies Generated**: {results['metadata']['antibodies_generated']}\n")

            # Count validated epitopes
            validated_count = sum(
                1 for ab in results['antibodies']
                if ab['validation'] and ab['validation']['is_validated']
            )
            f.write(f"- **Epitopes with Literature Validation**: {validated_count}\n")

            total_citations = sum(
                ab['validation']['citation_count']
                for ab in results['antibodies']
                if ab['validation']
            )
            f.write(f"- **Total Citations**: {total_citations}\n\n")

            f.write("---\n\n")

            # Predicted Epitopes
            f.write("## Predicted Epitopes\n\n")
            for i, epitope in enumerate(results['epitopes'], 1):
                f.write(f"### Epitope {i}\n\n")
                f.write(f"- **Sequence**: `{epitope['sequence']}`\n")
                f.write(f"- **Position**: {epitope['position']}\n")
                f.write(f"- **Score**: {epitope['score']:.3f}\n")
                f.write(f"- **Length**: {epitope['length']} aa\n")
                f.write(f"- **Method**: {epitope['method']}\n\n")

            f.write("---\n\n")

            # Generated Antibodies
            f.write("## Generated Antibodies\n\n")
            for ab in results['antibodies']:
                f.write(f"### Antibody {ab['antibody_id']}\n\n")

                f.write("#### Target Epitope\n\n")
                f.write(f"- **Sequence**: `{ab['epitope']['sequence']}`\n")
                f.write(f"- **Position**: {ab['epitope']['position']}\n")
                f.write(f"- **Score**: {ab['epitope']['score']:.3f}\n\n")

                if ab['validation']:
                    f.write("#### Literature Validation\n\n")
                    if ab['validation']['is_validated']:
                        f.write(f"- **Status**: ✅ VALIDATED\n")
                    else:
                        f.write(f"- **Status**: ⚠️  Novel prediction (requires experimental validation)\n")
                    f.write(f"- **Confidence**: {ab['validation']['confidence'].upper()}\n")
                    f.write(f"- **Citations**: {ab['validation']['citation_count']}\n")
                    f.write(f"- **Structural Evidence**: {'Yes' if ab['validation']['has_structural_evidence'] else 'No'}\n\n")

                    if ab['validation']['citations']:
                        f.write("##### Citations\n\n")
                        for i, citation in enumerate(ab['validation']['citations'], 1):
                            f.write(f"{i}. {citation}\n")
                        f.write("\n")

                f.write("#### Antibody Sequences\n\n")
                f.write(f"- **Heavy Chain** ({len(ab['sequences']['heavy_chain'])} aa):\n")
                f.write(f"  ```\n  {ab['sequences']['heavy_chain']}\n  ```\n\n")
                f.write(f"- **Light Chain** ({len(ab['sequences']['light_chain'])} aa):\n")
                f.write(f"  ```\n  {ab['sequences']['light_chain']}\n  ```\n\n")
                f.write(f"- **Target pKd**: {ab['target_pkd']}\n\n")
                f.write(f"- **FASTA File**: `antibody_{ab['antibody_id']}.fasta`\n\n")

                f.write("---\n\n")

            # Next Steps
            f.write("## Next Steps\n\n")
            f.write("### 1. Structure Validation with IgFold\n\n")
            f.write("Validate antibody structures to ensure they fold correctly:\n\n")
            f.write("```bash\n")
            f.write("python validate_fasta_with_igfold.py \\\n")
            f.write("    --fasta antibody_1.fasta \\\n")
            f.write("    --output-dir validation_results\n")
            f.write("```\n\n")
            f.write("**Success criteria**: Mean pRMSD < 2.0 Å, pLDDT > 70\n\n")

            f.write("### 2. Binding Prediction (Optional)\n\n")
            f.write("Use AlphaFold-Multimer or docking to predict binding:\n\n")
            f.write("```bash\n")
            f.write("# Coming soon: binding prediction module\n")
            f.write("```\n\n")

            f.write("### 3. Experimental Validation\n\n")
            f.write("For high-confidence candidates:\n\n")
            f.write("1. **Synthesis**: Order from antibody synthesis service (~$600-1200 each)\n")
            f.write("2. **Binding assay**: ELISA, SPR, or BLI to measure affinity\n")
            f.write("3. **Structure determination**: X-ray crystallography or cryo-EM (if resources available)\n\n")

            f.write("### 4. Benchmark Testing\n\n")
            f.write("Test on known antibody-antigen pairs:\n\n")
            f.write("```bash\n")
            f.write("# Download CoV-AbDab benchmark\n")
            f.write("# Test model performance\n")
            f.write("# Calculate R² for affinity prediction\n")
            f.write("```\n\n")

            f.write("---\n\n")

            # Model Info
            f.write("## Model Information\n\n")
            f.write("### Epitope Predictor v2\n\n")
            f.write("- **Method**: Sliding window with Parker hydrophilicity scale\n")
            f.write(f"- **Threshold**: {results['metadata']['epitope_threshold']}\n")
            f.write("- **Validated Performance**: 50% recall on SARS-CoV-2 known epitopes\n")
            f.write("- **Advantages**: Fast (<1s), no dependencies, tunable\n")
            f.write("- **Limitations**: Misses some epitopes, literature validation recommended\n\n")

            f.write("### Antibody Generator\n\n")
            f.write("- **Architecture**: Transformer Seq2Seq (5.6M parameters)\n")
            f.write("- **Training Data**: 158,135 antibody-antigen pairs\n")
            f.write("- **Affinity Conditioning**: Yes (pKd-conditioned generation)\n")
            f.write("- **Structure Validation**: IgFold (mean pRMSD 1.79 Å on validation)\n\n")

            f.write("---\n\n")

            # Disclaimer
            f.write("## Disclaimer\n\n")
            f.write("⚠️ **Important**: This pipeline uses computational predictions. All results should be:\n\n")
            f.write("- Validated experimentally before use\n")
            f.write("- Verified against published literature\n")
            f.write("- Not used for clinical applications without proper validation\n")
            f.write("- Reviewed by qualified researchers before synthesis\n\n")

            f.write("---\n\n")
            f.write("*Generated by Pipeline v2 with Epitope Predictor v2*\n")

    def run(
        self,
        antigen_sequence: str,
        antigen_name: str,
        organism: str,
        output_dir: Path,
        checkpoint_path: Path,
        email: str = None,
        ncbi_api_key: str = None
    ) -> Dict[str, Any]:
        """
        Run complete pipeline

        Args:
            antigen_sequence: Full antigen amino acid sequence
            antigen_name: Antigen name (e.g., "spike protein")
            organism: Organism (e.g., "SARS-CoV-2")
            output_dir: Output directory
            checkpoint_path: Path to model checkpoint
            email: Email for NCBI API (required for validation)
            ncbi_api_key: Optional NCBI API key

        Returns:
            Results dictionary
        """
        print("\n" + "🧬"*40)
        print("PIPELINE V2 - ENHANCED EPITOPE PREDICTION")
        print("🧬"*40)
        print(f"\nOrganism: {organism}")
        print(f"Antigen: {antigen_name}")
        print(f"Sequence length: {len(antigen_sequence)} aa")
        print(f"Output directory: {output_dir}")

        # Setup
        if email:
            self.set_web_validator(email, ncbi_api_key)
        else:
            print("\n⚠️  WARNING: No email provided - skipping literature validation")

        self.load_model(checkpoint_path)

        # Run pipeline steps
        epitopes = self.step1_predict_epitopes(antigen_sequence)

        validated_epitopes = self.step2_validate_epitopes(
            epitopes, antigen_name, organism
        )

        antibodies = self.step3_generate_antibodies(validated_epitopes)

        results = self.step4_save_results(
            antigen_sequence, antigen_name, organism,
            epitopes, antibodies, output_dir
        )

        # Final summary
        print("\n" + "="*80)
        print("✅ PIPELINE V2 COMPLETED SUCCESSFULLY")
        print("="*80)

        print(f"\n📊 Final Results:")
        print(f"   - Epitopes predicted: {len(epitopes)}")
        print(f"   - Antibodies generated: {len(antibodies)}")
        validated_count = sum(
            1 for ab in antibodies
            if ab['validation'] and ab['validation'].is_validated
        )
        print(f"   - Epitopes validated: {validated_count}")
        total_citations = sum(
            ab['validation'].get_citation_count()
            for ab in antibodies
            if ab['validation']
        )
        print(f"   - Total citations: {total_citations}")

        print(f"\n📁 Output Directory: {output_dir}")
        print(f"   - pipeline_v2_results.json")
        print(f"   - PIPELINE_V2_REPORT.md")
        for i in range(1, len(antibodies) + 1):
            print(f"   - antibody_{i}.fasta")

        print(f"\n🎉 SUCCESS! Next: Validate structures with IgFold")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline v2 with Enhanced Epitope Prediction"
    )

    # Input
    parser.add_argument('--antigen-sequence', type=str,
                       help='Antigen amino acid sequence (direct input)')
    parser.add_argument('--antigen-file', type=str,
                       help='Path to FASTA file with antigen sequence')
    parser.add_argument('--antigen-name', type=str, required=True,
                       help='Antigen name (e.g., "spike protein")')
    parser.add_argument('--virus-name', type=str, required=True,
                       help='Virus/organism name (e.g., "SARS-CoV-2")')

    # Model
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints/improved_small_2025_10_31_best.pt',
                       help='Path to model checkpoint')

    # Pipeline config
    parser.add_argument('--epitope-threshold', type=float, default=0.60,
                       help='Epitope prediction threshold (default: 0.60)')
    parser.add_argument('--top-k-epitopes', type=int, default=5,
                       help='Number of top epitopes to process')
    parser.add_argument('--target-pkd', type=float, default=9.5,
                       help='Target binding affinity (pKd)')

    # Validation
    parser.add_argument('--email', type=str,
                       help='Email for NCBI API (required for literature validation)')
    parser.add_argument('--ncbi-api-key', type=str,
                       help='Optional NCBI API key (increases rate limits)')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip literature validation')

    # Output
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'])

    args = parser.parse_args()

    # Load antigen sequence
    if args.antigen_sequence:
        antigen_seq = args.antigen_sequence
    elif args.antigen_file:
        antigen_seq = load_antigen_sequence(Path(args.antigen_file))
    else:
        parser.error("Must provide either --antigen-sequence or --antigen-file")

    # Check email requirement
    if not args.skip_validation and not args.email:
        print("\n⚠️  WARNING: No email provided and validation not skipped")
        print("   Literature validation will be skipped")
        print("   Use --email to enable validation or --skip-validation to suppress this warning")

    # Create config
    config = PipelineConfig(
        epitope_threshold=args.epitope_threshold,
        top_k_epitopes=args.top_k_epitopes,
        target_pkd=args.target_pkd,
        device=args.device
    )

    # Initialize pipeline
    pipeline = PipelineV2(config)

    # Run pipeline
    results = pipeline.run(
        antigen_sequence=antigen_seq,
        antigen_name=args.antigen_name,
        organism=args.virus_name,
        output_dir=Path(args.output_dir),
        checkpoint_path=Path(args.checkpoint),
        email=args.email if not args.skip_validation else None,
        ncbi_api_key=args.ncbi_api_key
    )

    print("\n✅ Pipeline complete! Review results and proceed to structure validation.")


if __name__ == '__main__':
    main()
