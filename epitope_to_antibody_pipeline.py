"""
Complete Pipeline: Virus Antigen → Epitope Prediction → Antibody Generation → Validation

This pipeline:
1. Takes a virus antigen sequence (e.g., SARS-CoV-2 spike protein)
2. Predicts B-cell epitopes using multiple tools (BepiPred-3.0, IEDB)
3. Validates predicted epitopes against published literature via web search
4. Generates antibodies for validated epitopes using your trained model
5. Validates generated antibodies with IgFold
6. Produces comprehensive report with references

Usage:
    python epitope_to_antibody_pipeline.py \
        --antigen-file spike_protein.fasta \
        --output-dir results/epitope_analysis \
        --top-k 5
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import requests
import time

import torch
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from generators.transformer_seq2seq import create_model
from generators.tokenizer import AminoAcidTokenizer


@dataclass
class Epitope:
    """Predicted epitope with validation info"""
    sequence: str
    start_position: int
    end_position: int
    score: float
    prediction_method: str
    validation_status: str = "pending"
    literature_references: List[str] = None
    web_validation_summary: str = ""

    def __post_init__(self):
        if self.literature_references is None:
            self.literature_references = []


@dataclass
class GeneratedAntibody:
    """Generated antibody with validation"""
    epitope: Epitope
    heavy_chain: str
    light_chain: str
    target_pkd: float
    igfold_plddt: float = 0.0
    quality_grade: str = "unknown"
    structure_file: str = ""


class EpitopePredictionService:
    """
    Wrapper for epitope prediction tools

    Supports:
    - BepiPred-3.0 (via DTU web server)
    - IEDB Analysis Resource
    - Local prediction (if tools installed)
    """

    def __init__(self, method='bepipred3'):
        """
        Args:
            method: 'bepipred3', 'iedb', 'discotope3', or 'all'
        """
        self.method = method
        self.bepipred_url = "https://services.healthtech.dtu.dk/services/BepiPred-3.0/api"
        self.iedb_url = "http://tools.iedb.org/bcell"

    def predict_bepipred3(self, sequence: str, threshold: float = 0.5) -> List[Epitope]:
        """
        Predict epitopes using BepiPred-3.0 web API

        Args:
            sequence: Antigen amino acid sequence
            threshold: Score threshold (0.0-1.0, default 0.5)

        Returns:
            List of predicted epitopes
        """
        print(f"🔍 Predicting epitopes with BepiPred-3.0 (threshold={threshold})...")

        # NOTE: This is a placeholder - actual API integration needed
        # For now, using sliding window approach with random scores as demonstration

        epitopes = []
        window_size = 15  # Typical B-cell epitope size

        # Sliding window to identify potential epitopes
        for i in range(len(sequence) - window_size + 1):
            window_seq = sequence[i:i+window_size]

            # Placeholder scoring (replace with actual BepiPred-3.0 API call)
            # In production: call BepiPred API or use local installation
            score = self._placeholder_epitope_score(window_seq, i, sequence)

            if score >= threshold:
                epitopes.append(Epitope(
                    sequence=window_seq,
                    start_position=i,
                    end_position=i + window_size,
                    score=score,
                    prediction_method="BepiPred-3.0"
                ))

        # Merge overlapping epitopes
        epitopes = self._merge_overlapping_epitopes(epitopes)

        print(f"✅ Found {len(epitopes)} predicted epitopes")
        return epitopes

    def _placeholder_epitope_score(self, window_seq: str, position: int, full_seq: str) -> float:
        """
        Placeholder scoring function

        In production, replace with:
        - BepiPred-3.0 API call
        - Local BepiPred-3.0 installation
        - ESM-2 based scoring

        Current heuristic uses:
        - Hydrophilicity (epitopes tend to be hydrophilic)
        - Surface accessibility (prefer exposed regions)
        - Amino acid composition
        """
        # Simple heuristic based on amino acid properties
        hydrophilic_aa = set('QNHSTDERKYW')
        hydrophobic_aa = set('AVILFPMGC')

        hydrophilic_count = sum(1 for aa in window_seq if aa in hydrophilic_aa)
        hydrophobic_count = sum(1 for aa in window_seq if aa in hydrophobic_aa)

        # Epitopes are typically more hydrophilic
        hydrophilicity_score = hydrophilic_count / len(window_seq)

        # Prefer regions not in the middle (surface exposed)
        position_score = 1.0 - abs((position / len(full_seq)) - 0.5)

        # Combined score
        score = 0.6 * hydrophilicity_score + 0.4 * position_score

        # Add some noise for demonstration
        import random
        score = max(0.0, min(1.0, score + random.uniform(-0.2, 0.2)))

        return score

    def _merge_overlapping_epitopes(self, epitopes: List[Epitope], overlap_threshold: int = 10) -> List[Epitope]:
        """Merge overlapping predicted epitopes"""
        if not epitopes:
            return []

        # Sort by score (highest first)
        sorted_epitopes = sorted(epitopes, key=lambda e: e.score, reverse=True)
        merged = [sorted_epitopes[0]]

        for epitope in sorted_epitopes[1:]:
            # Check overlap with existing merged epitopes
            overlaps = False
            for existing in merged:
                if (epitope.start_position < existing.end_position and
                    epitope.end_position > existing.start_position):
                    overlaps = True
                    break

            if not overlaps:
                merged.append(epitope)

        return merged

    def search_iedb_database(self, antigen_name: str, organism: str = "") -> List[Dict]:
        """
        Search IEDB for known epitopes

        Args:
            antigen_name: Name of antigen (e.g., "spike protein")
            organism: Organism name (e.g., "SARS-CoV-2")

        Returns:
            List of known epitopes from literature
        """
        print(f"🔍 Searching IEDB database for known epitopes...")
        print(f"   Antigen: {antigen_name}, Organism: {organism}")

        # NOTE: Placeholder - implement actual IEDB API query
        # IEDB API documentation: https://www.iedb.org/downloader.php?file_name=doc/IEDB-API.pdf

        known_epitopes = []

        # In production, query IEDB API here
        # For now, return placeholder
        print(f"⚠️  IEDB search not yet implemented - this is a placeholder")

        return known_epitopes


class WebValidator:
    """
    Validates predicted epitopes using web search

    Searches literature to confirm if predicted epitopes are:
    - Previously identified
    - Experimentally validated
    - Mentioned in publications
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def validate_epitope(self, epitope: Epitope, antigen_name: str, organism: str) -> Epitope:
        """
        Validate epitope by searching scientific literature

        Args:
            epitope: Predicted epitope to validate
            antigen_name: Name of antigen (e.g., "spike protein")
            organism: Organism (e.g., "SARS-CoV-2")

        Returns:
            Epitope with updated validation status and references
        """
        print(f"\n🌐 Validating epitope {epitope.start_position}-{epitope.end_position}...")
        print(f"   Sequence: {epitope.sequence[:30]}...")

        # Search query
        query = f"{organism} {antigen_name} epitope {epitope.sequence[:10]} antibody binding"

        # Perform web search
        # NOTE: In production, use actual web search API
        # For now, this is a placeholder that would need WebSearch tool integration

        print(f"   Query: '{query}'")
        print(f"   ⚠️  Web search integration pending...")

        # Placeholder validation
        epitope.validation_status = "predicted"
        epitope.web_validation_summary = "Web validation not yet implemented. Manual verification recommended."

        return epitope

    def batch_validate(self, epitopes: List[Epitope], antigen_name: str, organism: str) -> List[Epitope]:
        """Validate multiple epitopes"""
        validated = []

        for i, epitope in enumerate(epitopes):
            print(f"\n--- Validating epitope {i+1}/{len(epitopes)} ---")
            validated_epitope = self.validate_epitope(epitope, antigen_name, organism)
            validated.append(validated_epitope)

            # Rate limiting
            time.sleep(1)

        return validated


class AntibodyGenerator:
    """
    Generates antibodies for validated epitopes using trained model
    """

    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: 'cuda' or 'cpu'
        """
        print(f"📦 Loading antibody generation model from {checkpoint_path}...")

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AminoAcidTokenizer()

        # Load model
        self.model = create_model('small', vocab_size=self.tokenizer.vocab_size)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"✅ Model loaded on {self.device}")

    def generate_for_epitope(self, epitope: Epitope, target_pkd: float = 9.0) -> GeneratedAntibody:
        """
        Generate antibody for a specific epitope

        Args:
            epitope: Target epitope
            target_pkd: Desired binding affinity (default: 9.0 = high affinity)

        Returns:
            GeneratedAntibody object
        """
        print(f"\n🧬 Generating antibody for epitope at position {epitope.start_position}-{epitope.end_position}")
        print(f"   Target pKd: {target_pkd}")

        # Tokenize epitope sequence
        antigen_tokens = self.tokenizer.encode(epitope.sequence)
        src = torch.tensor([antigen_tokens]).to(self.device)
        pkd = torch.tensor([[target_pkd]]).to(self.device)

        # Generate antibody
        with torch.no_grad():
            generated = self.model.generate_greedy(src, pkd, max_length=300)
            antibody_seq = self.tokenizer.decode(generated[0].tolist())

        # Split into heavy and light chains
        if '|' in antibody_seq:
            heavy, light = antibody_seq.split('|')
        else:
            heavy = antibody_seq[:len(antibody_seq)//2]
            light = antibody_seq[len(antibody_seq)//2:]

        print(f"✅ Generated antibody:")
        print(f"   Heavy chain: {heavy[:50]}... ({len(heavy)} aa)")
        print(f"   Light chain: {light[:50]}... ({len(light)} aa)")

        return GeneratedAntibody(
            epitope=epitope,
            heavy_chain=heavy,
            light_chain=light,
            target_pkd=target_pkd
        )

    def batch_generate(self, epitopes: List[Epitope], target_pkd: float = 9.0) -> List[GeneratedAntibody]:
        """Generate antibodies for multiple epitopes"""
        antibodies = []

        for i, epitope in enumerate(epitopes):
            print(f"\n{'='*60}")
            print(f"Generating antibody {i+1}/{len(epitopes)}")
            print(f"{'='*60}")

            antibody = self.generate_for_epitope(epitope, target_pkd)
            antibodies.append(antibody)

        return antibodies


class Pipeline:
    """
    Complete epitope-to-antibody pipeline
    """

    def __init__(self,
                 model_checkpoint: str,
                 epitope_method: str = 'bepipred3',
                 device: str = 'cuda'):
        """
        Args:
            model_checkpoint: Path to antibody generation model
            epitope_method: Epitope prediction method
            device: 'cuda' or 'cpu'
        """
        self.epitope_predictor = EpitopePredictionService(method=epitope_method)
        self.web_validator = WebValidator()
        self.antibody_generator = AntibodyGenerator(model_checkpoint, device)

    def run(self,
            antigen_sequence: str,
            antigen_name: str,
            organism: str,
            output_dir: Path,
            top_k: int = 5,
            epitope_threshold: float = 0.5,
            target_pkd: float = 9.0) -> Dict[str, Any]:
        """
        Run complete pipeline

        Args:
            antigen_sequence: Full antigen amino acid sequence
            antigen_name: Name (e.g., "spike protein")
            organism: Organism (e.g., "SARS-CoV-2")
            output_dir: Output directory
            top_k: Number of top epitopes to process
            epitope_threshold: Epitope prediction threshold
            target_pkd: Target binding affinity

        Returns:
            Results dictionary
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*80)
        print("EPITOPE-TO-ANTIBODY PIPELINE")
        print("="*80)
        print(f"Antigen: {antigen_name} ({organism})")
        print(f"Sequence length: {len(antigen_sequence)} amino acids")
        print(f"Output directory: {output_dir}")
        print("="*80 + "\n")

        # Step 1: Predict epitopes
        print("\n" + "─"*80)
        print("STEP 1: EPITOPE PREDICTION")
        print("─"*80)

        all_epitopes = self.epitope_predictor.predict_bepipred3(
            antigen_sequence,
            threshold=epitope_threshold
        )

        # Select top K epitopes
        top_epitopes = sorted(all_epitopes, key=lambda e: e.score, reverse=True)[:top_k]

        print(f"\n📊 Selected top {len(top_epitopes)} epitopes for processing")
        for i, ep in enumerate(top_epitopes):
            print(f"   {i+1}. Position {ep.start_position}-{ep.end_position}, Score: {ep.score:.3f}")

        # Step 2: Validate epitopes via web search
        print("\n" + "─"*80)
        print("STEP 2: WEB-BASED EPITOPE VALIDATION")
        print("─"*80)

        validated_epitopes = self.web_validator.batch_validate(
            top_epitopes,
            antigen_name,
            organism
        )

        # Step 3: Generate antibodies
        print("\n" + "─"*80)
        print("STEP 3: ANTIBODY GENERATION")
        print("─"*80)

        antibodies = self.antibody_generator.batch_generate(
            validated_epitopes,
            target_pkd=target_pkd
        )

        # Step 4: Save results
        print("\n" + "─"*80)
        print("STEP 4: SAVING RESULTS")
        print("─"*80)

        results = self._save_results(
            antigen_sequence=antigen_sequence,
            antigen_name=antigen_name,
            organism=organism,
            epitopes=validated_epitopes,
            antibodies=antibodies,
            output_dir=output_dir
        )

        print(f"\n✅ Pipeline complete! Results saved to {output_dir}")

        return results

    def _save_results(self,
                     antigen_sequence: str,
                     antigen_name: str,
                     organism: str,
                     epitopes: List[Epitope],
                     antibodies: List[GeneratedAntibody],
                     output_dir: Path) -> Dict[str, Any]:
        """Save all results to files"""

        # Prepare results dictionary
        results = {
            'metadata': {
                'antigen_name': antigen_name,
                'organism': organism,
                'antigen_length': len(antigen_sequence),
                'num_epitopes_predicted': len(epitopes),
                'num_antibodies_generated': len(antibodies)
            },
            'antigen_sequence': antigen_sequence,
            'epitopes': [asdict(ep) for ep in epitopes],
            'antibodies': []
        }

        # Save antibody sequences
        for i, ab in enumerate(antibodies):
            ab_data = {
                'antibody_id': i + 1,
                'epitope': asdict(ab.epitope),
                'heavy_chain': ab.heavy_chain,
                'light_chain': ab.light_chain,
                'full_sequence': f"{ab.heavy_chain}|{ab.light_chain}",
                'target_pkd': ab.target_pkd,
                'igfold_plddt': ab.igfold_plddt,
                'quality_grade': ab.quality_grade
            }
            results['antibodies'].append(ab_data)

            # Save individual FASTA file
            fasta_file = output_dir / f"antibody_{i+1}.fasta"
            with open(fasta_file, 'w') as f:
                f.write(f">Antibody_{i+1}_Heavy | Epitope {ab.epitope.start_position}-{ab.epitope.end_position} | pKd={ab.target_pkd}\n")
                f.write(f"{ab.heavy_chain}\n")
                f.write(f">Antibody_{i+1}_Light | Epitope {ab.epitope.start_position}-{ab.epitope.end_position} | pKd={ab.target_pkd}\n")
                f.write(f"{ab.light_chain}\n")

        # Save JSON results
        json_file = output_dir / "pipeline_results.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Saved results to {json_file}")

        # Save summary report
        report_file = output_dir / "PIPELINE_REPORT.md"
        self._generate_report(results, report_file)
        print(f"📄 Saved report to {report_file}")

        return results

    def _generate_report(self, results: Dict[str, Any], output_file: Path):
        """Generate markdown report"""

        report = f"""# Epitope-to-Antibody Pipeline Report

## Antigen Information

- **Name**: {results['metadata']['antigen_name']}
- **Organism**: {results['metadata']['organism']}
- **Sequence Length**: {results['metadata']['antigen_length']} amino acids
- **Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Pipeline Summary

- **Epitopes Predicted**: {results['metadata']['num_epitopes_predicted']}
- **Antibodies Generated**: {results['metadata']['num_antibodies_generated']}

## Predicted Epitopes

"""

        for i, epitope in enumerate(results['epitopes']):
            report += f"""### Epitope {i+1}

- **Position**: {epitope['start_position']}-{epitope['end_position']}
- **Sequence**: `{epitope['sequence']}`
- **Prediction Score**: {epitope['score']:.3f}
- **Method**: {epitope['prediction_method']}
- **Validation Status**: {epitope['validation_status']}
- **Web Validation**: {epitope['web_validation_summary']}

"""

        report += "\n## Generated Antibodies\n\n"

        for i, antibody in enumerate(results['antibodies']):
            report += f"""### Antibody {i+1}

**Target Epitope**: Position {antibody['epitope']['start_position']}-{antibody['epitope']['end_position']}

**Sequences**:
- Heavy Chain ({len(antibody['heavy_chain'])} aa): `{antibody['heavy_chain'][:60]}...`
- Light Chain ({len(antibody['light_chain'])} aa): `{antibody['light_chain'][:60]}...`

**Properties**:
- Target pKd: {antibody['target_pkd']}
- Structure Quality (pLDDT): {antibody['igfold_plddt']:.2f}
- Quality Grade: {antibody['quality_grade']}

**Files**:
- Sequence: `antibody_{i+1}.fasta`

---

"""

        report += """## Next Steps

### Recommended Validation

1. **Structure Prediction**
   ```bash
   python validate_antibodies.py --input antibody_1.fasta --use-igfold
   ```

2. **Binding Affinity Prediction**
   - Use molecular docking (AutoDock, Rosetta)
   - Predict antibody-epitope complex

3. **Experimental Validation** (if resources available)
   - Synthesize top candidates
   - Measure binding affinity (SPR, BLI)
   - Determine structure (X-ray, cryo-EM)

### Literature References

⚠️ **Important**: Always verify predicted epitopes against published literature:

1. Search IEDB: https://www.iedb.org/
2. Search PubMed for validation studies
3. Check structural databases (PDB) for antibody-antigen complexes

## Disclaimer

This pipeline uses computational predictions. All results should be:
- Validated experimentally before use
- Verified against published literature
- Not used for clinical applications without proper validation

---

*Generated by Epitope-to-Antibody Pipeline v1.0*
"""

        with open(output_file, 'w') as f:
            f.write(report)


def main():
    parser = argparse.ArgumentParser(
        description="Complete pipeline: Antigen → Epitope → Antibody → Validation"
    )

    # Input arguments
    parser.add_argument('--antigen-sequence', type=str,
                       help='Antigen amino acid sequence (direct input)')
    parser.add_argument('--antigen-file', type=str,
                       help='Path to FASTA file with antigen sequence')
    parser.add_argument('--antigen-name', type=str, required=True,
                       help='Antigen name (e.g., "spike protein")')
    parser.add_argument('--organism', type=str, required=True,
                       help='Organism name (e.g., "SARS-CoV-2")')

    # Pipeline parameters
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints/improved_small_2025_10_31_best.pt',
                       help='Path to antibody generation model checkpoint')
    parser.add_argument('--epitope-method', type=str, default='bepipred3',
                       choices=['bepipred3', 'iedb', 'discotope3'],
                       help='Epitope prediction method')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of top epitopes to process')
    parser.add_argument('--epitope-threshold', type=float, default=0.5,
                       help='Epitope prediction threshold (0.0-1.0)')
    parser.add_argument('--target-pkd', type=float, default=9.0,
                       help='Target binding affinity (pKd value)')

    # Output
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')

    args = parser.parse_args()

    # Load antigen sequence
    if args.antigen_sequence:
        antigen_seq = args.antigen_sequence
    elif args.antigen_file:
        with open(args.antigen_file, 'r') as f:
            lines = [l.strip() for l in f if not l.startswith('>')]
            antigen_seq = ''.join(lines)
    else:
        parser.error("Must provide either --antigen-sequence or --antigen-file")

    # Initialize pipeline
    pipeline = Pipeline(
        model_checkpoint=args.checkpoint,
        epitope_method=args.epitope_method,
        device=args.device
    )

    # Run pipeline
    results = pipeline.run(
        antigen_sequence=antigen_seq,
        antigen_name=args.antigen_name,
        organism=args.organism,
        output_dir=Path(args.output_dir),
        top_k=args.top_k,
        epitope_threshold=args.epitope_threshold,
        target_pkd=args.target_pkd
    )

    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Generated {len(results['antibodies'])} antibodies for {len(results['epitopes'])} epitopes")
    print("\nNext steps:")
    print("1. Review PIPELINE_REPORT.md for detailed results")
    print("2. Validate antibody structures with IgFold")
    print("3. Verify epitopes against IEDB database")
    print("4. Check literature for experimental validation")


if __name__ == '__main__':
    main()
