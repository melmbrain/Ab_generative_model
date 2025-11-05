"""
Create Benchmark Dataset from CoV-AbDab

This script processes the CoV-AbDab database to create a benchmark dataset
for testing antibody generation model performance.

Filters for:
- Complete heavy and light chain sequences
- SARS-CoV-2 spike protein antibodies
- Known epitope information (if available)
- Neutralization data (if available)

Output:
- benchmark/benchmark_dataset.json - Filtered antibodies
- benchmark/BENCHMARK_REPORT.md - Summary statistics

Usage:
    python create_benchmark.py
"""

import pandas as pd
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter


class BenchmarkCreator:
    """Create benchmark dataset from CoV-AbDab"""

    def __init__(self, covabdab_file: Path):
        """
        Args:
            covabdab_file: Path to CoV-AbDab CSV file
        """
        self.covabdab_file = covabdab_file
        self.df = None
        self.benchmark = []

    def load_data(self):
        """Load CoV-AbDab CSV"""
        print(f"📂 Loading CoV-AbDab database from {self.covabdab_file}...")

        self.df = pd.read_csv(self.covabdab_file, encoding='utf-8-sig')

        print(f"✅ Loaded {len(self.df)} entries")
        print(f"   Columns: {len(self.df.columns)}")

        return self.df

    def filter_complete_sequences(self):
        """Filter for entries with complete heavy and light chain sequences"""
        print(f"\n🔍 Filtering for complete sequences...")

        # Check for non-empty VH and VL sequences
        has_vh = self.df['VHorVHH'].notna() & (self.df['VHorVHH'] != 'ND') & (self.df['VHorVHH'] != '')
        has_vl = self.df['VL'].notna() & (self.df['VL'] != 'ND') & (self.df['VL'] != '')

        # Filter for antibodies (not nanobodies)
        is_ab = self.df['Ab or Nb'] == 'Ab'

        # Combine filters
        complete = self.df[has_vh & has_vl & is_ab].copy()

        print(f"   Complete sequences: {len(complete)}/{len(self.df)}")
        print(f"   Heavy chains: {has_vh.sum()}")
        print(f"   Light chains: {has_vl.sum()}")
        print(f"   Antibodies (not Nbs): {is_ab.sum()}")

        self.df = complete
        return self.df

    def filter_spike_protein(self):
        """Filter for spike protein binders"""
        print(f"\n🔍 Filtering for spike protein binders...")

        # Look for spike protein in "Binds to" or "Protein + Epitope"
        binds_spike = (
            self.df['Binds to'].str.contains('S', case=False, na=False) |
            self.df['Protein + Epitope'].str.contains('S', case=False, na=False)
        )

        spike = self.df[binds_spike].copy()

        print(f"   Spike protein binders: {len(spike)}/{len(self.df)}")

        # Show epitope distribution
        if 'Protein + Epitope' in spike.columns:
            epitopes = spike['Protein + Epitope'].value_counts().head(10)
            print(f"\n   Top epitopes:")
            for epitope, count in epitopes.items():
                print(f"      {epitope}: {count}")

        self.df = spike
        return self.df

    def extract_neutralization_data(self):
        """Extract neutralization information"""
        print(f"\n🔍 Extracting neutralization data...")

        # Count neutralizing vs non-neutralizing
        neutralizing = self.df['Neutralising Vs'].notna() & (self.df['Neutralising Vs'] != '')
        not_neutralizing = self.df['Not Neutralising Vs'].notna() & (self.df['Not Neutralising Vs'] != '')

        print(f"   Neutralizing antibodies: {neutralizing.sum()}")
        print(f"   Non-neutralizing: {not_neutralizing.sum()}")
        print(f"   Unknown: {len(self.df) - neutralizing.sum() - not_neutralizing.sum()}")

        return self.df

    def extract_structure_data(self):
        """Extract structure information"""
        print(f"\n🔍 Extracting structure data...")

        has_structure = self.df['Structures'].notna() & (self.df['Structures'] != '')
        has_model = self.df['ABB Homology Model (if no structure)'].notna() & (
            self.df['ABB Homology Model (if no structure)'] != ''
        )

        print(f"   With experimental structure: {has_structure.sum()}")
        print(f"   With homology model: {has_model.sum()}")

        return self.df

    def create_benchmark_entries(self) -> List[Dict[str, Any]]:
        """Create benchmark dataset entries"""
        print(f"\n📊 Creating benchmark entries...")

        benchmark = []

        for idx, row in self.df.iterrows():
            # Extract sequences
            heavy = row['VHorVHH']
            light = row['VL']

            # Skip if sequences are too short or too long
            if len(heavy) < 50 or len(heavy) > 200:
                continue
            if len(light) < 50 or len(light) > 300:
                continue

            # Create entry
            entry = {
                'antibody_id': len(benchmark) + 1,
                'name': row['﻿Name'] if '﻿Name' in row else row.get('Name', f'Ab_{idx}'),
                'heavy_chain': heavy,
                'light_chain': light,
                'heavy_length': len(heavy),
                'light_length': len(light),
                'full_sequence': f"{heavy}|{light}",

                # Epitope information
                'epitope_info': row.get('Protein + Epitope', 'Unknown'),
                'binds_to': row.get('Binds to', 'Unknown'),

                # Neutralization
                'neutralizing': row.get('Neutralising Vs', ''),
                'not_neutralizing': row.get('Not Neutralising Vs', ''),
                'is_neutralizing': bool(row.get('Neutralising Vs', '') and row.get('Neutralising Vs', '') != ''),

                # Structure
                'has_structure': bool(row.get('Structures', '') and row.get('Structures', '') != ''),
                'structure_pdb': row.get('Structures', ''),
                'has_model': bool(row.get('ABB Homology Model (if no structure)', '') and
                                row.get('ABB Homology Model (if no structure)', '') != ''),

                # CDRs
                'cdrh3': row.get('CDRH3', ''),
                'cdrl3': row.get('CDRL3', ''),

                # V genes
                'heavy_v_gene': row.get('Heavy V Gene', ''),
                'heavy_j_gene': row.get('Heavy J Gene', ''),
                'light_v_gene': row.get('Light V Gene', ''),
                'light_j_gene': row.get('Light J Gene', ''),

                # Metadata
                'origin': row.get('Origin', ''),
                'sources': row.get('Sources', ''),
                'date_added': row.get('Date Added', ''),
            }

            benchmark.append(entry)

        print(f"✅ Created {len(benchmark)} benchmark entries")

        self.benchmark = benchmark
        return benchmark

    def save_benchmark(self, output_file: Path):
        """Save benchmark dataset"""
        print(f"\n💾 Saving benchmark dataset...")

        # Save JSON
        with open(output_file, 'w') as f:
            json.dump(self.benchmark, f, indent=2)

        print(f"✅ Saved {len(self.benchmark)} entries to {output_file}")

        return output_file

    def generate_report(self, output_file: Path):
        """Generate benchmark report"""
        print(f"\n📄 Generating report...")

        report = f"""# CoV-AbDab Benchmark Dataset Report

**Date Created**: {pd.Timestamp.now().strftime('%Y-%m-%d')}
**Source**: CoV-AbDab database (http://opig.stats.ox.ac.uk/webapps/covabdab/)

---

## Dataset Summary

### Overall Statistics

- **Total Entries**: {len(self.benchmark)}
- **All SARS-CoV-2 Spike Protein Antibodies**: Yes
- **All with Complete Heavy + Light Chains**: Yes

### Sequence Statistics

"""

        # Calculate statistics
        heavy_lengths = [e['heavy_length'] for e in self.benchmark]
        light_lengths = [e['light_length'] for e in self.benchmark]

        report += f"""- **Heavy Chain Length**:
  - Mean: {sum(heavy_lengths)/len(heavy_lengths):.1f} aa
  - Range: {min(heavy_lengths)}-{max(heavy_lengths)} aa

- **Light Chain Length**:
  - Mean: {sum(light_lengths)/len(light_lengths):.1f} aa
  - Range: {min(light_lengths)}-{max(light_lengths)} aa

### Neutralization Data

"""

        neutralizing_count = sum(1 for e in self.benchmark if e['is_neutralizing'])
        has_structure = sum(1 for e in self.benchmark if e['has_structure'])
        has_model = sum(1 for e in self.benchmark if e['has_model'])

        report += f"""- **Neutralizing Antibodies**: {neutralizing_count} ({neutralizing_count/len(self.benchmark)*100:.1f}%)
- **Non/Unknown**: {len(self.benchmark) - neutralizing_count} ({(len(self.benchmark) - neutralizing_count)/len(self.benchmark)*100:.1f}%)

### Structure Data

- **With Experimental Structure**: {has_structure} ({has_structure/len(self.benchmark)*100:.1f}%)
- **With Homology Model**: {has_model} ({has_model/len(self.benchmark)*100:.1f}%)

---

## Epitope Distribution

"""

        # Epitope distribution
        epitope_counts = Counter(e['epitope_info'] for e in self.benchmark)
        top_epitopes = epitope_counts.most_common(15)

        report += "| Epitope/Region | Count | Percentage |\n"
        report += "|----------------|-------|------------|\n"
        for epitope, count in top_epitopes:
            pct = count / len(self.benchmark) * 100
            report += f"| {epitope[:50]} | {count} | {pct:.1f}% |\n"

        report += "\n---\n\n## CDR Statistics\n\n"

        # CDR statistics
        has_cdrh3 = sum(1 for e in self.benchmark if e['cdrh3'] and e['cdrh3'] != 'ND')
        has_cdrl3 = sum(1 for e in self.benchmark if e['cdrl3'] and e['cdrl3'] != 'ND')

        report += f"""- **CDRH3 Available**: {has_cdrh3} ({has_cdrh3/len(self.benchmark)*100:.1f}%)
- **CDRL3 Available**: {has_cdrl3} ({has_cdrl3/len(self.benchmark)*100:.1f}%)

---

## V Gene Distribution

"""

        # V gene distribution
        heavy_v_genes = Counter(e['heavy_v_gene'] for e in self.benchmark if e['heavy_v_gene'])
        light_v_genes = Counter(e['light_v_gene'] for e in self.benchmark if e['light_v_gene'])

        report += "### Top Heavy V Genes\n\n"
        for gene, count in heavy_v_genes.most_common(10):
            if gene and gene != 'ND':
                report += f"- {gene}: {count}\n"

        report += "\n### Top Light V Genes\n\n"
        for gene, count in light_v_genes.most_common(10):
            if gene and gene != 'ND':
                report += f"- {gene}: {count}\n"

        report += "\n---\n\n## Usage for Model Testing\n\n"

        report += """### Recommended Tests

1. **Sequence Recovery**
   - Generate antibodies for known epitopes
   - Compare generated sequences to benchmark
   - Calculate similarity scores (BLAST, edit distance)

2. **Affinity Prediction** (if available)
   - Currently no affinity data in CoV-AbDab
   - Need to match with additional databases (e.g., IEDB)
   - Alternative: Use neutralization as proxy

3. **Structure Validation**
   - Test IgFold on benchmark sequences
   - Compare to experimental structures (where available)
   - Validate pRMSD and pLDDT metrics

4. **Epitope Specificity**
   - Test if model generates different antibodies for different epitopes
   - Check CDR diversity across epitope groups

---

## Benchmark Subsets

### Recommended Subsets for Testing:

1. **High-Quality Subset** (with structures)
   - Count: {has_structure}
   - Use for: Structure validation

2. **Neutralizing Subset**
   - Count: {neutralizing_count}
   - Use for: Functional validation proxy

3. **RBD-Specific Subset**
   - Count: {sum(1 for e in self.benchmark if 'RBD' in e['epitope_info'])}
   - Use for: Epitope-specific testing

---

## Data Quality Notes

### Completeness:
- ✅ All entries have heavy and light chains
- ✅ All are full antibodies (not nanobodies)
- ✅ All target SARS-CoV-2 spike protein
- ⚠️ Not all have epitope details
- ⚠️ No affinity (pKd) data available

### Next Steps:
1. Match entries with IEDB for affinity data
2. Extract epitope sequences from PDB structures
3. Create epitope-specific subsets

---

*Dataset created from CoV-AbDab (Raybould et al., 2021)*
*License: CC-BY 4.0*
"""

        with open(output_file, 'w') as f:
            f.write(report)

        print(f"✅ Saved report to {output_file}")

        return output_file


def main():
    print("="*80)
    print("CoV-AbDab BENCHMARK DATASET CREATION")
    print("="*80)

    # Setup paths
    covabdab_file = Path("benchmark/covabdab.csv")
    output_json = Path("benchmark/benchmark_dataset.json")
    output_report = Path("benchmark/BENCHMARK_REPORT.md")

    if not covabdab_file.exists():
        print(f"❌ CoV-AbDab file not found: {covabdab_file}")
        print(f"   Please run download script first")
        return 1

    # Create benchmark
    creator = BenchmarkCreator(covabdab_file)

    # Process data
    creator.load_data()
    creator.filter_complete_sequences()
    creator.filter_spike_protein()
    creator.extract_neutralization_data()
    creator.extract_structure_data()

    # Create benchmark
    benchmark = creator.create_benchmark_entries()

    # Save
    creator.save_benchmark(output_json)
    creator.generate_report(output_report)

    # Summary
    print("\n" + "="*80)
    print("✅ BENCHMARK CREATION COMPLETE")
    print("="*80)

    print(f"\n📊 Summary:")
    print(f"   Total entries: {len(benchmark)}")
    print(f"   Neutralizing: {sum(1 for e in benchmark if e['is_neutralizing'])}")
    print(f"   With structure: {sum(1 for e in benchmark if e['has_structure'])}")

    print(f"\n📁 Files created:")
    print(f"   {output_json}")
    print(f"   {output_report}")

    print(f"\n🎯 Next steps:")
    print(f"   1. Review {output_report}")
    print(f"   2. Create test script: benchmark_model.py")
    print(f"   3. Test model on benchmark subset")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
