"""
Analyze Training Data Coverage

This script analyzes your training data to understand:
1. How many antibody-antigen pairs
2. Which viruses are covered
3. Affinity distribution
4. Epitope characteristics
5. Identified gaps

Usage:
    python analyze_training_data.py --data data/training_data.csv
"""

import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import Counter


def analyze_dataset(data_path):
    """
    Comprehensive analysis of training data
    """
    print("="*80)
    print("TRAINING DATA ANALYSIS")
    print("="*80)

    # Try to load data
    data_path = Path(data_path)

    if not data_path.exists():
        print(f"\n❌ Error: File not found: {data_path}")
        print("\nLooking for training data in common locations...")

        # Search for potential training data files
        search_dirs = [
            Path('data'),
            Path('datasets'),
            Path('.'),
        ]

        potential_files = []
        for search_dir in search_dirs:
            if search_dir.exists():
                potential_files.extend(search_dir.glob('*.csv'))
                potential_files.extend(search_dir.glob('*.tsv'))
                potential_files.extend(search_dir.glob('*.json'))

        if potential_files:
            print("\nFound potential training data files:")
            for i, f in enumerate(potential_files, 1):
                print(f"  {i}. {f}")
            print("\nPlease specify the correct file with --data flag")
        else:
            print("\n⚠️  No training data files found!")
            print("\nExpected format (CSV):")
            print("  epitope_sequence,heavy_chain,light_chain,pkd")
            print("  YQAGSTPCNGVEG,EVQLV...,DIQMT...,9.5")

        return None

    # Load data
    try:
        if data_path.suffix == '.csv':
            df = pd.read_csv(data_path)
        elif data_path.suffix == '.tsv':
            df = pd.read_csv(data_path, sep='\t')
        elif data_path.suffix == '.json':
            df = pd.read_json(data_path)
        else:
            df = pd.read_csv(data_path)  # Try CSV by default

        print(f"\n✅ Loaded data from: {data_path}")
        print(f"   Total rows: {len(df):,}")

    except Exception as e:
        print(f"\n❌ Error loading file: {e}")
        return None

    # Basic statistics
    print(f"\n{'─'*80}")
    print("DATASET OVERVIEW")
    print(f"{'─'*80}")

    print(f"\nColumns found: {', '.join(df.columns)}")
    print(f"Dataset size: {len(df):,} pairs")

    # Identify key columns
    column_mapping = {}

    # Try to find epitope column
    epitope_candidates = ['epitope', 'epitope_sequence', 'antigen_sequence', 'peptide']
    for col in df.columns:
        if any(c in col.lower() for c in epitope_candidates):
            column_mapping['epitope'] = col
            break

    # Try to find heavy chain
    heavy_candidates = ['heavy', 'heavy_chain', 'vh', 'h_chain']
    for col in df.columns:
        if any(c in col.lower() for c in heavy_candidates):
            column_mapping['heavy'] = col
            break

    # Try to find light chain
    light_candidates = ['light', 'light_chain', 'vl', 'l_chain']
    for col in df.columns:
        if any(c in col.lower() for c in light_candidates):
            column_mapping['light'] = col
            break

    # Try to find affinity
    affinity_candidates = ['pkd', 'kd', 'affinity', 'binding_affinity']
    for col in df.columns:
        if any(c in col.lower() for c in affinity_candidates):
            column_mapping['affinity'] = col
            break

    # Try to find organism/virus
    organism_candidates = ['organism', 'virus', 'species', 'antigen_name']
    for col in df.columns:
        if any(c in col.lower() for c in organism_candidates):
            column_mapping['organism'] = col
            break

    print(f"\nColumn mapping:")
    for key, col in column_mapping.items():
        print(f"  {key}: {col}")

    if not column_mapping.get('epitope') or not column_mapping.get('heavy'):
        print("\n⚠️  Warning: Could not identify all required columns")
        print("   Expected: epitope, heavy_chain, light_chain")
        print("   Please check your data format")
        return df

    # Analyze epitopes
    print(f"\n{'─'*80}")
    print("EPITOPE ANALYSIS")
    print(f"{'─'*80}")

    epitope_col = column_mapping['epitope']

    if epitope_col in df.columns:
        # Remove NaN values
        df_clean = df.dropna(subset=[epitope_col])

        epitope_lengths = df_clean[epitope_col].str.len()

        print(f"\nEpitope Statistics:")
        print(f"  Total unique epitopes: {df_clean[epitope_col].nunique():,}")
        print(f"  Length range: {epitope_lengths.min()}-{epitope_lengths.max()} aa")
        print(f"  Mean length: {epitope_lengths.mean():.1f} aa")
        print(f"  Median length: {epitope_lengths.median():.0f} aa")

        # Length distribution
        print(f"\n  Length distribution:")
        length_bins = [0, 8, 12, 16, 20, 25, 100]
        labels = ['<8', '8-12', '12-16', '16-20', '20-25', '>25']
        length_dist = pd.cut(epitope_lengths, bins=length_bins, labels=labels).value_counts().sort_index()

        for label, count in length_dist.items():
            pct = count / len(df_clean) * 100
            print(f"    {label} aa: {count:6,} ({pct:5.1f}%)")

    # Analyze antibody sequences
    print(f"\n{'─'*80}")
    print("ANTIBODY SEQUENCE ANALYSIS")
    print(f"{'─'*80}")

    if column_mapping.get('heavy') in df.columns:
        heavy_col = column_mapping['heavy']
        df_clean = df.dropna(subset=[heavy_col])

        heavy_lengths = df_clean[heavy_col].str.len()

        print(f"\nHeavy Chain Statistics:")
        print(f"  Unique sequences: {df_clean[heavy_col].nunique():,}")
        print(f"  Length range: {heavy_lengths.min()}-{heavy_lengths.max()} aa")
        print(f"  Mean length: {heavy_lengths.mean():.1f} aa")
        print(f"  Median length: {heavy_lengths.median():.0f} aa")

    if column_mapping.get('light') in df.columns:
        light_col = column_mapping['light']
        df_clean = df.dropna(subset=[light_col])

        light_lengths = df_clean[light_col].str.len()

        print(f"\nLight Chain Statistics:")
        print(f"  Unique sequences: {df_clean[light_col].nunique():,}")
        print(f"  Length range: {light_lengths.min()}-{light_lengths.max()} aa")
        print(f"  Mean length: {light_lengths.mean():.1f} aa")
        print(f"  Median length: {light_lengths.median():.0f} aa")

    # Analyze affinity
    if column_mapping.get('affinity') in df.columns:
        print(f"\n{'─'*80}")
        print("AFFINITY ANALYSIS")
        print(f"{'─'*80}")

        affinity_col = column_mapping['affinity']
        df_affinity = df.dropna(subset=[affinity_col])

        print(f"\nAffinity Statistics:")
        print(f"  Pairs with affinity data: {len(df_affinity):,} ({len(df_affinity)/len(df)*100:.1f}%)")

        if len(df_affinity) > 0:
            affinities = df_affinity[affinity_col]

            # Check if it's Kd or pKd
            if affinities.mean() < 15:  # Likely pKd (0-15 range)
                print(f"  Metric: pKd (detected)")
                print(f"  Mean: {affinities.mean():.2f}")
                print(f"  Std: {affinities.std():.2f}")
                print(f"  Min: {affinities.min():.2f}")
                print(f"  Max: {affinities.max():.2f}")

                # Distribution
                print(f"\n  Affinity distribution:")
                bins = [0, 6, 7, 8, 9, 10, 15]
                labels = ['<6 (weak)', '6-7', '7-8', '8-9', '9-10', '>10 (strong)']
                affinity_dist = pd.cut(affinities, bins=bins, labels=labels).value_counts().sort_index()

                for label, count in affinity_dist.items():
                    pct = count / len(df_affinity) * 100
                    print(f"    {label}: {count:6,} ({pct:5.1f}%)")
            else:
                print(f"  Metric: Kd in nM (detected)")
                print(f"  Mean: {affinities.mean():.2f} nM")
                print(f"  Median: {affinities.median():.2f} nM")

    # Analyze organism/virus coverage
    if column_mapping.get('organism') in df.columns:
        print(f"\n{'─'*80}")
        print("ORGANISM/VIRUS COVERAGE")
        print(f"{'─'*80}")

        organism_col = column_mapping['organism']
        df_organism = df.dropna(subset=[organism_col])

        organism_counts = df_organism[organism_col].value_counts()

        print(f"\nTop 20 organisms/viruses:")
        for i, (org, count) in enumerate(organism_counts.head(20).items(), 1):
            pct = count / len(df_organism) * 100
            print(f"  {i:2d}. {org[:60]:<60} {count:6,} ({pct:5.1f}%)")

        # Check for viral coverage
        viral_keywords = ['virus', 'viral', 'SARS', 'influenza', 'HIV', 'hepatitis',
                         'dengue', 'zika', 'ebola', 'corona', 'COVID']

        is_viral = df_organism[organism_col].str.contains('|'.join(viral_keywords),
                                                          case=False, na=False)

        print(f"\nViral vs Non-viral:")
        print(f"  Viral antigens: {is_viral.sum():,} ({is_viral.sum()/len(df_organism)*100:.1f}%)")
        print(f"  Non-viral: {(~is_viral).sum():,} ({(~is_viral).sum()/len(df_organism)*100:.1f}%)")

        # Check for specific viruses of interest
        important_viruses = {
            'SARS-CoV-2': ['SARS-CoV-2', '2019-nCoV', 'COVID-19'],
            'HIV': ['HIV', 'Human immunodeficiency'],
            'Influenza': ['Influenza', 'H1N1', 'H3N2'],
            'Hepatitis': ['Hepatitis', 'HCV', 'HBV'],
            'Dengue': ['Dengue'],
            'Ebola': ['Ebola']
        }

        print(f"\nImportant viruses coverage:")
        for virus, keywords in important_viruses.items():
            has_virus = df_organism[organism_col].str.contains('|'.join(keywords),
                                                               case=False, na=False).any()
            count = df_organism[organism_col].str.contains('|'.join(keywords),
                                                           case=False, na=False).sum()

            if has_virus:
                print(f"  ✅ {virus}: {count:,} pairs")
            else:
                print(f"  ❌ {virus}: 0 pairs (MISSING)")

    # Identify gaps
    print(f"\n{'='*80}")
    print("IDENTIFIED GAPS & RECOMMENDATIONS")
    print(f"{'='*80}")

    gaps = []

    # Gap 1: Dataset size
    if len(df) < 1000:
        gaps.append({
            'severity': 'HIGH',
            'gap': f'Small dataset size ({len(df):,} pairs)',
            'recommendation': 'Augment with SAbDab (~8k structures) or CoV-AbDab (~10k SARS-CoV-2)',
            'priority': 1
        })
    elif len(df) < 5000:
        gaps.append({
            'severity': 'MEDIUM',
            'gap': f'Moderate dataset size ({len(df):,} pairs)',
            'recommendation': 'Consider augmenting for better generalization',
            'priority': 2
        })

    # Gap 2: Affinity data
    if column_mapping.get('affinity'):
        affinity_col = column_mapping['affinity']
        affinity_coverage = df[affinity_col].notna().sum() / len(df)

        if affinity_coverage < 0.3:
            gaps.append({
                'severity': 'HIGH',
                'gap': f'Low affinity coverage ({affinity_coverage*100:.1f}%)',
                'recommendation': 'Add pairs with measured Kd from literature',
                'priority': 1
            })
    else:
        gaps.append({
            'severity': 'CRITICAL',
            'gap': 'No affinity data found',
            'recommendation': 'Add affinity column (pKd) for conditional generation',
            'priority': 1
        })

    # Gap 3: Viral coverage
    if column_mapping.get('organism'):
        organism_col = column_mapping['organism']
        viral_keywords = ['virus', 'viral']
        is_viral = df[organism_col].str.contains('|'.join(viral_keywords), case=False, na=False)
        viral_coverage = is_viral.sum() / len(df)

        if viral_coverage < 0.3:
            gaps.append({
                'severity': 'HIGH',
                'gap': f'Low viral antigen coverage ({viral_coverage*100:.1f}%)',
                'recommendation': 'Add viral antibody data from CoV-AbDab or HIV-CATNAP',
                'priority': 1
            })

    # Gap 4: Epitope diversity
    if column_mapping.get('epitope'):
        epitope_col = column_mapping['epitope']
        unique_epitopes = df[epitope_col].nunique()

        if unique_epitopes < 100:
            gaps.append({
                'severity': 'MEDIUM',
                'gap': f'Limited epitope diversity ({unique_epitopes} unique)',
                'recommendation': 'Add more diverse epitopes from IEDB',
                'priority': 2
            })

    # Print gaps
    if gaps:
        print(f"\n🔍 Found {len(gaps)} gaps:")

        # Sort by priority
        gaps.sort(key=lambda x: x['priority'])

        for gap in gaps:
            severity_emoji = {
                'CRITICAL': '🔴',
                'HIGH': '🟠',
                'MEDIUM': '🟡',
                'LOW': '🟢'
            }

            print(f"\n  {severity_emoji[gap['severity']]} {gap['severity']} PRIORITY {gap['priority']}")
            print(f"     Gap: {gap['gap']}")
            print(f"     Recommendation: {gap['recommendation']}")
    else:
        print(f"\n✅ No major gaps identified!")
        print(f"   Your training data appears well-balanced.")

    # Save analysis results
    output_dir = Path('analysis_results')
    output_dir.mkdir(exist_ok=True)

    analysis_results = {
        'dataset_path': str(data_path),
        'total_pairs': len(df),
        'column_mapping': column_mapping,
        'statistics': {
            'epitopes': {
                'unique': int(df_clean[epitope_col].nunique()) if column_mapping.get('epitope') else 0,
                'mean_length': float(epitope_lengths.mean()) if column_mapping.get('epitope') else 0
            }
        },
        'gaps': gaps
    }

    output_file = output_dir / 'training_data_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Analysis saved to: {output_file}")
    print(f"{'='*80}\n")

    return df


def main():
    parser = argparse.ArgumentParser(description='Analyze training data coverage')
    parser.add_argument('--data', type=str, default='data/training_data.csv',
                       help='Path to training data CSV file')

    args = parser.parse_args()

    df = analyze_dataset(args.data)

    if df is not None:
        print("\n✅ Analysis complete!")
        print("\nNext steps:")
        print("  1. Review gaps and recommendations above")
        print("  2. If gaps exist, run: python download_augmentation_data.py")
        print("  3. After augmentation, retrain with: python train.py")
    else:
        print("\n⚠️  Could not complete analysis")
        print("   Please check your data file and try again")


if __name__ == '__main__':
    main()
