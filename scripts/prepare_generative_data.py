"""
Data Preparation for Generative Model Training

Loads 159k Ab-Ag pairs from Docking prediction project and prepares them for
sequence-to-sequence model training.

Usage:
    python scripts/prepare_generative_data.py

Output:
    - data/generative/train.pkl (127k pairs)
    - data/generative/val.pkl (16k pairs)
    - data/generative/test.pkl (16k pairs)
    - data/generative/data_stats.json
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
SOURCE_DATA = "/mnt/c/Users/401-24/Desktop/Docking prediction/data/raw/agab/agab_phase2_full.csv"
OUTPUT_DIR = Path("data/generative")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Amino acid vocabulary
VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')


def validate_sequence(seq, name="sequence"):
    """Validate protein sequence"""
    if pd.isna(seq) or len(seq) == 0:
        return False

    # Remove common separators
    seq = str(seq).replace(' ', '').replace('\n', '').upper()

    # Check for invalid characters
    invalid = set(seq) - VALID_AA
    if invalid:
        return False

    # Check length
    if len(seq) < 10 or len(seq) > 2000:
        return False

    return True


def analyze_dataset(df):
    """Analyze dataset statistics"""
    stats = {
        'total_samples': len(df),
        'valid_samples': len(df),

        # Sequence lengths
        'antibody_heavy_len': {
            'mean': df['antibody_heavy'].str.len().mean(),
            'std': df['antibody_heavy'].str.len().std(),
            'min': df['antibody_heavy'].str.len().min(),
            'max': df['antibody_heavy'].str.len().max()
        },
        'antibody_light_len': {
            'mean': df['antibody_light'].str.len().mean(),
            'std': df['antibody_light'].str.len().std(),
            'min': df['antibody_light'].str.len().min(),
            'max': df['antibody_light'].str.len().max()
        },
        'antigen_len': {
            'mean': df['antigen_sequence'].str.len().mean(),
            'std': df['antigen_sequence'].str.len().std(),
            'min': df['antigen_sequence'].str.len().min(),
            'max': df['antigen_sequence'].str.len().max()
        },

        # Affinity statistics
        'pKd': {
            'mean': df['pKd'].mean(),
            'std': df['pKd'].std(),
            'min': df['pKd'].min(),
            'max': df['pKd'].max(),
            'median': df['pKd'].median()
        },

        # Amino acid frequencies
        'heavy_aa_freq': Counter(''.join(df['antibody_heavy'].tolist())),
        'light_aa_freq': Counter(''.join(df['antibody_light'].dropna().tolist())),
        'antigen_aa_freq': Counter(''.join(df['antigen_sequence'].tolist()))
    }

    return stats


def plot_statistics(df, output_dir):
    """Create visualization of dataset statistics"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Sequence length distributions
    ax = axes[0, 0]
    ax.hist(df['antibody_heavy'].str.len(), bins=50, alpha=0.7, label='Heavy')
    ax.hist(df['antibody_light'].str.len(), bins=50, alpha=0.7, label='Light')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Count')
    ax.set_title('Antibody Sequence Lengths')
    ax.legend()

    # Plot 2: Antigen length distribution
    ax = axes[0, 1]
    ax.hist(df['antigen_sequence'].str.len(), bins=50, alpha=0.7, color='green')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Count')
    ax.set_title('Antigen Sequence Lengths')

    # Plot 3: pKd distribution
    ax = axes[0, 2]
    ax.hist(df['pKd'], bins=50, alpha=0.7, color='orange')
    ax.axvline(df['pKd'].mean(), color='red', linestyle='--', label=f'Mean: {df["pKd"].mean():.2f}')
    ax.set_xlabel('pKd')
    ax.set_ylabel('Count')
    ax.set_title('Binding Affinity Distribution')
    ax.legend()

    # Plot 4: Heavy chain AA frequency
    ax = axes[1, 0]
    aa_freq = Counter(''.join(df['antibody_heavy'].tolist()))
    aa_sorted = sorted(aa_freq.items(), key=lambda x: x[1], reverse=True)
    aa_names, aa_counts = zip(*aa_sorted)
    ax.bar(aa_names, aa_counts, alpha=0.7, color='blue')
    ax.set_xlabel('Amino Acid')
    ax.set_ylabel('Frequency')
    ax.set_title('Heavy Chain AA Frequency')

    # Plot 5: Light chain AA frequency
    ax = axes[1, 1]
    aa_freq = Counter(''.join(df['antibody_light'].dropna().tolist()))
    aa_sorted = sorted(aa_freq.items(), key=lambda x: x[1], reverse=True)
    aa_names, aa_counts = zip(*aa_sorted)
    ax.bar(aa_names, aa_counts, alpha=0.7, color='purple')
    ax.set_xlabel('Amino Acid')
    ax.set_ylabel('Frequency')
    ax.set_title('Light Chain AA Frequency')

    # Plot 6: Antigen AA frequency
    ax = axes[1, 2]
    aa_freq = Counter(''.join(df['antigen_sequence'].tolist()))
    aa_sorted = sorted(aa_freq.items(), key=lambda x: x[1], reverse=True)
    aa_names, aa_counts = zip(*aa_sorted)
    ax.bar(aa_names, aa_counts, alpha=0.7, color='green')
    ax.set_xlabel('Amino Acid')
    ax.set_ylabel('Frequency')
    ax.set_title('Antigen AA Frequency')

    plt.tight_layout()
    plt.savefig(output_dir / 'data_statistics.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization: {output_dir / 'data_statistics.png'}")


def prepare_data():
    """Main data preparation pipeline"""
    print("="*70)
    print("Data Preparation for Generative Model")
    print("="*70)

    # Load data
    print(f"\n1. Loading data from: {SOURCE_DATA}")
    df = pd.read_csv(SOURCE_DATA)
    print(f"   Loaded {len(df)} rows")
    print(f"   Columns: {list(df.columns)}")

    # Split antibody sequences (format: heavy|light)
    print("\n2. Parsing antibody sequences...")
    if '|' in df['antibody_sequence'].iloc[0]:
        df[['antibody_heavy', 'antibody_light']] = df['antibody_sequence'].str.split('|', expand=True)
    else:
        # If no separator, assume it's just heavy chain
        df['antibody_heavy'] = df['antibody_sequence']
        df['antibody_light'] = ''

    print(f"   Heavy chains: {df['antibody_heavy'].notna().sum()}")
    print(f"   Light chains: {df['antibody_light'].notna().sum()}")

    # Clean data
    print("\n3. Cleaning data...")
    initial_count = len(df)

    # Validate sequences
    df['heavy_valid'] = df['antibody_heavy'].apply(lambda x: validate_sequence(x, 'heavy'))
    df['antigen_valid'] = df['antigen_sequence'].apply(lambda x: validate_sequence(x, 'antigen'))

    # Keep only valid sequences
    df = df[df['heavy_valid'] & df['antigen_valid']].copy()

    # Remove invalid pKd
    df = df[df['pKd'].notna() & (df['pKd'] > 0) & (df['pKd'] < 20)].copy()

    # Remove duplicates
    df = df.drop_duplicates(subset=['antibody_heavy', 'antigen_sequence'])

    final_count = len(df)
    removed = initial_count - final_count
    print(f"   Initial samples: {initial_count}")
    print(f"   Removed: {removed} ({removed/initial_count*100:.1f}%)")
    print(f"   Final samples: {final_count}")

    # Analyze dataset
    print("\n4. Analyzing dataset statistics...")
    stats = analyze_dataset(df)

    print(f"\n   Sequence Lengths:")
    print(f"     Heavy chain: {stats['antibody_heavy_len']['mean']:.1f} ± {stats['antibody_heavy_len']['std']:.1f} AA")
    print(f"                  Range: {stats['antibody_heavy_len']['min']:.0f}-{stats['antibody_heavy_len']['max']:.0f}")
    print(f"     Light chain: {stats['antibody_light_len']['mean']:.1f} ± {stats['antibody_light_len']['std']:.1f} AA")
    print(f"                  Range: {stats['antibody_light_len']['min']:.0f}-{stats['antibody_light_len']['max']:.0f}")
    print(f"     Antigen:     {stats['antigen_len']['mean']:.1f} ± {stats['antigen_len']['std']:.1f} AA")
    print(f"                  Range: {stats['antigen_len']['min']:.0f}-{stats['antigen_len']['max']:.0f}")

    print(f"\n   Binding Affinity (pKd):")
    print(f"     Mean: {stats['pKd']['mean']:.2f} ± {stats['pKd']['std']:.2f}")
    print(f"     Range: {stats['pKd']['min']:.2f} - {stats['pKd']['max']:.2f}")
    print(f"     Median: {stats['pKd']['median']:.2f}")

    # Create train/val/test splits
    print("\n5. Creating train/val/test splits...")

    # First split: 80% train, 20% temp
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=pd.cut(df['pKd'], bins=5, labels=False)  # Stratify by pKd range
    )

    # Second split: split temp into 50% val, 50% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=pd.cut(temp_df['pKd'], bins=5, labels=False)
    )

    print(f"   Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Val:   {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   Test:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")

    # Verify pKd distribution is similar
    print(f"\n   pKd distributions:")
    print(f"     Train: {train_df['pKd'].mean():.2f} ± {train_df['pKd'].std():.2f}")
    print(f"     Val:   {val_df['pKd'].mean():.2f} ± {val_df['pKd'].std():.2f}")
    print(f"     Test:  {test_df['pKd'].mean():.2f} ± {test_df['pKd'].std():.2f}")

    # Save splits
    print("\n6. Saving data splits...")

    # Select columns to keep
    columns_to_keep = [
        'antibody_heavy',
        'antibody_light',
        'antigen_sequence',
        'pKd',
        'affinity_type',
        'dataset',
        'confidence'
    ]

    train_df[columns_to_keep].to_pickle(OUTPUT_DIR / 'train.pkl')
    val_df[columns_to_keep].to_pickle(OUTPUT_DIR / 'val.pkl')
    test_df[columns_to_keep].to_pickle(OUTPUT_DIR / 'test.pkl')

    print(f"   ✅ Saved: {OUTPUT_DIR / 'train.pkl'}")
    print(f"   ✅ Saved: {OUTPUT_DIR / 'val.pkl'}")
    print(f"   ✅ Saved: {OUTPUT_DIR / 'test.pkl'}")

    # Save statistics (convert Counter to dict for JSON serialization)
    stats['heavy_aa_freq'] = dict(stats['heavy_aa_freq'])
    stats['light_aa_freq'] = dict(stats['light_aa_freq'])
    stats['antigen_aa_freq'] = dict(stats['antigen_aa_freq'])

    with open(OUTPUT_DIR / 'data_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"   ✅ Saved: {OUTPUT_DIR / 'data_stats.json'}")

    # Create visualizations
    print("\n7. Creating visualizations...")
    try:
        plot_statistics(df, OUTPUT_DIR)
    except Exception as e:
        print(f"   ⚠️  Could not create plots: {e}")
        print(f"   (matplotlib/seaborn may not be available)")

    # Summary
    print("\n" + "="*70)
    print("✅ Data Preparation Complete!")
    print("="*70)
    print(f"\nDataset Summary:")
    print(f"  Total samples: {len(df)}")
    print(f"  Train:         {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation:    {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:          {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    print(f"\nNext steps:")
    print(f"  1. Review data_stats.json for detailed statistics")
    print(f"  2. Check data_statistics.png for visualizations")
    print(f"  3. Proceed to model training")
    print("="*70)


if __name__ == '__main__':
    prepare_data()
