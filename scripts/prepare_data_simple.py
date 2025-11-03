"""
Simple Data Preparation for Generative Model Training
(No external dependencies - uses only Python standard library)

Loads 159k Ab-Ag pairs and prepares them for model training.

Usage:
    python3 scripts/prepare_data_simple.py
"""

import csv
import json
import random
from pathlib import Path
from collections import Counter

# Paths
SOURCE_DATA = "/mnt/c/Users/401-24/Desktop/Docking prediction/data/raw/agab/agab_phase2_full.csv"
OUTPUT_DIR = Path("data/generative")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Amino acid vocabulary
VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')


def validate_sequence(seq):
    """Validate protein sequence"""
    if not seq or len(seq) == 0:
        return False

    # Remove spaces and convert to upper
    seq = str(seq).replace(' ', '').replace('\n', '').upper()

    # Check for invalid characters
    if not set(seq).issubset(VALID_AA):
        return False

    # Check length
    if len(seq) < 10 or len(seq) > 2000:
        return False

    return True


def parse_float(value):
    """Safely parse float value"""
    try:
        f = float(value)
        if f > 0 and f < 20:  # Valid pKd range
            return f
        return None
    except:
        return None


def load_and_clean_data():
    """Load and clean the dataset"""
    print("="*70)
    print("Simple Data Preparation for Generative Model")
    print("="*70)

    print(f"\n1. Loading data from: {SOURCE_DATA}")

    data = []
    skipped = 0

    with open(SOURCE_DATA, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            # Parse antibody sequence (format: heavy|light)
            ab_seq = row.get('antibody_sequence', '')

            if '|' in ab_seq:
                parts = ab_seq.split('|')
                heavy = parts[0]
                light = parts[1] if len(parts) > 1 else ''
            else:
                heavy = ab_seq
                light = ''

            antigen = row.get('antigen_sequence', '')
            pKd_str = row.get('pKd', '')

            # Validate
            pKd = parse_float(pKd_str)

            if not validate_sequence(heavy):
                skipped += 1
                continue

            if not validate_sequence(antigen):
                skipped += 1
                continue

            if pKd is None:
                skipped += 1
                continue

            # Store valid sample
            data.append({
                'antibody_heavy': heavy,
                'antibody_light': light,
                'antigen_sequence': antigen,
                'pKd': pKd,
                'affinity_type': row.get('affinity_type', ''),
                'dataset': row.get('dataset', ''),
                'confidence': row.get('confidence', '')
            })

            if (i + 1) % 10000 == 0:
                print(f"   Processed {i+1} rows, kept {len(data)} valid samples...")

    print(f"\n   Total rows processed: {i+1}")
    print(f"   Valid samples: {len(data)}")
    print(f"   Skipped: {skipped} ({skipped/(i+1)*100:.1f}%)")

    return data


def analyze_statistics(data):
    """Compute dataset statistics"""
    print("\n2. Analyzing dataset statistics...")

    # Sequence lengths
    heavy_lens = [len(d['antibody_heavy']) for d in data]
    light_lens = [len(d['antibody_light']) for d in data if d['antibody_light']]
    antigen_lens = [len(d['antigen_sequence']) for d in data]

    # Affinity values
    pKds = [d['pKd'] for d in data]

    stats = {
        'total_samples': len(data),

        'heavy_length': {
            'mean': sum(heavy_lens) / len(heavy_lens),
            'min': min(heavy_lens),
            'max': max(heavy_lens)
        },

        'light_length': {
            'mean': sum(light_lens) / len(light_lens) if light_lens else 0,
            'min': min(light_lens) if light_lens else 0,
            'max': max(light_lens) if light_lens else 0
        },

        'antigen_length': {
            'mean': sum(antigen_lens) / len(antigen_lens),
            'min': min(antigen_lens),
            'max': max(antigen_lens)
        },

        'pKd': {
            'mean': sum(pKds) / len(pKds),
            'min': min(pKds),
            'max': max(pKds)
        }
    }

    print(f"\n   Sequence Lengths:")
    print(f"     Heavy chain: {stats['heavy_length']['mean']:.1f} AA")
    print(f"                  Range: {stats['heavy_length']['min']}-{stats['heavy_length']['max']}")
    print(f"     Light chain: {stats['light_length']['mean']:.1f} AA")
    print(f"                  Range: {stats['light_length']['min']}-{stats['light_length']['max']}")
    print(f"     Antigen:     {stats['antigen_length']['mean']:.1f} AA")
    print(f"                  Range: {stats['antigen_length']['min']}-{stats['antigen_length']['max']}")

    print(f"\n   Binding Affinity (pKd):")
    print(f"     Mean: {stats['pKd']['mean']:.2f}")
    print(f"     Range: {stats['pKd']['min']:.2f} - {stats['pKd']['max']:.2f}")

    return stats


def create_splits(data):
    """Create train/val/test splits"""
    print("\n3. Creating train/val/test splits...")

    # Shuffle data
    random.seed(42)
    random.shuffle(data)

    # Split: 80% train, 10% val, 10% test
    n_total = len(data)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)

    train = data[:n_train]
    val = data[n_train:n_train + n_val]
    test = data[n_train + n_val:]

    print(f"   Train: {len(train)} samples ({len(train)/n_total*100:.1f}%)")
    print(f"   Val:   {len(val)} samples ({len(val)/n_total*100:.1f}%)")
    print(f"   Test:  {len(test)} samples ({len(test)/n_total*100:.1f}%)")

    return train, val, test


def save_split(data, filename):
    """Save data split to JSON file"""
    filepath = OUTPUT_DIR / filename

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"   ✅ Saved: {filepath} ({len(data)} samples)")


def main():
    """Main pipeline"""

    # Load and clean
    data = load_and_clean_data()

    if len(data) == 0:
        print("\n❌ No valid data found!")
        return

    # Analyze
    stats = analyze_statistics(data)

    # Create splits
    train, val, test = create_splits(data)

    # Save splits
    print("\n4. Saving data splits...")
    save_split(train, 'train.json')
    save_split(val, 'val.json')
    save_split(test, 'test.json')

    # Save statistics
    with open(OUTPUT_DIR / 'data_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"   ✅ Saved: {OUTPUT_DIR / 'data_stats.json'}")

    # Summary
    print("\n" + "="*70)
    print("✅ Data Preparation Complete!")
    print("="*70)
    print(f"\nDataset Summary:")
    print(f"  Total samples: {len(data)}")
    print(f"  Train:         {len(train)} ({len(train)/len(data)*100:.1f}%)")
    print(f"  Validation:    {len(val)} ({len(val)/len(data)*100:.1f}%)")
    print(f"  Test:          {len(test)} ({len(test)/len(data)*100:.1f}%)")
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    print(f"\nFiles created:")
    print(f"  - train.json ({len(train)} samples)")
    print(f"  - val.json ({len(val)} samples)")
    print(f"  - test.json ({len(test)} samples)")
    print(f"  - data_stats.json (statistics)")
    print(f"\nNext steps:")
    print(f"  1. Review data_stats.json for detailed statistics")
    print(f"  2. Proceed to model training")
    print("="*70)


if __name__ == '__main__':
    main()
