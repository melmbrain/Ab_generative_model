"""
Data Loader for Generative Model Training

Loads and batches antibody-antigen pairs for sequence-to-sequence training.
Note: This implementation uses Python standard library only (no PyTorch DataLoader).
"""

import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generators.tokenizer import AminoAcidTokenizer


class AbAgDataset:
    """
    Dataset for antibody-antigen pairs

    Each sample contains:
        - antigen_sequence: Target antigen
        - antibody_heavy: Heavy chain sequence
        - antibody_light: Light chain sequence (optional)
        - pKd: Binding affinity
    """

    def __init__(self, data_path: str, tokenizer: AminoAcidTokenizer):
        """
        Initialize dataset

        Args:
            data_path: Path to JSON data file (train/val/test)
            tokenizer: AminoAcidTokenizer instance
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer

        # Load data
        print(f"Loading dataset from: {self.data_path}")
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get a single sample"""
        return self.data[idx]

    def get_batch(self, indices: List[int],
                  max_antigen_len: int = 512,
                  max_antibody_len: int = 300) -> Dict:
        """
        Get a batch of samples

        Args:
            indices: List of sample indices
            max_antigen_len: Maximum antigen sequence length
            max_antibody_len: Maximum antibody sequence length

        Returns:
            Dictionary with batched and tokenized data
        """
        batch_data = [self.data[i] for i in indices]

        # Extract sequences
        antigen_seqs = [d['antigen_sequence'] for d in batch_data]
        antibody_heavies = [d['antibody_heavy'] for d in batch_data]
        antibody_lights = [d.get('antibody_light', '') for d in batch_data]
        pKds = [d['pKd'] for d in batch_data]

        # Tokenize antigens
        antigen_tokens, antigen_masks = self.tokenizer.batch_encode(
            antigen_seqs,
            max_length=max_antigen_len,
            padding=True,
            add_special_tokens=True
        )

        # Tokenize antibodies (heavy + light)
        antibody_tokens = []
        antibody_masks = []

        for heavy, light in zip(antibody_heavies, antibody_lights):
            tokens = self.tokenizer.encode_pair(heavy, light, add_special_tokens=True)

            # Truncate if needed
            if len(tokens) > max_antibody_len:
                tokens = tokens[:max_antibody_len]

            # Create mask
            mask = [1] * len(tokens)

            # Pad if needed
            if len(tokens) < max_antibody_len:
                padding_length = max_antibody_len - len(tokens)
                tokens = tokens + [self.tokenizer.pad_token_id] * padding_length
                mask = mask + [0] * padding_length

            antibody_tokens.append(tokens)
            antibody_masks.append(mask)

        return {
            'antigen_tokens': antigen_tokens,
            'antigen_mask': antigen_masks,
            'antibody_tokens': antibody_tokens,
            'antibody_mask': antibody_masks,
            'pKd': pKds,
            'batch_size': len(indices)
        }


class DataLoader:
    """
    Simple data loader for batching

    Note: This is a simplified version without PyTorch's DataLoader.
    For production, use PyTorch DataLoader with custom Dataset class.
    """

    def __init__(self,
                 dataset: AbAgDataset,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 drop_last: bool = False):
        """
        Initialize data loader

        Args:
            dataset: AbAgDataset instance
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data
            drop_last: Whether to drop last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.num_samples = len(dataset)
        self.num_batches = self.num_samples // batch_size
        if not drop_last and self.num_samples % batch_size != 0:
            self.num_batches += 1

    def __len__(self):
        """Number of batches"""
        return self.num_batches

    def __iter__(self):
        """Iterate over batches"""
        # Create indices
        indices = list(range(self.num_samples))

        # Shuffle if needed
        if self.shuffle:
            random.shuffle(indices)

        # Yield batches
        for i in range(self.num_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.num_samples)

            batch_indices = indices[start_idx:end_idx]

            # Skip if last batch is incomplete and drop_last=True
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue

            # Get batch data
            batch = self.dataset.get_batch(batch_indices)
            yield batch


def test_data_loader():
    """Test data loader functionality"""
    print("="*70)
    print("Testing Data Loader")
    print("="*70)

    # Initialize tokenizer
    tokenizer = AminoAcidTokenizer()

    # Test on validation set (smaller)
    data_path = "data/generative/val.json"

    if not Path(data_path).exists():
        print(f"\n❌ Data file not found: {data_path}")
        print("   Run prepare_data_simple.py first")
        return

    # Load dataset
    print("\n1. Loading dataset...")
    dataset = AbAgDataset(data_path, tokenizer)
    print(f"   Dataset size: {len(dataset)}")
    print("   ✅ Loaded")

    # Test single sample
    print("\n2. Testing single sample access...")
    sample = dataset[0]
    print(f"   Keys: {list(sample.keys())}")
    print(f"   Antigen length: {len(sample['antigen_sequence'])}")
    print(f"   Heavy length: {len(sample['antibody_heavy'])}")
    print(f"   pKd: {sample['pKd']}")
    print("   ✅ Passed")

    # Test batch creation
    print("\n3. Testing batch creation...")
    batch = dataset.get_batch([0, 1, 2], max_antigen_len=100, max_antibody_len=50)
    print(f"   Batch size: {batch['batch_size']}")
    print(f"   Antigen tokens shape: {len(batch['antigen_tokens'])} x {len(batch['antigen_tokens'][0])}")
    print(f"   Antibody tokens shape: {len(batch['antibody_tokens'])} x {len(batch['antibody_tokens'][0])}")
    print(f"   pKds: {batch['pKd']}")
    print("   ✅ Passed")

    # Test data loader iteration
    print("\n4. Testing data loader iteration...")
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    print(f"   Number of batches: {len(loader)}")

    # Iterate through first few batches
    for i, batch in enumerate(loader):
        if i >= 3:  # Only test first 3 batches
            break
        print(f"   Batch {i+1}: {batch['batch_size']} samples")

    print("   ✅ Passed")

    # Summary
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
    print(f"\nDataset info:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Batch size: 16")
    print(f"  Number of batches: {len(loader)}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print("\nReady for model training!")
    print("="*70)


if __name__ == '__main__':
    test_data_loader()
