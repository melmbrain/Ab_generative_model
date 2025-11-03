"""
Amino Acid Tokenizer for Sequence-to-Sequence Model

Handles tokenization of protein sequences for antibody generation.
"""

import json
from typing import List, Dict, Tuple


class AminoAcidTokenizer:
    """
    Tokenizer for protein sequences

    Vocabulary:
        - Special tokens: <PAD>, <START>, <END>, <UNK>
        - 20 standard amino acids: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
    """

    def __init__(self):
        # Define vocabulary
        self.special_tokens = {
            '<PAD>': 0,    # Padding token
            '<START>': 1,  # Start of sequence
            '<END>': 2,    # End of sequence
            '<UNK>': 3,    # Unknown token
            '<SEP>': 4     # Separator (for heavy|light)
        }

        # 20 standard amino acids
        self.amino_acids = [
            'A', 'C', 'D', 'E', 'F',
            'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R',
            'S', 'T', 'V', 'W', 'Y'
        ]

        # Build vocabulary
        self.token_to_id = self.special_tokens.copy()
        for i, aa in enumerate(self.amino_acids):
            self.token_to_id[aa] = i + len(self.special_tokens)

        # Reverse mapping
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        self.vocab_size = len(self.token_to_id)
        self.pad_token_id = self.token_to_id['<PAD>']
        self.start_token_id = self.token_to_id['<START>']
        self.end_token_id = self.token_to_id['<END>']
        self.unk_token_id = self.token_to_id['<UNK>']
        self.sep_token_id = self.token_to_id['<SEP>']

    def encode(self, sequence: str, add_special_tokens: bool = True) -> List[int]:
        """
        Convert amino acid sequence to token IDs

        Args:
            sequence: Protein sequence string
            add_special_tokens: Whether to add <START> and <END> tokens

        Returns:
            List of token IDs
        """
        # Clean sequence
        sequence = sequence.upper().replace(' ', '').replace('\n', '')

        # Convert to tokens
        tokens = []

        if add_special_tokens:
            tokens.append(self.start_token_id)

        for aa in sequence:
            if aa in self.token_to_id:
                tokens.append(self.token_to_id[aa])
            else:
                tokens.append(self.unk_token_id)

        if add_special_tokens:
            tokens.append(self.end_token_id)

        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Convert token IDs back to amino acid sequence

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Protein sequence string
        """
        sequence = []

        special_ids = {self.pad_token_id, self.start_token_id,
                      self.end_token_id, self.unk_token_id}

        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                continue

            if token_id == self.sep_token_id:
                sequence.append('|')  # Keep separator
            elif token_id in self.id_to_token:
                sequence.append(self.id_to_token[token_id])

        return ''.join(sequence)

    def encode_pair(self, heavy: str, light: str = '', add_special_tokens: bool = True) -> List[int]:
        """
        Encode antibody heavy and light chains

        Args:
            heavy: Heavy chain sequence
            light: Light chain sequence (optional)
            add_special_tokens: Whether to add special tokens

        Returns:
            List of token IDs
        """
        tokens = []

        if add_special_tokens:
            tokens.append(self.start_token_id)

        # Encode heavy chain
        for aa in heavy.upper().replace(' ', ''):
            tokens.append(self.token_to_id.get(aa, self.unk_token_id))

        # Add separator if light chain exists
        if light and len(light) > 0:
            tokens.append(self.sep_token_id)

            # Encode light chain
            for aa in light.upper().replace(' ', ''):
                tokens.append(self.token_to_id.get(aa, self.unk_token_id))

        if add_special_tokens:
            tokens.append(self.end_token_id)

        return tokens

    def batch_encode(self, sequences: List[str],
                    max_length: int = None,
                    padding: bool = True,
                    add_special_tokens: bool = True) -> Tuple[List[List[int]], List[int]]:
        """
        Encode batch of sequences

        Args:
            sequences: List of protein sequences
            max_length: Maximum sequence length (None = use longest in batch)
            padding: Whether to pad sequences to max_length
            add_special_tokens: Whether to add special tokens

        Returns:
            Tuple of (token_ids, attention_mask)
        """
        # Encode all sequences
        batch_tokens = [self.encode(seq, add_special_tokens) for seq in sequences]

        # Determine max length
        if max_length is None:
            max_length = max(len(tokens) for tokens in batch_tokens)

        # Pad sequences
        padded_tokens = []
        attention_masks = []

        for tokens in batch_tokens:
            # Truncate if too long
            if len(tokens) > max_length:
                tokens = tokens[:max_length]

            # Create attention mask (1 = real token, 0 = padding)
            mask = [1] * len(tokens)

            # Pad if needed
            if padding and len(tokens) < max_length:
                padding_length = max_length - len(tokens)
                tokens = tokens + [self.pad_token_id] * padding_length
                mask = mask + [0] * padding_length

            padded_tokens.append(tokens)
            attention_masks.append(mask)

        return padded_tokens, attention_masks

    def save_vocab(self, filepath: str):
        """Save vocabulary to JSON file"""
        vocab_data = {
            'token_to_id': self.token_to_id,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'amino_acids': self.amino_acids
        }

        with open(filepath, 'w') as f:
            json.dump(vocab_data, f, indent=2)

    @classmethod
    def from_vocab_file(cls, filepath: str):
        """Load tokenizer from vocabulary file"""
        tokenizer = cls()
        # For now, just use default vocab
        # In future, could load custom vocab
        return tokenizer

    def __len__(self):
        return self.vocab_size

    def __repr__(self):
        return f"AminoAcidTokenizer(vocab_size={self.vocab_size})"


def test_tokenizer():
    """Test tokenizer functionality"""
    print("="*70)
    print("Testing AminoAcidTokenizer")
    print("="*70)

    tokenizer = AminoAcidTokenizer()

    # Test 1: Basic encoding/decoding
    print("\n1. Basic encoding/decoding:")
    seq = "EVQLQQSGAE"
    tokens = tokenizer.encode(seq)
    decoded = tokenizer.decode(tokens)
    print(f"   Original:  {seq}")
    print(f"   Tokens:    {tokens}")
    print(f"   Decoded:   {decoded}")
    assert decoded == seq, "Decoding failed!"
    print("   ✅ Passed")

    # Test 2: Heavy + Light chain
    print("\n2. Antibody pair encoding:")
    heavy = "QVQLVQSGAE"
    light = "DIQMTQSPS"
    tokens = tokenizer.encode_pair(heavy, light)
    decoded = tokenizer.decode(tokens)
    print(f"   Heavy:     {heavy}")
    print(f"   Light:     {light}")
    print(f"   Tokens:    {tokens}")
    print(f"   Decoded:   {decoded}")
    expected = heavy + "|" + light
    assert decoded == expected, f"Expected {expected}, got {decoded}"
    print("   ✅ Passed")

    # Test 3: Batch encoding with padding
    print("\n3. Batch encoding:")
    sequences = ["ACDE", "FGHIKLM", "NP"]
    padded, masks = tokenizer.batch_encode(sequences, max_length=10)
    print(f"   Sequences: {sequences}")
    print(f"   Padded:    {padded}")
    print(f"   Masks:     {masks}")
    assert len(padded) == 3, "Wrong batch size"
    assert all(len(p) == 10 for p in padded), "Padding failed"
    print("   ✅ Passed")

    # Test 4: Vocabulary info
    print("\n4. Vocabulary info:")
    print(f"   Vocab size:    {tokenizer.vocab_size}")
    print(f"   PAD token:     {tokenizer.pad_token_id}")
    print(f"   START token:   {tokenizer.start_token_id}")
    print(f"   END token:     {tokenizer.end_token_id}")
    print(f"   UNK token:     {tokenizer.unk_token_id}")
    print("   ✅ Complete")

    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)


if __name__ == '__main__':
    test_tokenizer()
