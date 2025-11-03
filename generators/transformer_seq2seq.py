"""
Transformer Sequence-to-Sequence Model for Antibody Generation

State-of-the-art architecture for antigen-conditioned antibody generation.
Uses multi-head self-attention for better sequence modeling.

Architecture:
    Encoder: Transformer encoder over antigen sequence
    Decoder: Transformer decoder generates antibody sequence autoregressively
    Conditioning: pKd value added to encoder output
"""

import math
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Please install: pip install torch")


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerSeq2Seq(nn.Module):
    """
    Complete Transformer Seq2Seq model with affinity conditioning

    Usage:
        model = TransformerSeq2Seq(vocab_size=25, d_model=512, nhead=8)
        output = model(antigen_tokens, antibody_tokens, pKd_values)
    """
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 max_src_len: int = 1000,
                 max_tgt_len: int = 500):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_tgt_len = max_tgt_len

        # Embeddings
        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # Positional encodings
        self.src_pos_encoder = PositionalEncoding(d_model, max_src_len, dropout)
        self.tgt_pos_encoder = PositionalEncoding(d_model, max_tgt_len, dropout)

        # Transformer with Pre-LN and GELU (2024 best practices)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',  # GELU instead of ReLU (used in BERT, GPT, ESM2)
            norm_first=True     # Pre-LN instead of Post-LN (GPT-3, modern transformers)
        )

        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Affinity conditioning - project to d_model and add to memory
        self.affinity_projection = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),  # GELU for consistency with transformer
            nn.Linear(d_model, d_model)
        )

        # Special tokens
        self.pad_token_id = 0
        self.start_token_id = 1
        self.end_token_id = 2

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters with Xavier/Glorot"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz: int):
        """
        Generate causal mask for decoder (prevents looking ahead)

        Args:
            sz: sequence length

        Returns:
            [sz, sz] mask with -inf for future positions
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        mask = mask.float().masked_fill(mask, float('-inf'))
        return mask

    def create_padding_mask(self, seq, pad_token_id=0):
        """
        Create padding mask

        Args:
            seq: [batch_size, seq_len]
            pad_token_id: padding token id

        Returns:
            [batch_size, seq_len] - True for padding positions
        """
        return seq == pad_token_id

    def forward(self, src, tgt, pKd):
        """
        Training forward pass

        Args:
            src: [batch_size, src_len] - antigen tokens
            tgt: [batch_size, tgt_len] - antibody tokens (with <START>)
            pKd: [batch_size, 1] - binding affinity

        Returns:
            [batch_size, tgt_len, vocab_size] - logits
        """
        batch_size = src.size(0)
        src_len = src.size(1)
        tgt_len = tgt.size(1)
        device = src.device

        # Create padding masks
        src_key_padding_mask = self.create_padding_mask(src, self.pad_token_id)
        tgt_key_padding_mask = self.create_padding_mask(tgt, self.pad_token_id)

        # Create causal mask for decoder
        tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(device)

        # Embed and add positional encoding
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.src_pos_encoder(src_emb)

        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_pos_encoder(tgt_emb)

        # Condition on affinity - add to source embedding
        affinity_emb = self.affinity_projection(pKd)  # [batch, d_model]
        affinity_emb = affinity_emb.unsqueeze(1)  # [batch, 1, d_model]
        src_emb = src_emb + affinity_emb

        # Run transformer
        output = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        # Project to vocabulary
        output = self.fc_out(output)

        return output

    def generate_greedy(self, src, pKd, max_length=None):
        """
        Generate antibody sequence using greedy decoding

        Args:
            src: [batch_size, src_len] - antigen tokens
            pKd: [batch_size, 1] - target binding affinity
            max_length: maximum generation length

        Returns:
            [batch_size, max_len] - generated antibody tokens
        """
        self.eval()
        batch_size = src.size(0)
        max_length = max_length or self.max_tgt_len
        device = src.device

        with torch.no_grad():
            # Encode source with affinity conditioning
            src_key_padding_mask = self.create_padding_mask(src, self.pad_token_id)
            src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
            src_emb = self.src_pos_encoder(src_emb)

            affinity_emb = self.affinity_projection(pKd).unsqueeze(1)
            src_emb = src_emb + affinity_emb

            # Encode with transformer encoder
            memory = self.transformer.encoder(
                src_emb,
                src_key_padding_mask=src_key_padding_mask
            )

            # Start with <START> token
            ys = torch.full((batch_size, 1), self.start_token_id, dtype=torch.long, device=device)

            # Generate tokens one by one
            for i in range(max_length - 1):
                # Embed target
                tgt_emb = self.tgt_embedding(ys) * math.sqrt(self.d_model)
                tgt_emb = self.tgt_pos_encoder(tgt_emb)

                # Create causal mask
                tgt_mask = self.generate_square_subsequent_mask(ys.size(1)).to(device)

                # Decode
                out = self.transformer.decoder(
                    tgt_emb,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_key_padding_mask
                )

                # Project to vocabulary
                out = self.fc_out(out)

                # Get next token (greedy)
                next_token = out[:, -1, :].argmax(dim=-1, keepdim=True)

                # Append to sequence
                ys = torch.cat([ys, next_token], dim=1)

                # Stop if all sequences generated <END>
                if (next_token == self.end_token_id).all():
                    break

        return ys

    def generate(self, src, pKd, max_length=None, temperature=1.0, top_k=None, top_p=None):
        """
        Generate antibody sequence with sampling

        Args:
            src: [batch_size, src_len] - antigen tokens
            pKd: [batch_size, 1] - target binding affinity
            max_length: maximum generation length
            temperature: sampling temperature (higher = more random)
            top_k: if set, only sample from top k tokens
            top_p: if set, nucleus sampling

        Returns:
            [batch_size, max_len] - generated antibody tokens
        """
        self.eval()
        batch_size = src.size(0)
        max_length = max_length or self.max_tgt_len
        device = src.device

        with torch.no_grad():
            # Encode source with affinity conditioning
            src_key_padding_mask = self.create_padding_mask(src, self.pad_token_id)
            src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
            src_emb = self.src_pos_encoder(src_emb)

            affinity_emb = self.affinity_projection(pKd).unsqueeze(1)
            src_emb = src_emb + affinity_emb

            # Encode with transformer encoder
            memory = self.transformer.encoder(
                src_emb,
                src_key_padding_mask=src_key_padding_mask
            )

            # Start with <START> token
            ys = torch.full((batch_size, 1), self.start_token_id, dtype=torch.long, device=device)

            # Generate tokens one by one
            for i in range(max_length - 1):
                # Embed target
                tgt_emb = self.tgt_embedding(ys) * math.sqrt(self.d_model)
                tgt_emb = self.tgt_pos_encoder(tgt_emb)

                # Create causal mask
                tgt_mask = self.generate_square_subsequent_mask(ys.size(1)).to(device)

                # Decode
                out = self.transformer.decoder(
                    tgt_emb,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_key_padding_mask
                )

                # Project to vocabulary
                out = self.fc_out(out)

                # Get logits for next token
                logits = out[:, -1, :] / temperature

                # Top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                # Top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0

                    for batch_idx in range(batch_size):
                        indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                        logits[batch_idx, indices_to_remove] = -float('Inf')

                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                ys = torch.cat([ys, next_token], dim=1)

                # Stop if all sequences generated <END>
                if (next_token == self.end_token_id).all():
                    break

        return ys

    def get_model_size(self):
        """Calculate number of parameters"""
        return sum(p.numel() for p in self.parameters())


# Model configurations
MODEL_CONFIGS = {
    'tiny': {
        'd_model': 128,
        'nhead': 4,
        'num_encoder_layers': 2,
        'num_decoder_layers': 2,
        'dim_feedforward': 512,
        'dropout': 0.1
    },
    'small': {
        'd_model': 256,
        'nhead': 8,
        'num_encoder_layers': 3,
        'num_decoder_layers': 3,
        'dim_feedforward': 1024,
        'dropout': 0.1
    },
    'medium': {
        'd_model': 512,
        'nhead': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'dim_feedforward': 2048,
        'dropout': 0.1
    },
    'large': {
        'd_model': 768,
        'nhead': 12,
        'num_encoder_layers': 8,
        'num_decoder_layers': 8,
        'dim_feedforward': 3072,
        'dropout': 0.1
    }
}


def create_model(config_name='small', vocab_size=25, max_src_len=1000, max_tgt_len=500):
    """
    Create model from configuration

    Args:
        config_name: 'tiny', 'small', 'medium', or 'large'
        vocab_size: vocabulary size
        max_src_len: maximum source sequence length
        max_tgt_len: maximum target sequence length

    Returns:
        model: TransformerSeq2Seq instance
    """
    if config_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Choose from {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[config_name]

    model = TransformerSeq2Seq(
        vocab_size=vocab_size,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len
    )

    return model


def test_model():
    """Test Transformer Seq2Seq model"""
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available. Cannot test model.")
        return

    print("="*70)
    print("Testing Transformer Seq2Seq Model")
    print("="*70)

    # Test configurations
    for config_name in ['tiny', 'small']:
        print(f"\n{config_name.upper()} Configuration:")
        print("-" * 50)

        # Create model
        model = create_model(config_name, vocab_size=25)

        # Print model info
        num_params = model.get_model_size()
        print(f"  Parameters: {num_params:,} ({num_params/1e6:.2f}M)")

        # Test forward pass
        batch_size = 4
        src_len = 50
        tgt_len = 30

        src = torch.randint(1, 25, (batch_size, src_len))
        tgt = torch.randint(1, 25, (batch_size, tgt_len))
        pKd = torch.randn(batch_size, 1) * 2 + 7

        tgt[:, 0] = 1  # Set first token to <START>

        # Forward pass
        outputs = model(src, tgt, pKd)
        print(f"  Input: antigen [{batch_size}, {src_len}], antibody [{batch_size}, {tgt_len}]")
        print(f"  Output: {list(outputs.shape)} = [batch, tgt_len, vocab_size]")

        # Test greedy generation
        generated = model.generate_greedy(src, pKd, max_length=30)
        print(f"  Generated (greedy): {list(generated.shape)} = [batch, max_len]")
        print(f"  Sample sequence: {generated[0].tolist()[:10]}...")

        # Test sampling generation
        generated_sample = model.generate(src, pKd, max_length=30, temperature=0.8, top_k=10)
        print(f"  Generated (sample): {list(generated_sample.shape)} = [batch, max_len]")
        print(f"  Sample sequence: {generated_sample[0].tolist()[:10]}...")

        print(f"  ✅ {config_name.upper()} model working!")

    print("\n" + "="*70)
    print("✅ All model tests passed!")
    print("="*70)
    print("\nModel configurations available:")
    for name, config in MODEL_CONFIGS.items():
        print(f"  {name:8s}: d_model={config['d_model']}, layers={config['num_encoder_layers']}/{config['num_decoder_layers']}")
    print("="*70)


if __name__ == '__main__':
    test_model()
