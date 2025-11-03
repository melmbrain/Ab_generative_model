"""
LSTM Sequence-to-Sequence Model for Antibody Generation

Faster and simpler alternative to Transformer.
Good baseline for antigen-conditioned antibody generation.

Architecture:
    Encoder: Bidirectional LSTM over antigen sequence
    Decoder: LSTM generates antibody sequence autoregressively
    Conditioning: pKd value concatenated to encoder output
"""

import math
import random
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


class LSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder for antigen sequences
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

        # Project bidirectional hidden states to single direction
        self.hidden_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.cell_projection = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src, src_lengths=None):
        """
        Args:
            src: [batch_size, seq_len] - antigen token ids
            src_lengths: [batch_size] - actual lengths (for packing)

        Returns:
            outputs: [batch_size, seq_len, hidden_dim * 2]
            hidden: tuple of (h_n, c_n) each [num_layers, batch_size, hidden_dim]
        """
        # Embed
        embedded = self.dropout(self.embedding(src))  # [batch, seq_len, emb_dim]

        # Pack if lengths provided (optional optimization)
        if src_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        # Encode
        outputs, (hidden, cell) = self.lstm(embedded)

        # Unpack if we packed
        if src_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        # hidden: [num_layers * 2, batch, hidden_dim] (bidirectional)
        # Combine forward and backward
        num_layers = hidden.size(0) // 2

        # Reshape: [num_layers, 2, batch, hidden_dim]
        hidden = hidden.view(num_layers, 2, -1, hidden.size(2))
        cell = cell.view(num_layers, 2, -1, cell.size(2))

        # Concatenate forward and backward: [num_layers, batch, hidden_dim * 2]
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
        cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)

        # Project to decoder hidden size: [num_layers, batch, hidden_dim]
        hidden = self.hidden_projection(hidden)
        cell = self.cell_projection(cell)

        return outputs, (hidden, cell)


class LSTMDecoder(nn.Module):
    """
    LSTM decoder with attention for antibody generation
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM input: embedding + context (from attention)
        self.lstm = nn.LSTM(
            embedding_dim + hidden_dim * 2,  # embedding + encoder output
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Attention mechanism
        self.attention = nn.Linear(hidden_dim + hidden_dim * 2, hidden_dim)
        self.attention_combine = nn.Linear(hidden_dim * 2, hidden_dim * 2)

        # Output projection
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        """
        Single step decoding

        Args:
            input: [batch_size, 1] - current token
            hidden: tuple of (h, c)
            encoder_outputs: [batch_size, src_len, hidden_dim * 2]

        Returns:
            output: [batch_size, vocab_size]
            hidden: updated hidden state
            attention_weights: [batch_size, src_len]
        """
        # Embed input
        embedded = self.dropout(self.embedding(input))  # [batch, 1, emb_dim]

        # Compute attention
        # hidden[0]: [num_layers, batch, hidden_dim]
        # Use last layer for attention
        h_last = hidden[0][-1].unsqueeze(1)  # [batch, 1, hidden_dim]

        # Repeat for each encoder output
        h_repeated = h_last.repeat(1, encoder_outputs.size(1), 1)  # [batch, src_len, hidden_dim]

        # Concatenate and compute attention scores
        attention_input = torch.cat([h_repeated, encoder_outputs], dim=2)  # [batch, src_len, hidden + hidden*2]
        attention_scores = torch.tanh(self.attention(attention_input))  # [batch, src_len, hidden_dim]
        attention_weights = F.softmax(attention_scores.sum(dim=2), dim=1)  # [batch, src_len]

        # Apply attention to encoder outputs
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # [batch, 1, hidden*2]

        # Combine embedding and context
        lstm_input = torch.cat([embedded, context], dim=2)  # [batch, 1, emb_dim + hidden*2]

        # Decode
        output, hidden = self.lstm(lstm_input, hidden)

        # Project to vocabulary
        output = self.out(output.squeeze(1))  # [batch, vocab_size]

        return output, hidden, attention_weights


class LSTMSeq2Seq(nn.Module):
    """
    Complete LSTM Seq2Seq model with affinity conditioning

    Usage:
        model = LSTMSeq2Seq(vocab_size=25, embedding_dim=128, hidden_dim=256)
        output = model(antigen_tokens, antibody_tokens, pKd_values)
    """
    def __init__(self,
                 vocab_size,
                 embedding_dim=128,
                 hidden_dim=256,
                 num_layers=2,
                 dropout=0.1,
                 max_length=300):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_length = max_length

        # Encoder and decoder
        self.encoder = LSTMEncoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)

        # Affinity conditioning
        self.affinity_projection = nn.Linear(1, hidden_dim)

        # Special tokens
        self.pad_token_id = 0
        self.start_token_id = 1
        self.end_token_id = 2

    def forward(self, src, tgt, pKd, teacher_forcing_ratio=0.5):
        """
        Training forward pass with teacher forcing

        Args:
            src: [batch_size, src_len] - antigen tokens
            tgt: [batch_size, tgt_len] - antibody tokens
            pKd: [batch_size, 1] - binding affinity
            teacher_forcing_ratio: probability of using teacher forcing

        Returns:
            outputs: [batch_size, tgt_len, vocab_size]
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)

        # Encode antigen
        encoder_outputs, hidden = self.encoder(src)

        # Condition on affinity
        affinity_emb = self.affinity_projection(pKd)  # [batch, hidden_dim]

        # Add to hidden state
        h, c = hidden
        h[-1] = h[-1] + affinity_emb  # Add to last layer
        hidden = (h, c)

        # Initialize outputs tensor
        outputs = torch.zeros(batch_size, tgt_len, self.vocab_size).to(src.device)

        # First input is <START> token
        input = tgt[:, 0].unsqueeze(1)  # [batch, 1]

        # Decode step by step
        for t in range(1, tgt_len):
            # Decode one step
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)

            # Store output
            outputs[:, t] = output

            # Teacher forcing: use ground truth as next input
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)

            input = tgt[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)

        return outputs

    def generate(self, src, pKd, max_length=None, temperature=1.0, top_k=None):
        """
        Generate antibody sequence given antigen and target affinity

        Args:
            src: [batch_size, src_len] - antigen tokens
            pKd: [batch_size, 1] - target binding affinity
            max_length: maximum generation length
            temperature: sampling temperature (higher = more random)
            top_k: if set, only sample from top k tokens

        Returns:
            generated: [batch_size, max_len] - generated antibody tokens
        """
        self.eval()
        batch_size = src.size(0)
        max_length = max_length or self.max_length

        with torch.no_grad():
            # Encode antigen
            encoder_outputs, hidden = self.encoder(src)

            # Condition on affinity
            affinity_emb = self.affinity_projection(pKd)
            h, c = hidden
            h[-1] = h[-1] + affinity_emb
            hidden = (h, c)

            # Start with <START> token
            input = torch.full((batch_size, 1), self.start_token_id, dtype=torch.long).to(src.device)

            # Store generated tokens
            generated = [input]

            # Generate until <END> or max_length
            for _ in range(max_length - 1):
                # Decode one step
                output, hidden, _ = self.decoder(input, hidden, encoder_outputs)

                # Apply temperature
                logits = output / temperature

                # Top-k filtering
                if top_k is not None:
                    values, _ = torch.topk(logits, top_k)
                    logits[logits < values[:, -1, None]] = -float('Inf')

                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)  # [batch, 1]

                generated.append(next_token)

                # Check if all sequences generated <END>
                if (next_token == self.end_token_id).all():
                    break

                input = next_token

            # Concatenate all generated tokens
            generated = torch.cat(generated, dim=1)

        return generated

    def get_model_size(self):
        """Calculate number of parameters"""
        return sum(p.numel() for p in self.parameters())


# Model configurations
MODEL_CONFIGS = {
    'tiny': {
        'embedding_dim': 64,
        'hidden_dim': 128,
        'num_layers': 1,
        'dropout': 0.1
    },
    'small': {
        'embedding_dim': 128,
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.1
    },
    'medium': {
        'embedding_dim': 256,
        'hidden_dim': 512,
        'num_layers': 2,
        'dropout': 0.2
    },
    'large': {
        'embedding_dim': 512,
        'hidden_dim': 1024,
        'num_layers': 3,
        'dropout': 0.2
    }
}


def create_model(config_name='tiny', vocab_size=25, max_length=300):
    """
    Create model from configuration

    Args:
        config_name: 'tiny', 'small', 'medium', or 'large'
        vocab_size: vocabulary size
        max_length: maximum sequence length

    Returns:
        model: LSTMSeq2Seq instance
    """
    if config_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Choose from {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[config_name]

    model = LSTMSeq2Seq(
        vocab_size=vocab_size,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        max_length=max_length
    )

    return model


def test_model():
    """Test LSTM Seq2Seq model"""
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available. Cannot test model.")
        return

    print("="*70)
    print("Testing LSTM Seq2Seq Model")
    print("="*70)

    # Test all configurations
    for config_name in ['tiny', 'small', 'medium']:
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

        src = torch.randint(0, 25, (batch_size, src_len))
        tgt = torch.randint(0, 25, (batch_size, tgt_len))
        pKd = torch.randn(batch_size, 1) * 2 + 7  # Mean ~7, std ~2

        # Forward pass
        outputs = model(src, tgt, pKd, teacher_forcing_ratio=0.5)
        print(f"  Input: antigen [{batch_size}, {src_len}], antibody [{batch_size}, {tgt_len}]")
        print(f"  Output: {list(outputs.shape)} = [batch, tgt_len, vocab_size]")

        # Test generation
        generated = model.generate(src, pKd, max_length=30)
        print(f"  Generated: {list(generated.shape)} = [batch, max_len]")
        print(f"  Sample sequence: {generated[0].tolist()[:10]}...")

        print(f"  ✅ {config_name.upper()} model working!")

    print("\n" + "="*70)
    print("✅ All model tests passed!")
    print("="*70)


if __name__ == '__main__':
    test_model()
