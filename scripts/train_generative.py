"""
Progressive Training Script for Generative Model

Supports three training stages:
    - Stage 1 (tiny): 1k samples, tiny model, 10 min
    - Stage 2 (small): 10k samples, small model, 1-2 hours
    - Stage 3 (full): 158k samples, full model, 10-20 hours

Usage:
    python scripts/train_generative.py --stage tiny
    python scripts/train_generative.py --stage small
    python scripts/train_generative.py --stage full
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader as TorchDataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("❌ PyTorch not available. Install with: pip install torch")
    sys.exit(1)

from generators.tokenizer import AminoAcidTokenizer
from generators.lstm_seq2seq import create_model, MODEL_CONFIGS


# Training configurations
TRAINING_CONFIGS = {
    'tiny': {
        'n_samples': 1000,
        'epochs': 10,
        'batch_size': 16,
        'learning_rate': 0.001,
        'model_config': 'tiny',
        'max_antigen_len': 200,
        'max_antibody_len': 150
    },
    'small': {
        'n_samples': 10000,
        'epochs': 20,
        'batch_size': 32,
        'learning_rate': 0.0005,
        'model_config': 'small',
        'max_antigen_len': 300,
        'max_antibody_len': 200
    },
    'full': {
        'n_samples': None,  # Use all data
        'epochs': 50,
        'batch_size': 64,
        'learning_rate': 0.0003,
        'model_config': 'medium',
        'max_antigen_len': 512,
        'max_antibody_len': 300
    }
}


class AbAgDataset(Dataset):
    """PyTorch Dataset for Ab-Ag pairs"""

    def __init__(self, data_path, tokenizer, max_samples=None,
                 max_antigen_len=512, max_antibody_len=300, filter_pKd=None):
        self.tokenizer = tokenizer
        self.max_antigen_len = max_antigen_len
        self.max_antibody_len = max_antibody_len

        # Load data
        print(f"Loading data from: {data_path}")
        with open(data_path, 'r') as f:
            data = json.load(f)

        # Filter by pKd if specified (use high-quality samples)
        if filter_pKd is not None:
            data = [d for d in data if d.get('pKd', 0) >= filter_pKd]
            print(f"  Filtered to {len(data)} samples with pKd >= {filter_pKd}")

        # Limit samples
        if max_samples is not None:
            data = data[:max_samples]

        self.data = data
        print(f"  Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Tokenize antigen
        antigen_tokens = self.tokenizer.encode(
            sample['antigen_sequence'],
            add_special_tokens=True
        )

        # Tokenize antibody
        antibody_tokens = self.tokenizer.encode_pair(
            sample['antibody_heavy'],
            sample.get('antibody_light', ''),
            add_special_tokens=True
        )

        # Truncate if needed
        if len(antigen_tokens) > self.max_antigen_len:
            antigen_tokens = antigen_tokens[:self.max_antigen_len]

        if len(antibody_tokens) > self.max_antibody_len:
            antibody_tokens = antibody_tokens[:self.max_antibody_len]

        # Get pKd
        pKd = float(sample.get('pKd', 0))

        return {
            'antigen_tokens': antigen_tokens,
            'antibody_tokens': antibody_tokens,
            'pKd': pKd
        }


def collate_fn(batch, pad_token_id=0):
    """Custom collate function for batching"""

    # Get max lengths in this batch
    max_antigen_len = max(len(item['antigen_tokens']) for item in batch)
    max_antibody_len = max(len(item['antibody_tokens']) for item in batch)

    # Pad sequences
    antigen_padded = []
    antibody_padded = []
    pKds = []

    for item in batch:
        # Pad antigen
        antigen = item['antigen_tokens'] + [pad_token_id] * (max_antigen_len - len(item['antigen_tokens']))
        antigen_padded.append(antigen)

        # Pad antibody
        antibody = item['antibody_tokens'] + [pad_token_id] * (max_antibody_len - len(item['antibody_tokens']))
        antibody_padded.append(antibody)

        # pKd
        pKds.append([item['pKd']])

    # Convert to tensors
    return {
        'antigen': torch.LongTensor(antigen_padded),
        'antibody': torch.LongTensor(antibody_padded),
        'pKd': torch.FloatTensor(pKds)
    }


def train_epoch(model, dataloader, optimizer, criterion, device, teacher_forcing_ratio=0.5):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in dataloader:
        # Move to device
        src = batch['antigen'].to(device)
        tgt = batch['antibody'].to(device)
        pKd = batch['pKd'].to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(src, tgt, pKd, teacher_forcing_ratio=teacher_forcing_ratio)

        # Calculate loss (ignore padding)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)  # Skip first token (<START>)
        tgt = tgt[:, 1:].reshape(-1)  # Skip first token

        loss = criterion(output, tgt)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation set"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            src = batch['antigen'].to(device)
            tgt = batch['antibody'].to(device)
            pKd = batch['pKd'].to(device)

            # Forward pass (no teacher forcing for evaluation)
            output = model(src, tgt, pKd, teacher_forcing_ratio=0)

            # Calculate loss
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)

            loss = criterion(output, tgt)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def generate_samples(model, dataloader, tokenizer, device, num_samples=5):
    """Generate sample antibodies"""
    model.eval()
    samples = []

    with torch.no_grad():
        batch = next(iter(dataloader))
        src = batch['antigen'][:num_samples].to(device)
        pKd = batch['pKd'][:num_samples].to(device)
        tgt = batch['antibody'][:num_samples]

        # Generate
        generated = model.generate(src, pKd, max_length=150)

        # Decode
        for i in range(num_samples):
            antigen_seq = tokenizer.decode(src[i].tolist())
            target_seq = tokenizer.decode(tgt[i].tolist())
            generated_seq = tokenizer.decode(generated[i].tolist())

            samples.append({
                'antigen': antigen_seq[:50] + '...',  # Truncate for display
                'target_pKd': pKd[i].item(),
                'target_antibody': target_seq[:50] + '...',
                'generated_antibody': generated_seq[:50] + '...'
            })

    return samples


def train(stage='tiny', resume_from=None):
    """Main training function"""

    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available")
        return

    print("="*70)
    print(f"Training Generative Model - Stage: {stage.upper()}")
    print("="*70)

    # Get configuration
    config = TRAINING_CONFIGS[stage]
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Create output directory
    output_dir = Path(f'models/generative/{stage}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = AminoAcidTokenizer()
    print(f"  Vocabulary size: {tokenizer.vocab_size}")

    # Load data
    print("\nLoading datasets...")
    train_dataset = AbAgDataset(
        'data/generative/train.json',
        tokenizer,
        max_samples=config['n_samples'],
        max_antigen_len=config['max_antigen_len'],
        max_antibody_len=config['max_antibody_len'],
        filter_pKd=7.0 if stage == 'tiny' else None  # Use high-quality for tiny
    )

    val_dataset = AbAgDataset(
        'data/generative/val.json',
        tokenizer,
        max_samples=min(1000, len(train_dataset) // 10),  # 10% or 1k max
        max_antigen_len=config['max_antigen_len'],
        max_antibody_len=config['max_antibody_len']
    )

    # Create dataloaders
    train_loader = TorchDataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id),
        num_workers=0  # Set to 0 to avoid issues on some systems
    )

    val_loader = TorchDataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id),
        num_workers=0
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Create model
    print(f"\nCreating {config['model_config']} model...")
    model = create_model(
        config['model_config'],
        vocab_size=tokenizer.vocab_size,
        max_length=config['max_antibody_len']
    )
    model = model.to(device)

    num_params = model.get_model_size()
    print(f"  Parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Training loop
    print(f"\nTraining for {config['epochs']} epochs...")
    print("-" * 70)

    best_val_loss = float('inf')
    training_history = []

    start_time = time.time()

    for epoch in range(config['epochs']):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            teacher_forcing_ratio=0.5
        )

        # Validate
        val_loss = evaluate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        # Print progress
        print(f"Epoch {epoch+1:3d}/{config['epochs']}: "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Time: {epoch_time:.1f}s")

        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'time': epoch_time
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }, output_dir / 'best_model.pth')
            print(f"  ✅ Saved best model (val_loss: {val_loss:.4f})")

        # Generate samples every 5 epochs
        if (epoch + 1) % 5 == 0:
            print("\n  Sample generations:")
            samples = generate_samples(model, val_loader, tokenizer, device, num_samples=3)
            for i, sample in enumerate(samples):
                print(f"    {i+1}. pKd={sample['target_pKd']:.2f}")
                print(f"       Generated: {sample['generated_antibody']}")
            print()

    total_time = time.time() - start_time

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'training_history': training_history
    }, output_dir / 'final_model.pth')

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)

    # Summary
    print("\n" + "="*70)
    print("✅ Training Complete!")
    print("="*70)
    print(f"\nSummary:")
    print(f"  Stage: {stage.upper()}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Final train loss: {train_loss:.4f}")
    print(f"  Model saved to: {output_dir}")
    print(f"\nNext steps:")
    if stage == 'tiny':
        print(f"  ✅ Verify model works")
        print(f"  → Run: python scripts/train_generative.py --stage small")
    elif stage == 'small':
        print(f"  ✅ Check generated sequence quality")
        print(f"  → If good: python scripts/train_generative.py --stage full")
        print(f"  → If bad: debug and iterate")
    else:
        print(f"  ✅ Evaluate on test set")
        print(f"  → Validate with discriminator")
        print(f"  → Deploy for production use")

    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train generative model')
    parser.add_argument('--stage', type=str, default='tiny',
                      choices=['tiny', 'small', 'full'],
                      help='Training stage (tiny/small/full)')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume from')

    args = parser.parse_args()

    train(stage=args.stage, resume_from=args.resume)
