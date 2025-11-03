"""
Training Script for Antibody Generation Model

Trains Transformer seq2seq model on antibody-antigen pairs
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from generators.tokenizer import AminoAcidTokenizer
from generators.data_loader import AbAgDataset, DataLoader
from generators.transformer_seq2seq import create_model
from generators.metrics import SequenceMetrics, TrainingLogger


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """
    Create learning rate scheduler with linear warmup and cosine decay

    This is the standard LR schedule used in modern transformer training (GPT, BERT, etc.)

    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: number of steps for linear warmup
        num_training_steps: total number of training steps
        min_lr_ratio: minimum LR as ratio of initial LR (default: 0.1)

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step):
        # Linear warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay
        progress = (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Scale to [min_lr_ratio, 1.0]
        return max(min_lr_ratio, cosine_decay)

    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    """
    Handles training loop for antibody generation model
    """

    def __init__(self,
                 model,
                 tokenizer,
                 train_loader,
                 val_loader,
                 device='cpu',
                 learning_rate=1e-4,
                 num_epochs=10,
                 experiment_name='transformer'):
        """
        Initialize trainer

        Args:
            model: TransformerSeq2Seq model
            tokenizer: AminoAcidTokenizer
            train_loader: training data loader
            val_loader: validation data loader
            device: 'cpu' or 'cuda'
            learning_rate: initial learning rate
            num_epochs: number of epochs (for scheduler calculation)
            experiment_name: name for logs
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.experiment_name = experiment_name

        # Loss function with label smoothing (2024 best practice)
        # Label smoothing prevents overconfidence and improves generalization
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_token_id,
            label_smoothing=0.1
        )

        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Learning rate scheduler with warmup + cosine decay (2024 best practice)
        # Used in GPT-3, BERT, and all modern LLM training
        num_training_steps = num_epochs * len(train_loader)
        num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_ratio=0.1
        )

        print(f"   LR Schedule: {num_warmup_steps} warmup steps, {num_training_steps} total steps")

        # Metrics and logging
        self.metrics = SequenceMetrics(tokenizer)
        self.logger = TrainingLogger(log_dir='logs', experiment_name=experiment_name)

        # Tracking
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

    def train_epoch(self):
        """
        Train for one epoch

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        for step, batch in enumerate(self.train_loader):
            # Convert to tensors and move to device
            src = torch.tensor(batch['antigen_tokens']).to(self.device)
            tgt = torch.tensor(batch['antibody_tokens']).to(self.device)
            pKd = torch.tensor(batch['pKd']).unsqueeze(1).float().to(self.device)

            # Forward pass
            output = self.model(src, tgt, pKd)

            # Compute loss
            # Shift targets (predict next token)
            output_flat = output[:, :-1, :].reshape(-1, output.size(-1))
            tgt_flat = tgt[:, 1:].reshape(-1)

            loss = self.criterion(output_flat, tgt_flat)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (prevents gradient explosion)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Update learning rate (per-step for cosine schedule with warmup)
            self.scheduler.step()

            # Track loss
            total_loss += loss.item()
            num_batches += 1

            # Log periodically
            if step % 100 == 0:
                self.logger.log_train_step(step, loss.item(), batch['batch_size'])

        return total_loss / num_batches

    def validate(self):
        """
        Validate model

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # Convert to tensors
                src = torch.tensor(batch['antigen_tokens']).to(self.device)
                tgt = torch.tensor(batch['antibody_tokens']).to(self.device)
                pKd = torch.tensor(batch['pKd']).unsqueeze(1).float().to(self.device)

                # Forward pass
                output = self.model(src, tgt, pKd)

                # Compute loss
                output_flat = output[:, :-1, :].reshape(-1, output.size(-1))
                tgt_flat = tgt[:, 1:].reshape(-1)

                loss = self.criterion(output_flat, tgt_flat)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def generate_and_evaluate(self, num_samples=100):
        """
        Generate samples and evaluate quality

        Args:
            num_samples: number of samples to generate

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        generated_sequences = []
        sample_info = []

        with torch.no_grad():
            # Take first few batches
            samples_generated = 0

            for batch in self.val_loader:
                if samples_generated >= num_samples:
                    break

                # Get antigens and target pKd
                src = torch.tensor(batch['antigen_tokens']).to(self.device)
                pKd = torch.tensor(batch['pKd']).unsqueeze(1).float().to(self.device)

                # Take only what we need
                batch_size = min(src.size(0), num_samples - samples_generated)
                src = src[:batch_size]
                pKd = pKd[:batch_size]

                # Generate
                generated = self.model.generate_greedy(src, pKd, max_length=300)

                # Decode sequences
                for i in range(batch_size):
                    seq = self.tokenizer.decode(generated[i].cpu().tolist())
                    generated_sequences.append(seq)
                    sample_info.append((seq, pKd[i].item()))

                samples_generated += batch_size

        # Evaluate
        eval_results = self.metrics.evaluate_batch(generated_sequences)

        # Add sample info
        eval_results['samples'] = sample_info[:5]  # Keep first 5 for logging

        return eval_results

    def load_checkpoint(self, checkpoint_path):
        """
        Load checkpoint to resume training

        Args:
            checkpoint_path: path to checkpoint file

        Returns:
            epoch number to resume from
        """
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        epoch = checkpoint['epoch']
        val_loss = checkpoint['val_loss']

        print(f"✅ Resumed from epoch {epoch}, val_loss={val_loss:.4f}")
        return epoch

    def save_checkpoint(self, epoch, val_loss, checkpoint_dir='checkpoints', is_best=False):
        """
        Save model checkpoint

        Args:
            epoch: current epoch
            val_loss: validation loss
            checkpoint_dir: directory to save checkpoint
            is_best: if True, also save as best model
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)

        # Create checkpoint with all necessary state
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'val_loss': val_loss,
            'config': {
                'vocab_size': self.tokenizer.vocab_size,
                'd_model': self.model.d_model,
            }
        }

        # Save epoch checkpoint
        checkpoint_path = checkpoint_dir / f"{self.experiment_name}_epoch{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")

        # Also save as "latest" for easy resuming
        latest_path = checkpoint_dir / f"{self.experiment_name}_latest.pt"
        torch.save(checkpoint, latest_path)

        # Save as best model if this is the best so far
        if is_best:
            best_path = checkpoint_dir / f"{self.experiment_name}_best.pt"
            torch.save(checkpoint, best_path)
            print(f"⭐ Best model saved: {best_path}")

        return str(checkpoint_path)

    def train(self, num_epochs=10, eval_interval=1, save_best=True, early_stopping_patience=5, start_epoch=0):
        """
        Main training loop

        Args:
            num_epochs: number of epochs to train
            eval_interval: evaluate every N epochs
            save_best: save checkpoint when validation improves
            early_stopping_patience: stop if no improvement for N epochs
            start_epoch: epoch to start from (for resuming)
        """
        print("="*70)
        print(f"Starting Training: {self.experiment_name}")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.get_model_size():,}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        if start_epoch > 0:
            print(f"Resuming from epoch: {start_epoch}")
        print("="*70)

        for epoch in range(start_epoch + 1, num_epochs + 1):
            self.logger.log_epoch_start(epoch)

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.validate()

            # Evaluate generation quality (every eval_interval epochs)
            val_metrics = None
            if epoch % eval_interval == 0:
                print("\nEvaluating generation quality...")
                val_metrics = self.generate_and_evaluate(num_samples=100)

                # Log sample generations
                if 'samples' in val_metrics:
                    self.logger.log_generation_samples(epoch, val_metrics['samples'])

            # Log epoch
            self.logger.log_epoch_end(epoch, train_loss, val_loss, val_metrics)

            # Note: LR is updated per-step (in train_epoch), not per-epoch

            # Check if this is the best model so far
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                print(f"✅ New best validation loss: {val_loss:.4f}")
            else:
                self.epochs_without_improvement += 1

            # Save checkpoint EVERY epoch (not just best)
            checkpoint_path = self.save_checkpoint(epoch, val_loss, is_best=is_best)

            # Save checkpoint info
            if val_metrics:
                self.logger.save_checkpoint_info(epoch, checkpoint_path, val_metrics)

            # Early stopping
            if self.epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered (no improvement for {early_stopping_patience} epochs)")
                break

        # Training complete
        self.logger.print_summary()

        return self.best_val_loss


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train antibody generation model')

    # Model args
    parser.add_argument('--config', type=str, default='small',
                       choices=['tiny', 'small', 'medium', 'large'],
                       help='Model configuration')

    # Data args
    parser.add_argument('--train-data', type=str, default='data/generative/train.json',
                       help='Path to training data')
    parser.add_argument('--val-data', type=str, default='data/generative/val.json',
                       help='Path to validation data')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Limit number of training samples (for testing)')

    # Training args
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device (cpu or cuda)')
    parser.add_argument('--eval-interval', type=int, default=1,
                       help='Evaluate every N epochs')
    parser.add_argument('--early-stopping', type=int, default=5,
                       help='Early stopping patience')

    # Experiment args
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Resume training from checkpoint (path to .pt file)')

    args = parser.parse_args()

    # Set experiment name
    if args.name is None:
        args.name = f"{args.config}_{int(time.time())}"

    print(f"\n{'='*70}")
    print(f"Experiment: {args.name}")
    print(f"{'='*70}")

    # Initialize tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AminoAcidTokenizer()
    print(f"   Vocab size: {tokenizer.vocab_size}")

    # Load data
    print("\n2. Loading datasets...")
    train_dataset = AbAgDataset(args.train_data, tokenizer)
    val_dataset = AbAgDataset(args.val_data, tokenizer)

    # Limit samples if specified (for testing)
    if args.max_samples:
        train_dataset.data = train_dataset.data[:args.max_samples]
        val_dataset.data = val_dataset.data[:min(args.max_samples // 10, len(val_dataset.data))]
        print(f"   Limited to {len(train_dataset)} training samples")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"   Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"   Val:   {len(val_dataset)} samples, {len(val_loader)} batches")

    # Create model
    print("\n3. Creating model...")
    model = create_model(
        args.config,
        vocab_size=tokenizer.vocab_size,
        max_src_len=512,
        max_tgt_len=300
    )
    print(f"   Config: {args.config}")
    print(f"   Parameters: {model.get_model_size():,}")

    # Create trainer
    print("\n4. Initializing trainer...")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        experiment_name=args.name
    )
    print(f"   Device: {args.device}")
    print(f"   Learning rate: {args.lr}")

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume_from:
        print(f"\n📁 Resuming from checkpoint: {args.resume_from}")
        start_epoch = trainer.load_checkpoint(args.resume_from)
        print(f"   Will continue from epoch {start_epoch + 1}")

    # Train
    print("\n5. Starting training...")
    best_val_loss = trainer.train(
        num_epochs=args.epochs,
        eval_interval=args.eval_interval,
        save_best=True,
        early_stopping_patience=args.early_stopping,
        start_epoch=start_epoch
    )

    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Logs: logs/{args.name}.jsonl")
    print(f"Checkpoints: checkpoints/{args.name}_epoch*.pt")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
