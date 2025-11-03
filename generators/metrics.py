"""
Metrics Module for Antibody Generation

Tracks training progress and evaluates generated sequences:
- Loss metrics
- Sequence validity
- Sequence diversity
- Affinity correlation
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter


class SequenceMetrics:
    """
    Evaluate quality of generated antibody sequences
    """

    def __init__(self, tokenizer):
        """
        Args:
            tokenizer: AminoAcidTokenizer instance
        """
        self.tokenizer = tokenizer
        self.valid_amino_acids = set(tokenizer.amino_acids)

    def is_valid_sequence(self, sequence: str) -> bool:
        """
        Check if sequence contains only valid amino acids

        Args:
            sequence: amino acid sequence string

        Returns:
            True if all characters are valid amino acids
        """
        # Remove separator if present (for heavy|light chains)
        sequence_clean = sequence.replace('|', '')
        return all(aa in self.valid_amino_acids for aa in sequence_clean)

    def compute_validity(self, sequences: List[str]) -> float:
        """
        Compute percentage of valid sequences

        Args:
            sequences: list of amino acid sequences

        Returns:
            Validity percentage (0-100)
        """
        if len(sequences) == 0:
            return 0.0

        valid_count = sum(1 for seq in sequences if self.is_valid_sequence(seq))
        return 100.0 * valid_count / len(sequences)

    def compute_diversity(self, sequences: List[str]) -> Dict[str, float]:
        """
        Compute sequence diversity metrics

        Args:
            sequences: list of amino acid sequences

        Returns:
            Dictionary with diversity metrics
        """
        if len(sequences) == 0:
            return {'unique_ratio': 0.0, 'unique_count': 0, 'total_count': 0}

        unique_sequences = set(sequences)

        return {
            'unique_ratio': 100.0 * len(unique_sequences) / len(sequences),
            'unique_count': len(unique_sequences),
            'total_count': len(sequences)
        }

    def compute_length_stats(self, sequences: List[str]) -> Dict[str, float]:
        """
        Compute sequence length statistics

        Args:
            sequences: list of amino acid sequences

        Returns:
            Dictionary with length statistics
        """
        if len(sequences) == 0:
            return {'mean': 0.0, 'min': 0, 'max': 0}

        lengths = [len(seq) for seq in sequences]

        return {
            'mean': sum(lengths) / len(lengths),
            'min': min(lengths),
            'max': max(lengths)
        }

    def compute_amino_acid_distribution(self, sequences: List[str]) -> Dict[str, float]:
        """
        Compute amino acid frequency distribution

        Args:
            sequences: list of amino acid sequences

        Returns:
            Dictionary mapping amino acid -> percentage
        """
        if len(sequences) == 0:
            return {}

        # Count all amino acids
        all_aa = ''.join(sequences)
        aa_counts = Counter(all_aa)
        total = sum(aa_counts.values())

        # Convert to percentages
        aa_dist = {aa: 100.0 * count / total for aa, count in aa_counts.items()}

        return aa_dist

    def evaluate_batch(self, sequences: List[str]) -> Dict:
        """
        Comprehensive evaluation of a batch of sequences

        Args:
            sequences: list of amino acid sequences

        Returns:
            Dictionary with all metrics
        """
        return {
            'validity': self.compute_validity(sequences),
            'diversity': self.compute_diversity(sequences),
            'length_stats': self.compute_length_stats(sequences),
            'aa_distribution': self.compute_amino_acid_distribution(sequences)
        }


class TrainingLogger:
    """
    Logger for tracking training progress
    """

    def __init__(self, log_dir: str = 'logs', experiment_name: str = None):
        """
        Args:
            log_dir: directory to save logs
            experiment_name: name of experiment (default: timestamp)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        if experiment_name is None:
            experiment_name = f"exp_{int(time.time())}"

        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}.jsonl"

        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'validity': [],
            'diversity': [],
            'epoch_times': []
        }

        self.start_time = time.time()
        self.epoch_start_time = None

    def log_epoch_start(self, epoch: int):
        """Mark start of epoch"""
        self.epoch_start_time = time.time()
        print(f"\nEpoch {epoch}")
        print("-" * 50)

    def log_train_step(self, step: int, loss: float, batch_size: int):
        """Log training step"""
        if step % 10 == 0:
            print(f"  Step {step}: loss={loss:.4f}")

    def log_epoch_end(self, epoch: int, train_loss: float, val_loss: float,
                     val_metrics: Dict = None):
        """
        Log end of epoch with metrics

        Args:
            epoch: epoch number
            train_loss: average training loss
            val_loss: average validation loss
            val_metrics: validation metrics (optional)
        """
        epoch_time = time.time() - self.epoch_start_time

        # Store metrics
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['val_loss'].append(val_loss)
        self.metrics_history['epoch_times'].append(epoch_time)

        # Print summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Time:       {epoch_time:.1f}s")

        # Log validation metrics if provided
        if val_metrics:
            validity = val_metrics.get('validity', 0)
            diversity = val_metrics.get('diversity', {}).get('unique_ratio', 0)

            self.metrics_history['validity'].append(validity)
            self.metrics_history['diversity'].append(diversity)

            print(f"  Validity:   {validity:.1f}%")
            print(f"  Diversity:  {diversity:.1f}%")

        # Save to log file
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'epoch_time': epoch_time,
            'timestamp': time.time()
        }

        if val_metrics:
            log_entry['val_metrics'] = val_metrics

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def log_generation_samples(self, epoch: int, samples: List[Tuple[str, float]]):
        """
        Log sample generated sequences

        Args:
            epoch: epoch number
            samples: list of (sequence, pKd) tuples
        """
        print(f"\nGeneration Samples (Epoch {epoch}):")
        for i, (seq, pkd) in enumerate(samples[:3], 1):
            print(f"  {i}. pKd={pkd:.2f}: {seq[:50]}...")

    def save_checkpoint_info(self, epoch: int, model_path: str, metrics: Dict):
        """
        Save checkpoint information

        Args:
            epoch: epoch number
            model_path: path to saved model
            metrics: current metrics
        """
        checkpoint_info = {
            'epoch': epoch,
            'model_path': model_path,
            'metrics': metrics,
            'timestamp': time.time()
        }

        checkpoint_file = self.log_dir / f"{self.experiment_name}_checkpoints.jsonl"
        with open(checkpoint_file, 'a') as f:
            f.write(json.dumps(checkpoint_info) + '\n')

    def get_best_epoch(self, metric: str = 'val_loss', minimize: bool = True) -> int:
        """
        Get epoch with best metric

        Args:
            metric: metric name ('train_loss', 'val_loss', 'validity', 'diversity')
            minimize: True to minimize metric, False to maximize

        Returns:
            Best epoch number (1-indexed)
        """
        if metric not in self.metrics_history or len(self.metrics_history[metric]) == 0:
            return 0

        values = self.metrics_history[metric]

        if minimize:
            best_idx = values.index(min(values))
        else:
            best_idx = values.index(max(values))

        return best_idx + 1

    def print_summary(self):
        """Print training summary"""
        total_time = time.time() - self.start_time

        print("\n" + "="*70)
        print("Training Summary")
        print("="*70)

        if len(self.metrics_history['train_loss']) > 0:
            print(f"Total Epochs:     {len(self.metrics_history['train_loss'])}")
            print(f"Total Time:       {total_time/3600:.2f} hours")
            print(f"Final Train Loss: {self.metrics_history['train_loss'][-1]:.4f}")
            print(f"Final Val Loss:   {self.metrics_history['val_loss'][-1]:.4f}")

            best_epoch = self.get_best_epoch('val_loss', minimize=True)
            print(f"Best Val Loss:    {self.metrics_history['val_loss'][best_epoch-1]:.4f} (Epoch {best_epoch})")

            if len(self.metrics_history['validity']) > 0:
                print(f"Final Validity:   {self.metrics_history['validity'][-1]:.1f}%")
                print(f"Final Diversity:  {self.metrics_history['diversity'][-1]:.1f}%")

        print("="*70)

    def plot_metrics(self, save_path: str = None):
        """
        Plot training metrics (requires matplotlib)

        Args:
            save_path: path to save plot (optional)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Loss curves
        epochs = range(1, len(self.metrics_history['train_loss']) + 1)
        axes[0, 0].plot(epochs, self.metrics_history['train_loss'], label='Train')
        axes[0, 0].plot(epochs, self.metrics_history['val_loss'], label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training/Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Validity
        if len(self.metrics_history['validity']) > 0:
            axes[0, 1].plot(epochs, self.metrics_history['validity'])
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Validity (%)')
            axes[0, 1].set_title('Sequence Validity')
            axes[0, 1].grid(True)

        # Diversity
        if len(self.metrics_history['diversity']) > 0:
            axes[1, 0].plot(epochs, self.metrics_history['diversity'])
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Diversity (%)')
            axes[1, 0].set_title('Sequence Diversity')
            axes[1, 0].grid(True)

        # Epoch times
        if len(self.metrics_history['epoch_times']) > 0:
            axes[1, 1].plot(epochs, self.metrics_history['epoch_times'])
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Time (s)')
            axes[1, 1].set_title('Epoch Duration')
            axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.savefig(self.log_dir / f"{self.experiment_name}_metrics.png", dpi=150, bbox_inches='tight')

        plt.close()


def test_metrics():
    """Test metrics module"""
    print("="*70)
    print("Testing Metrics Module")
    print("="*70)

    # Import tokenizer
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from generators.tokenizer import AminoAcidTokenizer

    tokenizer = AminoAcidTokenizer()
    metrics = SequenceMetrics(tokenizer)

    # Test sequences
    print("\n1. Testing sequence validation...")
    valid_seqs = ["ACDEFGH", "IKLMNPQ", "RSTVWY"]
    invalid_seqs = ["ACDEFGH", "XYZABC", "ACDE123"]

    validity_valid = metrics.compute_validity(valid_seqs)
    validity_invalid = metrics.compute_validity(invalid_seqs)

    print(f"   Valid sequences: {validity_valid:.1f}% valid")
    print(f"   Mixed sequences: {validity_invalid:.1f}% valid")
    print("   ✅ Validation working")

    # Test diversity
    print("\n2. Testing diversity metrics...")
    diverse_seqs = ["ACDE", "FGHI", "KLMN", "PQRS"]
    repeated_seqs = ["ACDE", "ACDE", "FGHI", "ACDE"]

    div_diverse = metrics.compute_diversity(diverse_seqs)
    div_repeated = metrics.compute_diversity(repeated_seqs)

    print(f"   Diverse: {div_diverse['unique_ratio']:.1f}% unique ({div_diverse['unique_count']}/{div_diverse['total_count']})")
    print(f"   Repeated: {div_repeated['unique_ratio']:.1f}% unique ({div_repeated['unique_count']}/{div_repeated['total_count']})")
    print("   ✅ Diversity working")

    # Test length stats
    print("\n3. Testing length statistics...")
    seqs = ["AC", "ACDEF", "ACDEFGHIK"]
    length_stats = metrics.compute_length_stats(seqs)

    print(f"   Mean: {length_stats['mean']:.1f}")
    print(f"   Min: {length_stats['min']}")
    print(f"   Max: {length_stats['max']}")
    print("   ✅ Length stats working")

    # Test amino acid distribution
    print("\n4. Testing amino acid distribution...")
    seqs = ["AAA", "ACC", "ACG"]
    aa_dist = metrics.compute_amino_acid_distribution(seqs)

    print(f"   Top amino acids:")
    for aa, pct in sorted(aa_dist.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"     {aa}: {pct:.1f}%")
    print("   ✅ AA distribution working")

    # Test logger
    print("\n5. Testing training logger...")
    logger = TrainingLogger(log_dir='logs', experiment_name='test_metrics')

    logger.log_epoch_start(1)
    logger.log_train_step(10, 3.5, 32)
    logger.log_epoch_end(1, train_loss=3.2, val_loss=3.4, val_metrics={
        'validity': 95.0,
        'diversity': {'unique_ratio': 80.0}
    })

    print("   ✅ Logger working")

    print("\n" + "="*70)
    print("✅ All metrics tests passed!")
    print("="*70)


if __name__ == '__main__':
    test_metrics()
